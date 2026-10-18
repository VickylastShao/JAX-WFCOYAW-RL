"""Bounded N81 computation profiling through the existing LES/SAC trainer.

This is a computational diagnostic, never a scientific policy run. It starts
unburned states at an explicit synthetic episode clock, uses fixed collector
actions, and measures real SAC work, LES collection/reset, synchronization,
and full-state I/O. Scientific trainer sources are not changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time


SCHEMA = "paper1_n81_bounded_computation_profile_v2"
SEED = 28101181
GLOBAL_ENVS = 32
GLOBAL_SAMPLE = 256
COLLECTION = 8
WARM_BLOCKS = 11
CONTROLS = (WARM_BLOCKS + 1) * COLLECTION
CLOCK_OFFSET = 500 - CONTROLS
REPLAY = 262144
MAX_YAW_DEG = 40.0
YAW_OBSERVATION_SCALE = 4.0
TARGET_YAW_DEG = 20.0
CONTROL_DT_SECONDS = 10.0
MAX_YAW_SPEED_DEG_PER_S = 1.0
TOPOLOGY_GPU_HOUR_CAP = 5.0
TOTAL_GPU_HOUR_CAP = 15.0
CHECKPOINT_SAVE_RESERVE_SECONDS = 600.0
BOUNDARY = (
    "Bounded computational profile: no initial burnin; synthetic episode clock "
    "404; fixed deterministic collector actions; real SAC updates do not drive "
    "the collector. Includes physical reset150, synchronized collection/learning "
    "and full checkpoint I/O. Not an end-to-end speedup of the scientific "
    "500-control training protocol, not a learned controller or gain result. "
    "Learner sample counts match; stochastic minibatch realizations and "
    "floating-point reduction order are not promised bitwise identical."
)


def fixed_yaw_rate_action(normalized_yaw, target_yaw_deg, *, clip):
    """Pure scalar/array calculation shared by Python/NumPy tests and JAX.

    stacked_observation encodes yaw as degrees * 4 / max_yaw_angle_deg.
    The per_turbine action is a normalized rate, not an absolute yaw target.
    """
    previous_yaw_deg = normalized_yaw * (MAX_YAW_DEG / YAW_OBSERVATION_SCALE)
    max_increment = MAX_YAW_SPEED_DEG_PER_S * CONTROL_DT_SECONDS
    return clip((target_yaw_deg - previous_yaw_deg) / max_increment, -1.0, 1.0)


def fixed_action_evidence(rows):
    """Check actual warm-block physical metrics, not collector intent alone."""
    evidence = [{"iteration": row.get("iteration"),
                 "mean_abs_yaw_deg": row.get("mean_abs_yaw_deg"),
                 "mean_abs_yaw_speed_deg_per_s": row.get("mean_abs_yaw_speed_deg_per_s")}
                for row in rows]

    def matches(key, expected, tolerance):
        values = [row[key] for row in evidence]
        return len(values) == WARM_BLOCKS and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            and abs(value - expected) <= tolerance for value in values)

    return evidence, {
        "actual_warm_yaw_matches_fixed_20_degree_target": matches("mean_abs_yaw_deg", TARGET_YAW_DEG, 1.0e-4),
        "actual_warm_executed_yaw_rate_is_zero_at_target": matches("mean_abs_yaw_speed_deg_per_s", 0.0, 1.0e-6),
    }


def validate_contract(protocol):
    expected = contract(protocol["global_environments"])
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("profile work contract/schema changed; v1 and v2 results cannot be combined")
    return expected


def checkpoint_completion_checks(checkpoint, io_records, saved_states, barriers,
                                 actor_records, *, rank, topology, local_environments,
                                 global_interactions, expected_final_cursors):
    """Require the final full state on every rank, rather than any rolling save."""
    final_path = checkpoint.get("full_state_process_local")
    records = [record for record in io_records if record["path"] == final_path]
    states = [record for record in saved_states if record["path"] == final_path]
    final = records[-1] if records else {}
    state = states[-1] if states else {}
    required_keys = {"actor_states", "plane_source_states", "plane_cursors", "learner",
                     "configuration", "history", "process_id", "num_processes",
                     "global_interactions", "completed_collector_iterations_including_warmup"}
    return {
        "final_complete_checkpoint_saved_and_hashed": bool(final) and final.get("bytes", 0) > 0
            and len(final.get("sha256", "")) == 64
            and final.get("sha256") == checkpoint.get("full_state_sha256"),
        "final_checkpoint_payload_matches_exact_work": bool(final)
            and final.get("global_interactions") == global_interactions
            and final.get("completed_collector_blocks") == WARM_BLOCKS + 1
            and final.get("history_blocks") == WARM_BLOCKS
            and final.get("process_id") == rank and final.get("num_processes") == topology
            and required_keys.issubset(final.get("payload_keys", [])),
        "final_checkpoint_contains_complete_local_environment_roster": bool(state)
            and len(state.get("local_les_steps", [])) == local_environments
            and len(state.get("local_transition_clocks", [])) == local_environments
            and len(state.get("local_final_yaw_max_abs_deg", [])) == local_environments,
        "final_checkpoint_cursors_match_frozen_archive_work": bool(state)
            and state.get("plane_cursors") == expected_final_cursors,
        "real_les_step_count_in_complete_state": bool(state) and bool(state.get("local_les_steps"))
            and all(step == (CONTROLS + 150) * 50 for step in state["local_les_steps"]),
        "synthetic_clock_is_separate_from_reported_scientific_interactions": bool(state)
            and bool(state.get("local_transition_clocks"))
            and all(count == CLOCK_OFFSET + CONTROLS for count in state["local_transition_clocks"]),
        "final_checkpoint_learner_replay_and_updates_match_work": bool(state)
            and state.get("learner_updates") == global_interactions
            and state.get("learner_replay_size") == global_interactions,
        "final_physical_reset_zeroed_yaw": bool(state)
            and bool(state.get("local_final_yaw_max_abs_deg"))
            and all(math.isfinite(value) and value <= 1.0e-6
                    for value in state["local_final_yaw_max_abs_deg"]),
        "final_checkpoint_global_barrier_completed": bool(final) and any(
            barrier.get("name") == "microbatch-checkpoint-written"
            and barrier.get("completed") is True
            and barrier.get("started_seconds", -1.0) >= final.get("finished_seconds", float("inf"))
            for barrier in barriers),
        "final_rank0_actor_checkpoint_saved_and_hashed": rank != 0 or any(
            record.get("path") == checkpoint.get("path_process0_only")
            and record.get("global_interactions") == global_interactions
            and record.get("bytes", 0) > 0 and len(record.get("sha256", "")) == 64
            for record in actor_records),
    }


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replace_arg(argv, name, value):
    result = list(argv)
    if name in result:
        result[result.index(name) + 1] = str(value)
    else:
        result.extend((name, str(value)))
    return result


def contract(global_environments=GLOBAL_ENVS):
    if type(global_environments) is not int or global_environments < 4 or global_environments % 4:
        raise ValueError("common global E must be a positive multiple of four")
    return {
        "schema": SCHEMA, "purpose": "computational_diagnostic_only",
        "target_turbines": 81, "global_environments": global_environments,
        "global_sample_size": GLOBAL_SAMPLE, "collection_steps": COLLECTION,
        "warm_loop_blocks": WARM_BLOCKS, "retained_cold_blocks": 1,
        "initial_episode_clock_offset": CLOCK_OFFSET,
        "initial_burnin_interactions": 0, "episode_interactions": 500,
        "reset_interactions": 150, "resets_per_environment": 1,
        "controls_per_environment": CONTROLS,
        "global_collected_interactions": global_environments * CONTROLS,
        "global_learner_updates": global_environments * CONTROLS,
        "global_training_examples": global_environments * CONTROLS * GLOBAL_SAMPLE,
        "les_steps_per_control": 50,
        "les_steps_over_all_environments": global_environments * (CONTROLS + 150) * 50,
        "planes_consumed_per_environment": (CONTROLS + 150) * 50,
        "replay_capacity_per_rank": REPLAY, "microbatch": 1,
        "seed": SEED, "source_role": "train_A_only",
        "total_gpu_hour_cap": TOTAL_GPU_HOUR_CAP, "topology_gpu_hour_cap": TOPOLOGY_GPU_HOUR_CAP,
        "checkpoint_save_reserve_seconds": CHECKPOINT_SAVE_RESERVE_SECONDS,
        "automatic_retry": False,
        "topologies": [1, 2, 4],
        "fixed_actions": "alternating turbine target yaw +/-20deg; decode yaw_deg=normalized_yaw*40/4; normalized rate clip((target_deg-yaw_deg)/10,-1,1)",
        "fixed_action_contract": {"target_yaw_abs_deg": TARGET_YAW_DEG,
                                  "max_yaw_angle_deg": MAX_YAW_DEG,
                                  "yaw_observation_scale": YAW_OBSERVATION_SCALE,
                                  "control_dt_seconds": CONTROL_DT_SECONDS,
                                  "max_yaw_speed_deg_per_s": MAX_YAW_SPEED_DEG_PER_S,
                                  "action_parameterization": "per_turbine",
                                  "expected_warm_mean_abs_yaw_deg": TARGET_YAW_DEG,
                                  "expected_warm_mean_abs_yaw_speed_deg_per_s": 0.0,
                                  "expected_final_yaw_after_reset_deg": 0.0},
        "global_environment_seed_rule": "seed+global_environment; independent of topology",
        "evidence_boundary": BOUNDARY,
        "memory_admission": {
            "state_reference_bytes": 378598470,
            "reference": "N81 full-state synthetic-inlet runtime probe; total file size used as a conservative per-environment persistent-state estimate",
            "replay_bytes": REPLAY * (2 * 6318 + 81 + 2) * 4,
            "reference_peak_bytes": 8820619264,
            "reference_largest_allocation_bytes": 6720384000,
            "require_actual_peak_per_topology": True,
            "learner_sample256_peak_not_yet_measured": True,
            "capacity_failure": "preserve failed evidence; freeze a smaller common global E before restarting all three topologies; charge all v2 attempts to the same prospective 15 GPUh cap; no automatic retry; v1 costs remain separately reported",
        },
    }


def rank_seed(base_seed, process_id, local_environments, local_index):
    if not 0 <= local_index < local_environments:
        raise ValueError("local initializer index outside frozen topology")
    return base_seed + process_id * local_environments + local_index


def profile_horizon(cursors, completed_planes, global_environments=GLOBAL_ENVS):
    value = contract(global_environments)
    ends = [int(c) + value["planes_consumed_per_environment"] for c in cursors]
    required = max(ends)
    if not cursors or any(c < 0 for c in cursors) or required > completed_planes:
        raise ValueError("bounded profile lacks the complete reset/collection input horizon")
    return {"schema": SCHEMA + "_input_horizon", "initial_cursors": list(cursors),
            "final_cursors_exclusive": ends, "required_archive_end_exclusive": required,
            "available_planes": completed_planes, "headroom_planes": completed_planes - required,
            "controls_per_environment": CONTROLS, "resets_per_environment": 1,
            "initial_burnin_interactions": 0, "initial_episode_clock_offset": CLOCK_OFFSET,
            "global_interactions_start": 0, "global_interactions_end": value["global_collected_interactions"],
            "retained_compile_blocks": 1, "planes_consumed_per_environment": value["planes_consumed_per_environment"]}


def profile_argv(original, *, topology, rank, root, global_environments=GLOBAL_ENVS):
    value = contract(global_environments)
    if topology not in value["topologies"] or not 0 <= rank < topology:
        raise ValueError("invalid profile topology/rank")
    local = global_environments // topology
    directory = Path(root) / ("p%d" % topology)
    argv = list(original)
    if "--save-initial-checkpoint" in argv:
        argv.remove("--save-initial-checkpoint")
    overrides = {
        "--coordinator": "127.0.0.1:%d" % (17520 + topology),
        "--process-id": rank, "--num-processes": topology,
        "--local-environments": local, "--sample-size": GLOBAL_SAMPLE // topology,
        "--plane-archive": str(directory / "archive"), "--plane-offset-add": rank * local * 20000,
        "--burnin-interactions": 0, "--iterations": WARM_BLOCKS,
        "--planned-global-interactions": value["global_collected_interactions"],
        "--minimum-replay-size": global_environments * COLLECTION,
        "--random-action-transitions": 0, "--graph-actor-minimum-replay-size": 0,
        "--graph-actor-warmup-updates": 0, "--seed": SEED,
        "--checkpoint": str(directory / "profile_actor.pkl"),
        "--full-checkpoint": str(directory / "profile_full_state.pkl"),
        "--checkpoint-milestones": "", "--full-checkpoint-interval": 0,
        "--stop-request": str(directory / "STOP"),
        # Prospective cap; reserve host conversion, save/hash, barrier and output.
        "--stop-after-seconds": value["topology_gpu_hour_cap"] * 3600 / topology - value["checkpoint_save_reserve_seconds"],
        "--output": str(directory / ("rank%d.trainer.json" % rank)),
    }
    for name, data in overrides.items():
        argv = replace_arg(argv, name, data)
    return argv


def prepare(args):
    source = args.source_archive
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("dtype") != "float32":
        raise ValueError("complete training-A fp32 archive required")
    provenance = manifest.get("provenance", {})
    if provenance.get("target_turbines") != 81 or manifest.get("plane_shape") != [3, 41, 483]:
        raise ValueError("profile requires the real N81 inlet geometry")
    if (source / "extension_manifest.json").exists():
        raise ValueError("profile view preparation currently requires an unextended base archive")
    recipe = json.loads(args.recipes.read_text())["training_runs"][0]
    original = recipe["ranks"][0]["argv"]
    value = contract(args.global_environments)
    value.update(source_archive=str(source), source_manifest_sha256=sha(source / "manifest.json"),
                 original_trainer_argv=original, wrapper_sha256=sha(__file__))
    ranks = []
    for topology in value["topologies"]:
        local = args.global_environments // topology
        directory = args.output_root / ("p%d" % topology)
        view = copy.deepcopy(manifest)
        view["member_offsets"] = [(i + 1) * 20000 for i in range(local)]
        for chunk in view["chunks"]:
            # Absolute container-visible paths point at unchanged, read-only chunks.
            chunk["path"] = str(source / chunk["path"])
        view["computation_profile_view"] = {
            "source_manifest_sha256": value["source_manifest_sha256"],
            "chunk_payloads_unchanged": True, "purpose": "topology-independent global cursor roster",
        }
        write_new(directory / "archive" / "manifest.json", view)
        horizon = profile_horizon([(i + 1) * 20000 for i in range(args.global_environments)],
                                  int(manifest["completed_planes"]), args.global_environments)
        for rank in range(topology):
            argv = profile_argv(original, topology=topology, rank=rank,
                                root=args.output_root, global_environments=args.global_environments)
            ranks.append({"topology": topology, "rank": rank, "gpu": rank,
                          "hard_wall_seconds": value["topology_gpu_hour_cap"] * 3600 / topology,
                          "argv": ["python", "-u", "codes/profile_mole_n81_efficiency.py", "rank",
                                   "--protocol", str(args.output_root / "PROFILE.json"),
                                   "--topology", str(topology), "--rank", str(rank)],
                          "trainer_argv": argv, "input_horizon": horizon,
                          "persistent_plus_single_probe_peak_bytes_estimate": (
                              local * value["memory_admission"]["state_reference_bytes"]
                              + value["memory_admission"]["replay_bytes"]
                              + value["memory_admission"]["reference_peak_bytes"]),
                          "estimate_excludes_unmeasured_large_learner_workspace": True})
    value["ranks"] = ranks
    write_new(args.output_root / "PROFILE.json", value)
    print(json.dumps({"status": "prepared_not_executed", "protocol": str(args.output_root / "PROFILE.json"),
                      "global_interactions": value["global_collected_interactions"], "boundary": BOUNDARY}))


def run_rank(args):
    protocol = json.loads(args.protocol.read_text())
    expected = validate_contract(protocol)
    if protocol.get("wrapper_sha256") != sha(__file__):
        raise ValueError("wrapper differs from the frozen profile identity")
    spec = next(row for row in protocol["ranks"] if row["topology"] == args.topology and row["rank"] == args.rank)
    argv = spec["trainer_argv"]
    output = Path(argv[argv.index("--output") + 1])
    if output.exists() or output.with_name("rank%d.profile.json" % args.rank).exists():
        raise FileExistsError("profile output must be new")
    sys.argv = argv[2:]
    import jax
    import jax.numpy as jnp
    import jax_mole_microbatch_distributed as trainer

    trainer_args = trainer.parse_args()
    if (trainer_args.action_parameterization != "per_turbine"
            or trainer_args.algorithm != "graph_sac_v2"
            or trainer_args.collection_steps != COLLECTION
            or trainer_args.reset_interactions != expected["reset_interactions"]):
        raise ValueError("fixed profile collector requires the frozen graph-SAC rate-action protocol")
    trainer.parse_args = lambda: trainer_args
    original_initialize = trainer.initialize_continuous_training
    signature = inspect.signature(original_initialize)
    initialized = 0
    actual_control_configuration = None

    def initialize(*positional, **keyword):
        nonlocal initialized, actual_control_configuration
        bound = signature.bind(*positional, **keyword)
        config = bound.arguments["config"]
        actual_control_configuration = {
            "max_yaw_angle_deg": config.max_yaw_angle_deg,
            "max_yaw_speed_deg_per_s": config.max_yaw_speed_deg_per_s,
            "control_dt_seconds": config.control_dt_seconds,
            "les_steps_per_interaction": config.les_steps_per_interaction,
        }
        if actual_control_configuration != {
                "max_yaw_angle_deg": MAX_YAW_DEG,
                "max_yaw_speed_deg_per_s": MAX_YAW_SPEED_DEG_PER_S,
                "control_dt_seconds": CONTROL_DT_SECONDS,
                "les_steps_per_interaction": expected["les_steps_per_control"]}:
            raise ValueError("physical yaw configuration differs from fixed-action decoder")
        bound.arguments["seed"] = rank_seed(SEED, args.rank, trainer_args.local_environments, initialized)
        result = original_initialize(*bound.args, **bound.kwargs)
        state = result[0]._replace(transitions=jnp.asarray(CLOCK_OFFSET, dtype=jnp.int32))
        initialized += 1
        return (state, *result[1:])

    def fixed_collector_action(actor, observations, key, action_bound, graph, frame_stack=1):
        del actor, key, action_bound, graph, frame_stack
        targets = jnp.where(jnp.arange(81) % 2 == 0, TARGET_YAW_DEG, -TARGET_YAW_DEG)
        action = fixed_yaw_rate_action(observations[..., -81:], targets, clip=jnp.clip)
        return action, jnp.zeros(observations.shape[:-1], dtype=observations.dtype)

    original_horizon = trainer.plane_horizon

    def horizon(**kwargs):
        if kwargs.get("resumed_global_interactions") is not None:
            raise ValueError("bounded computation profile cannot resume or train")
        ordinary = original_horizon(**kwargs)
        result = profile_horizon(kwargs["cursors"], kwargs["completed_planes"], protocol["global_environments"])
        if ordinary["global_interactions_end"] != result["global_interactions_end"]:
            raise ValueError("profile/trainer collection count mismatch")
        return result

    io_records = []
    original_save_full = trainer._save_full_checkpoint_atomic
    saved_states = []
    actor_io_records = []
    barrier_records = []
    host_conversion_records = []

    # Capture existing host conversions without additional full-state transfers.
    def timed_host_conversion(function, label):
        def convert(*positional, **keyword):
            before = time.perf_counter()
            converted = function(*positional, **keyword)
            host_conversion_records.append({"component": label,
                                            "seconds": time.perf_counter() - before})
            return converted
        return convert

    def save_full(path, payload):
        io_started = time.perf_counter()
        payload.update(purpose="computation_profile_full_state_only", computation_profile=expected,
                       wrapper_sha256=protocol["wrapper_sha256"])
        digest = original_save_full(path, payload)
        io_records.append({"path": str(path), "sha256": digest, "bytes": Path(path).stat().st_size,
                           "seconds": time.perf_counter() - io_started,
                           "finished_seconds": time.perf_counter() - started,
                           "timing_scope": "atomic serialization, fsync, replace and hash; host conversions recorded separately",
                           "global_interactions": payload["global_interactions"],
                           "process_id": payload["process_id"],
                           "num_processes": payload["num_processes"],
                           "completed_collector_blocks": payload["completed_collector_iterations_including_warmup"],
                           "history_blocks": len(payload["history"]),
                           "payload_keys": sorted(payload)})
        saved_states.append({"path": str(path), "plane_cursors": list(payload["plane_cursors"]),
                             "local_transition_clocks": [int(state.transitions) for state in payload["actor_states"]],
                             "local_les_steps": [int(state.farm_flow.step_index) for state in payload["actor_states"]],
                             "local_final_yaw_max_abs_deg": [float(trainer.np.max(trainer.np.abs(state.previous_yaw)))
                                                             for state in payload["actor_states"]],
                             "learner_updates": int(payload["learner"]["sac"].updates),
                             "learner_replay_size": int(payload["learner"]["replay"]["size"])})
        return digest

    original_save_actor = trainer.save_actor_checkpoint

    def save_actor(path, state, metadata, **kwargs):
        before = time.perf_counter()
        metadata = dict(metadata, purpose="computation_profile_actor_not_for_scientific_evaluation",
                        computation_profile=expected, wrapper_sha256=protocol["wrapper_sha256"])
        result = original_save_actor(path, state, metadata, **kwargs)
        actor_io_records.append({"path": str(path), "bytes": Path(path).stat().st_size,
                                 "sha256": sha(path), "seconds": time.perf_counter() - before,
                                 "global_interactions": metadata["actual_global_interactions"]})
        return result

    from jax.experimental import multihost_utils
    original_barrier = multihost_utils.sync_global_devices

    def barrier(name):
        before = time.perf_counter()
        result = original_barrier(name)
        barrier_records.append({"name": name, "started_seconds": before - started,
                                "seconds": time.perf_counter() - before, "completed": True})
        return result

    trainer.initialize_continuous_training = initialize
    trainer.graph_v2_actor_sample = fixed_collector_action
    trainer.plane_horizon = horizon
    trainer._save_full_checkpoint_atomic = save_full
    trainer.save_actor_checkpoint = save_actor
    trainer._tree_to_host = timed_host_conversion(trainer._tree_to_host, "environment_and_source_state_to_host")
    trainer._sharded_learner_to_checkpoint = timed_host_conversion(
        trainer._sharded_learner_to_checkpoint, "learner_and_replay_to_host")
    multihost_utils.sync_global_devices = barrier
    started = time.perf_counter()
    exit_code = trainer.main()
    result = json.loads(output.read_text())
    rows = result["history"]
    regular = [r for r in rows if not r["physical_reset_interactions_per_environment"]]
    reset = [r for r in rows if r["physical_reset_interactions_per_environment"]]
    actual = result["configuration"]["total_global_transitions_including_compile_warmup"]
    fixed_metrics, fixed_checks = fixed_action_evidence(rows)
    local = trainer_args.local_environments
    expected_final_cursors = spec["input_horizon"]["final_cursors_exclusive"][args.rank * local:(args.rank + 1) * local]
    checks = {
        "trainer_exited_successfully_after_complete_output": exit_code == 0 and result["status"] == "pass"
            and result["declared_budget_completed"] is True and result["recoverable_pause_reason"] is None,
        "trainer_internal_checks": all(result["checks"].values()),
        "exact_global_collection_work": actual == expected["global_collected_interactions"],
        "ten_warm_regular_blocks": len(regular) == 10,
        "one_real_150_control_reset": len(reset) == 1 and reset[0]["physical_reset_interactions_per_environment"] == 150,
        "initializers_match_frozen_local_E": initialized == trainer_args.local_environments,
        "real_sac_update_count": result["final_metrics"]["sac_updates"] == expected["global_learner_updates"],
        "real_actor_updates": any(r.get("sac_actor_gradient_norm", 0.) > 0. for r in rows),
        "common_global_sample_count": trainer_args.sample_size * args.topology == GLOBAL_SAMPLE,
        **fixed_checks,
        **checkpoint_completion_checks(result["checkpoint"], io_records, saved_states, barrier_records,
                                       actor_io_records, rank=args.rank, topology=args.topology,
                                       local_environments=local, global_interactions=expected["global_collected_interactions"],
                                       expected_final_cursors=expected_final_cursors),
    }
    detail = {
        "schema": SCHEMA, "status": "pass" if all(checks.values()) else "incomplete_or_failed",
        "checks": checks, "topology": args.topology, "rank": args.rank,
        "profile_protocol_sha256": sha(args.protocol), "trainer_output_sha256": sha(output),
        "configuration": expected, "actual_global_interactions": actual,
        "trainer_exit_code": exit_code, "trainer_status": result["status"],
        "actual_control_configuration": actual_control_configuration,
        "fixed_action_physical_evidence_by_warm_block": fixed_metrics,
        "cold_and_total_timing": result["timing"], "wrapper_total_seconds": time.perf_counter() - started,
        "normal_block_seconds": [r["wallclock_seconds"] for r in regular],
        "reset_blocks": reset,
        "normal_collector_seconds": [r["collector_seconds"] for r in regular],
        "normal_learner_and_sync_seconds": [r["learner_seconds"] for r in regular],
        "complete_checkpoint_io": io_records,
        "actor_checkpoint_io": actor_io_records,
        "checkpoint_host_conversion_timing": host_conversion_records,
        "global_barriers": barrier_records,
        "saved_physical_state_accounting": saved_states,
        "memory_stats": jax.local_devices()[0].memory_stats(),
        "actual_final_cursors": result["configuration"]["plane_cursors"],
        "evidence_boundary": BOUNDARY,
        "synchronization_scope": "real all-gather/pmean in learner, block_until_ready collector and learner, and global barrier after full checkpoint; communication is included, not independently isolated",
    }
    write_new(output.with_name("rank%d.profile.json" % args.rank), detail)
    print(json.dumps({"profile_status": detail["status"], "checks": checks,
                      "topology": args.topology, "rank": args.rank, "memory_stats": detail["memory_stats"]}), flush=True)
    return 0 if detail["status"] == "pass" and exit_code == 0 else 1


def summarize(args):
    protocol = json.loads(args.protocol.read_text())
    expected = validate_contract(protocol)
    root = args.protocol.parent
    topology_results = {}
    for topology in (1, 2, 4):
        reports = [json.loads((root / ("p%d" % topology) / ("rank%d.profile.json" % rank)).read_text())
                   for rank in range(topology)]
        if any(r.get("schema") != SCHEMA or r["status"] != "pass"
               or not r.get("checks") or not all(r["checks"].values())
               or r["profile_protocol_sha256"] != sha(args.protocol) for r in reports):
            raise ValueError("all exact frozen topology reports must pass before scaling summaries")
        if any(r["configuration"] != expected for r in reports):
            raise ValueError("topologies performed different computation contracts")
        trainer_results = [json.loads((root / ("p%d" % topology) / ("rank%d.trainer.json" % rank)).read_text())
                           for rank in range(topology)]
        if any(not all(fixed_action_evidence(result["history"])[1].values()) for result in trainer_results):
            raise ValueError("actual executed yaw/rate differs from the corrected fixed-action contract")
        regular = [max(r["normal_block_seconds"][i] for r in reports) for i in range(10)]
        collect = [max(r["normal_collector_seconds"][i] for r in reports) for i in range(10)]
        learner = [max(r["normal_learner_and_sync_seconds"][i] for r in reports) for i in range(10)]
        total = max(r["wrapper_total_seconds"] for r in reports)
        allocated = sum(r["wrapper_total_seconds"] for r in reports) / 3600
        topology_results[str(topology)] = {
            "normal_block_critical_rank_seconds": regular,
            "normal_block_seconds_median": statistics.median(regular),
            "normal_block_seconds_min": min(regular), "normal_block_seconds_max": max(regular),
            "normal_collector_seconds_median": statistics.median(collect),
            "normal_learner_including_communication_seconds_median": statistics.median(learner),
            "reset_block_seconds": max(r["reset_blocks"][0]["wallclock_seconds"] for r in reports),
            "checkpoint_io_seconds_critical_rank": max(sum(x["seconds"] for x in r["complete_checkpoint_io"]) for r in reports),
            "checkpoint_host_conversion_seconds_critical_rank": max(
                sum(x["seconds"] for x in r["checkpoint_host_conversion_timing"]) for r in reports),
            "actor_checkpoint_io_seconds_critical_rank": max(
                sum(x["seconds"] for x in r["actor_checkpoint_io"]) for r in reports),
            "final_checkpoint_barrier_seconds_critical_rank": max(
                sum(x["seconds"] for x in r["global_barriers"] if x["name"] == "microbatch-checkpoint-written")
                for r in reports),
            "total_including_cold_reset_learner_sync_and_checkpoint_seconds": total,
            "rank_accounted_gpu_hours": allocated,
            "peak_allocator_bytes": max(r["memory_stats"]["peak_bytes_in_use"] for r in reports),
            "actual_global_interactions": protocol["global_collected_interactions"],
            "bounded_profile_interactions_per_second": protocol["global_collected_interactions"] / total,
            "mean_farm_power_mw_by_warm_block": [sum(result["history"][i]["mean_farm_power_mw"]
                for result in trainer_results) / topology for i in range(WARM_BLOCKS)],
            "fixed_action_physical_evidence_by_rank": [r["fixed_action_physical_evidence_by_warm_block"] for r in reports],
        }
    one = topology_results["1"]
    for label, value in topology_results.items():
        p = int(label)
        value["regular_block_speedup"] = one["normal_block_seconds_median"] / value["normal_block_seconds_median"]
        value["regular_block_parallel_efficiency"] = value["regular_block_speedup"] / p
        value["bounded_profile_total_speedup"] = one["total_including_cold_reset_learner_sync_and_checkpoint_seconds"] / value["total_including_cold_reset_learner_sync_and_checkpoint_seconds"]
        value["bounded_profile_total_parallel_efficiency"] = value["bounded_profile_total_speedup"] / p
    value = {"schema": SCHEMA + "_summary", "status": "matched_profiles_complete",
             "protocol_sha256": sha(args.protocol), "topologies": topology_results,
             "rank_accounted_gpu_hours": sum(v["rank_accounted_gpu_hours"] for v in topology_results.values()),
             "evidence_boundary": BOUNDARY,
             "budget_note": "Prospective v2 budget is 5 GPUh per topology and 15 GPUh total; v1 costs/results remain separate. Rank timers exclude container start/import/prepare overhead and failed attempts; supervisor accounting remains authoritative. No automatic retry.",
             "uncertainty": "Ten serial blocks per topology are correlated timing samples, not independent full-training replicates; report range rather than an iid confidence interval."}
    write_new(args.output, value)
    print(json.dumps({"status": value["status"], "output": str(args.output), "evidence_boundary": BOUNDARY}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-archive", type=Path, required=True)
    prep.add_argument("--recipes", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--global-environments", type=int, default=32)
    rank = sub.add_parser("rank")
    rank.add_argument("--protocol", type=Path, required=True)
    rank.add_argument("--topology", type=int, choices=(1, 2, 4), required=True)
    rank.add_argument("--rank", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--protocol", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "summarize":
        return summarize(args)
    return run_rank(args)


if __name__ == "__main__":
    raise SystemExit(main())
