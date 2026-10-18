#!/usr/bin/env python3
"""Paired N81 collector component diagnostic, without changing trainer cadence.

Two independent processes resume the same audited single-GPU profile state.
Only the dispatch of eight control interactions differs: one eight-step scan,
or eight calls to the same one-step JIT collector. Both retain 32 environments,
256 transitions per block, and the original Graph-SAC learner. This is a
component timing experiment, not a scientific policy or convergence run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

SCHEMA = "paper1_n81_collection_dispatch_ablation_20261013_v1"
MODES = ("scan8", "single_jit8")
GLOBAL_ENVS = 32
COLLECTION = 8
WARM_BLOCKS = 2
PARENT_INTERACTIONS = 3072
PARENT_SHA256 = "03b151473db2462fe9f2d8d057d06b1ddae27ccccadc53ec1ee96a8530aad2d0"
PARENT_BYTES = 13802602842
PAIR_GPU_HOUR_CAP = 4.0
BRANCH_GPU_HOUR_CAP = 2.0
RESERVE_SECONDS = 600
FLOAT_RTOL = 1.0e-4
FLOAT_ATOL = 1.0e-5
BOUNDARY = (
    "Same N81 LES equations, grid, tensor SGS, inlet samples, restored Actor, "
    "environment and learner RNG, Replay, 32 environments, 8 controls per "
    "environment and 256 Graph-SAC updates per block. Only scan8 versus eight "
    "single-step JIT dispatches differs. Parent is the completed computation "
    "profile (3072 Replay entries, synthetic episode clock 500); not mature "
    "scientific training, not time-to-gain, not an end-to-end Mole comparison. "
    "Cold compilation and checkpoint costs are reported separately. Two "
    "serial blocks are timing observations, not independent training seeds."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replace_argument(argv, name, value):
    result = list(argv)
    if name in result:
        result[result.index(name) + 1] = str(value)
    else:
        result.extend((name, str(value)))
    return result


def contract():
    return {
        "schema": SCHEMA, "purpose": "component_computation_diagnostic_only",
        "modes": list(MODES), "target_turbines": 81, "global_environments": GLOBAL_ENVS,
        "collection_steps": COLLECTION, "warm_blocks_per_branch": WARM_BLOCKS,
        "new_interactions_per_branch": GLOBAL_ENVS * COLLECTION * WARM_BLOCKS,
        "new_learner_updates_per_branch": GLOBAL_ENVS * COLLECTION * WARM_BLOCKS,
        "global_sample_size": 256, "microbatch": 1,
        "parent_global_interactions": PARENT_INTERACTIONS,
        "parent_sha256": PARENT_SHA256, "parent_bytes": PARENT_BYTES,
        "final_global_interactions": PARENT_INTERACTIONS + GLOBAL_ENVS * COLLECTION * WARM_BLOCKS,
        "parent_local_transition_clocks": [500] * GLOBAL_ENVS,
        "parent_local_les_steps": [12300] * GLOBAL_ENVS,
        "final_local_transition_clocks": [516] * GLOBAL_ENVS,
        "final_local_les_steps": [13100] * GLOBAL_ENVS,
        "physical_reset_in_retained_blocks": False,
        "restored_cold_compile_block_is_discarded": True,
        "pair_gpu_hour_cap": PAIR_GPU_HOUR_CAP,
        "branch_gpu_hour_cap": BRANCH_GPU_HOUR_CAP,
        "checkpoint_save_reserve_seconds": RESERVE_SECONDS,
        "automatic_retry": False, "float_state_rtol": FLOAT_RTOL,
        "float_state_atol": FLOAT_ATOL, "integer_and_rng_comparison": "exact",
        "evidence_boundary": BOUNDARY,
    }


def validate_protocol(value):
    for key, expected in contract().items():
        if value.get(key) != expected:
            raise ValueError("frozen collector contract differs: " + key)
    if value.get("wrapper_sha256") != sha256(__file__):
        raise ValueError("wrapper differs from its frozen identity")
    return value


def branch_argv(original, root, parent, mode, *, archive=None, precursor=None):
    if mode not in MODES:
        raise ValueError("invalid collection mode")
    result = list(original)
    if "--save-initial-checkpoint" in result:
        result.remove("--save-initial-checkpoint")
    directory = Path(root) / mode
    overrides = {
        "--coordinator": "127.0.0.1:17631", "--process-id": 0, "--num-processes": 1,
        "--local-environments": GLOBAL_ENVS, "--microbatch": 1,
        "--collection-steps": COLLECTION, "--sample-size": 256,
        "--iterations": WARM_BLOCKS, "--burnin-interactions": 0,
        "--planned-global-interactions": contract()["final_global_interactions"],
        "--resume-checkpoint": parent,
        "--checkpoint": directory / "component_actor.pkl",
        "--full-checkpoint": directory / "component_full_state.pkl",
        "--checkpoint-milestones": "", "--full-checkpoint-interval": 0,
        "--stop-request": directory / "STOP",
        "--stop-after-seconds": BRANCH_GPU_HOUR_CAP * 3600 - RESERVE_SECONDS,
        "--output": directory / "trainer.json",
    }
    if archive is not None:
        overrides["--plane-archive"] = archive
    if precursor is not None:
        overrides["--precursor"] = precursor
    for name, value in overrides.items():
        result = replace_argument(result, name, value)
    return result


def prepare(args):
    profile = json.loads(args.profile_protocol.read_text())
    if profile.get("schema") != "paper1_n81_bounded_computation_profile_v2":
        raise ValueError("completed corrected v2 profile protocol required")
    original = next(row["trainer_argv"] for row in profile["ranks"]
                    if row["topology"] == 1 and row["rank"] == 0)
    profile_result = json.loads(args.profile_rank_result.read_text())
    if (profile_result.get("status") != "pass" or profile_result.get("topology") != 1
            or not profile_result.get("checks") or not all(profile_result["checks"].values())):
        raise ValueError("complete corrected single-GPU profile receipt required")
    if profile_result.get("profile_protocol_sha256") != sha256(args.profile_protocol):
        raise ValueError("profile receipt does not bind the supplied profile protocol")
    io = profile_result["complete_checkpoint_io"][-1]
    if (io["sha256"] != PARENT_SHA256 or io["bytes"] != PARENT_BYTES
            or io["global_interactions"] != PARENT_INTERACTIONS):
        raise ValueError("single-GPU parent does not match frozen receipt")
    value = contract()
    parent_integrity = None
    if not args.dry_run:
        stat = args.parent.stat()
        if stat.st_size != PARENT_BYTES or sha256(args.parent) != PARENT_SHA256:
            raise ValueError("parent full-state integrity differs from audited receipt")
        after = args.parent.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("parent changed during CPU integrity check")
        parent_integrity = {"sha256": PARENT_SHA256, "bytes": after.st_size,
                            "mtime_ns": after.st_mtime_ns, "checked_unix": time.time(),
                            "scope": "CPU preparation hashes once; runtime requires read-only input mount and unchanged stat"}
    value.update(wrapper_sha256=sha256(__file__), profile_protocol_sha256=sha256(args.profile_protocol),
                 profile_rank_result_sha256=sha256(args.profile_rank_result), parent_path=str(args.parent),
                 parent_integrity=parent_integrity,
                 branches=[{"mode": mode, "gpus": 1,
                            "hard_wall_seconds": BRANCH_GPU_HOUR_CAP * 3600,
                            "trainer_argv": branch_argv(original, args.output_root, args.parent, mode,
                                                        archive=args.archive, precursor=args.precursor)}
                           for mode in MODES])
    if not args.dry_run:
        write_new(args.output_root / "PROTOCOL.json", value)
    print(json.dumps({"status": "dry_run" if args.dry_run else "prepared_not_executed",
                      "protocol": value}, indent=2, sort_keys=True))


class SingleJitEight:
    """Host dispatch loop; each call executes the same JIT one-step factory."""
    def __init__(self, single, stack_mean, concatenate):
        self.single = single
        self.stack_mean = stack_mean
        self.concatenate = concatenate

    def __call__(self, state, *args, **kwargs):
        records = []
        for _ in range(COLLECTION):
            state, metrics = self.single(state, *args, **kwargs)
            records.append(metrics)
        result = dict(records[-1])
        for name in ("mean_farm_power_mw", "mean_reward", "mean_angle_penalty",
                     "mean_abs_yaw_deg", "mean_abs_yaw_speed_deg_per_s",
                     "inlet_interval_delta_rms", "episode_reset"):
            result[name] = self.stack_mean([row[name] for row in records])
        if "raw_transitions" in result:
            result["raw_transitions"] = {
                key: self.concatenate([row["raw_transitions"][key] for row in records])
                for key in result["raw_transitions"]}
        return state, result


def collector_factory(original, original_jit, mode, stack_mean, concatenate):
    signature = inspect.signature(original)

    def factory(*positional, **keyword):
        bound = signature.bind(*positional, **keyword)
        bound.apply_defaults()
        if bound.arguments["updates_per_iteration"] != 0:
            raise ValueError("component collector cannot contain learner updates")
        if bound.arguments["collection_steps_per_iteration"] != COLLECTION:
            raise ValueError("the trainer collection cadence must remain eight")
        if mode == "scan8":
            return original(*bound.args, **bound.kwargs)
        bound.arguments["collection_steps_per_iteration"] = 1
        single = original_jit(original(*bound.args, **bound.kwargs), donate_argnums=(0,))
        return SingleJitEight(single, stack_mean, concatenate)
    return factory


def selective_jit(original_jit):
    def jit(function, *args, **kwargs):
        # Jitting this host loop would fuse the eight calls and destroy the ablation.
        if isinstance(function, SingleJitEight):
            if args or kwargs != {"donate_argnums": (0,)}:
                raise ValueError("unexpected outer collector JIT configuration")
            return function
        return original_jit(function, *args, **kwargs)
    return jit


def run(args):
    started = time.perf_counter()
    value = validate_protocol(json.loads(args.protocol.read_text()))
    spec = next(row for row in value["branches"] if row["mode"] == args.collection_mode)
    parent = Path(value["parent_path"])
    stat = parent.stat()
    receipt = value.get("parent_integrity") or {}
    if (receipt.get("sha256") != PARENT_SHA256 or stat.st_size != receipt.get("bytes")
            or stat.st_mtime_ns != receipt.get("mtime_ns")):
        raise ValueError("parent CPU hash receipt or unchanged-stat binding failed")
    if args.dry_run:
        print(json.dumps({"status": "validated_not_executed", "mode": args.collection_mode,
                          "trainer_argv": spec["trainer_argv"], "evidence_boundary": BOUNDARY}))
        return 0
    argv = spec["trainer_argv"]
    output = Path(argv[argv.index("--output") + 1])
    report_path = output.with_name("component.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("component outputs must be new")
    import jax
    import jax.numpy as jnp
    import jax_mole_microbatch_distributed as trainer
    sys.argv = argv[2:]
    parsed = trainer.parse_args()
    if any((parsed.algorithm != "graph_sac_v2", parsed.target_turbines != 81,
            parsed.num_processes != 1, parsed.local_environments != GLOBAL_ENVS,
            parsed.collection_steps != COLLECTION, parsed.sample_size != 256,
            parsed.microbatch != 1, parsed.rhs_assembly != "tensor",
            parsed.compact_solver != "dense_inverse", parsed.resume_learner_only,
            parsed.diagnostic_updates_per_iteration != -1,
            parsed.action_parameterization != "per_turbine")):
        raise ValueError("same-physics/cadence component contract violated")
    trainer.parse_args = lambda: parsed
    original_factory = trainer.make_continuous_training_iteration
    original_jit = jax.jit
    trainer.make_continuous_training_iteration = collector_factory(
        original_factory, original_jit, args.collection_mode,
        lambda values: jnp.mean(jnp.stack(values), axis=0),
        lambda values: jnp.concatenate(values, axis=0))
    jax.jit = selective_jit(original_jit)
    original_load = trainer._load_full_checkpoint
    parent_record = {}

    def audited_load(path):
        payload = original_load(path)
        if (Path(path) != parent or payload.get("purpose") != "computation_profile_full_state_only"
                or payload["global_interactions"] != PARENT_INTERACTIONS
                or payload["process_id"] != 0 or payload["num_processes"] != 1):
            raise ValueError("restored parent lineage/topology differs")
        clocks = [int(state.transitions) for state in payload["actor_states"]]
        steps = [int(state.farm_flow.step_index) for state in payload["actor_states"]]
        if clocks != value["parent_local_transition_clocks"] or steps != value["parent_local_les_steps"]:
            raise ValueError("parent LES state or episode clock differs")
        parent_record.update(local_clocks=clocks, local_les_steps=steps,
                             plane_cursors=list(payload["plane_cursors"]),
                             configuration=payload["configuration"])
        return payload
    trainer._load_full_checkpoint = audited_load
    save_records = []
    original_save = trainer._save_full_checkpoint_atomic

    def save(path, payload):
        payload.update(purpose="collector_component_diagnostic_full_state_only",
                       collection_component_mode=args.collection_mode,
                       component_protocol_sha256=sha256(args.protocol),
                       component_parent_sha256=PARENT_SHA256)
        before = time.perf_counter()
        digest = original_save(path, payload)
        save_records.append({"path": str(path), "sha256": digest, "bytes": Path(path).stat().st_size,
                             "save_and_hash_seconds": time.perf_counter() - before,
                             "global_interactions": payload["global_interactions"],
                             "local_clocks": [int(state.transitions) for state in payload["actor_states"]],
                             "local_les_steps": [int(state.farm_flow.step_index) for state in payload["actor_states"]],
                             "plane_cursors": list(payload["plane_cursors"]),
                             "learner_updates": int(payload["learner"]["sac"].updates),
                             "learner_replay_size": int(payload["learner"]["replay"]["size"])})
        return digest
    trainer._save_full_checkpoint_atomic = save
    try:
        exit_code = trainer.main()
    finally:
        jax.jit = original_jit
    result = json.loads(output.read_text())
    rows = result["history"]
    final = save_records[-1] if save_records else {}
    checks = {
        "trainer_success": exit_code == 0 and result["status"] == "pass",
        "trainer_internal_checks": bool(result["checks"]) and all(result["checks"].values()),
        "exact_two_warm_blocks": len(rows) == WARM_BLOCKS,
        "exact_global_interactions": result["configuration"].get("total_global_transitions_including_compile_warmup") == value["final_global_interactions"],
        "no_reset_in_retained_blocks": all(row["physical_reset_interactions_per_environment"] == 0 for row in rows),
        "full_state_saved_and_hashed": bool(final) and len(final.get("sha256", "")) == 64,
        "complete_environment_roster": final.get("local_clocks") == value["final_local_transition_clocks"],
        "exact_LES_steps": final.get("local_les_steps") == value["final_local_les_steps"],
        "real_SAC_updates": final.get("learner_updates") == value["final_global_interactions"],
        "real_Replay_rows": final.get("learner_replay_size") == value["final_global_interactions"],
    }
    report = {"schema": SCHEMA, "mode": args.collection_mode,
              "status": "pass" if all(checks.values()) else "failed_or_incomplete", "checks": checks,
              "protocol_sha256": sha256(args.protocol), "parent_sha256": PARENT_SHA256,
              "parent_record": parent_record, "wrapper_seconds": time.perf_counter() - started,
              "cold_and_total_timing": result["timing"], "checkpoint_io": save_records,
              "normal_collector_seconds": [row["collector_seconds"] for row in rows],
              "normal_learner_seconds": [row["learner_seconds"] for row in rows],
              "normal_block_seconds": [row["wallclock_seconds"] for row in rows],
              "actual_cadence": {"global_environments": GLOBAL_ENVS, "collection_steps": COLLECTION,
                                 "samples_per_block": GLOBAL_ENVS * COLLECTION,
                                 "learner_updates_per_block": GLOBAL_ENVS * COLLECTION},
              "evidence_boundary": BOUNDARY}
    write_new(report_path, report)
    print(json.dumps({"status": report["status"], "mode": args.collection_mode,
                      "checks": checks, "output": str(report_path)}), flush=True)
    return 0 if report["status"] == "pass" else 1


def compare_numeric_trees(left, right):
    """CPU final-state check, including all physical, RNG, Replay and learner leaves."""
    import jax
    import numpy as np
    left_paths, left_tree = jax.tree.flatten_with_path(left)
    right_paths, right_tree = jax.tree.flatten_with_path(right)
    if left_tree != right_tree:
        return {"pass": False, "reason": "different state trees"}
    records = []
    for (path, a), (_, b) in zip(left_paths, right_paths):
        a, b = np.asarray(a), np.asarray(b)
        label = jax.tree_util.keystr(path)
        if a.shape != b.shape or a.dtype != b.dtype:
            records.append({"path": label, "pass": False, "reason": "shape/dtype differs"})
            continue
        exact = True
        close = True
        finite = True
        maximum = 0.0
        # Bound temporary arrays even for multi-GiB flow/Replay leaves.
        x, y = a.reshape(-1), b.reshape(-1)
        floating = np.issubdtype(a.dtype, np.inexact)
        for start in range(0, x.size, 1024 * 1024):
            ax, bx = x[start:start + 1024 * 1024], y[start:start + 1024 * 1024]
            exact = exact and bool(np.array_equal(ax, bx))
            if floating:
                finite = finite and bool(np.isfinite(ax).all() and np.isfinite(bx).all())
                close = close and bool(np.allclose(ax, bx, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=False))
                if ax.size:
                    maximum = max(maximum, float(np.max(np.abs(ax - bx))))
            else:
                close = close and bool(np.array_equal(ax, bx))
        records.append({"path": label, "shape": list(a.shape), "dtype": str(a.dtype),
                        "exact": exact, "finite": finite, "max_abs_difference": maximum,
                        "pass": finite and close})
    return {"pass": all(row["pass"] for row in records), "leaves": records,
            "float_rtol": FLOAT_RTOL, "float_atol": FLOAT_ATOL}


def summarize(args):
    value = validate_protocol(json.loads(args.protocol.read_text()))
    root = args.protocol.parent
    reports = {mode: json.loads((root / mode / "component.json").read_text()) for mode in MODES}
    if any(report["status"] != "pass" or not all(report["checks"].values())
           or report["protocol_sha256"] != sha256(args.protocol) for report in reports.values()):
        raise ValueError("both exact component receipts must pass")
    import jax_mole_microbatch_distributed as trainer
    states = {}
    for mode in MODES:
        receipt = reports[mode]["checkpoint_io"][-1]
        path = Path(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise ValueError("final full-state hash differs")
        payload = trainer._load_full_checkpoint(path)
        if payload.get("component_parent_sha256") != PARENT_SHA256:
            raise ValueError("final state lost parent binding")
        states[mode] = payload
    same_config = states[MODES[0]]["configuration"] == states[MODES[1]]["configuration"]
    same_cursors = states[MODES[0]]["plane_cursors"] == states[MODES[1]]["plane_cursors"]
    trees = {mode: {key: states[mode][key] for key in
                   ("actor_states", "plane_source_states", "learner")} for mode in MODES}
    equivalence = compare_numeric_trees(trees[MODES[0]], trees[MODES[1]])
    qualified = same_config and same_cursors and equivalence["pass"]
    medians = {mode: statistics.median(reports[mode]["normal_collector_seconds"]) for mode in MODES}
    summary = {"schema": SCHEMA + "_summary", "status": "pass" if qualified else "state_equivalence_failed",
               "same_configuration": same_config, "same_inlet_cursors": same_cursors,
               "state_equivalence": equivalence, "collector_median_seconds": medians,
               "collector_speedup_single_over_scan": medians["single_jit8"] / medians["scan8"] if qualified else None,
               "normal_block_median_seconds": {mode: statistics.median(reports[mode]["normal_block_seconds"]) for mode in MODES},
               "normal_learner_median_seconds": {mode: statistics.median(reports[mode]["normal_learner_seconds"]) for mode in MODES},
               "rank_timer_gpu_hours": sum(report["wrapper_seconds"] for report in reports.values()) / 3600,
               "budget_note": "Supervisor accounting includes container allocation, failed attempts and startup; it is authoritative.",
               "evidence_boundary": BOUNDARY}
    write_new(args.output, summary)
    print(json.dumps({"status": summary["status"], "speedup": summary["collector_speedup_single_over_scan"],
                      "output": str(args.output)}))
    return 0 if qualified else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--profile-protocol", type=Path, required=True)
    prep.add_argument("--profile-rank-result", type=Path, required=True)
    prep.add_argument("--parent", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--archive", type=Path)
    prep.add_argument("--precursor", type=Path)
    prep.add_argument("--dry-run", action="store_true")
    branch = sub.add_parser("run")
    branch.add_argument("--protocol", type=Path, required=True)
    branch.add_argument("--collection-mode", choices=MODES, required=True)
    branch.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--protocol", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        return prepare(args)
    if args.operation == "summarize":
        return summarize(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
