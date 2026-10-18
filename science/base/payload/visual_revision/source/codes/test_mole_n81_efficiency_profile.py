"""Stdlib contracts and closed-loop regression; optional NumPy array check."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from profile_mole_n81_efficiency import (
    CHECKPOINT_SAVE_RESERVE_SECONDS, CLOCK_OFFSET, CONTROLS, MAX_YAW_DEG,
    SCHEMA, TARGET_YAW_DEG, YAW_OBSERVATION_SCALE, checkpoint_completion_checks,
    contract, fixed_action_evidence, fixed_yaw_rate_action, prepare, profile_argv,
    profile_horizon, rank_seed, sha, summarize, validate_contract,
)

try:
    import numpy as np
except ImportError:
    np = None


def flag(argv, name):
    return argv[argv.index(name) + 1]


class MatchedEfficiencyProfileTest(unittest.TestCase):
    def test_fixed_action_recovers_degrees_and_converges_without_40_degree_saturation(self):
        clip = lambda value, low, high: min(high, max(low, value))
        for target in (TARGET_YAW_DEG, -TARGET_YAW_DEG):
            for initial in (0.0, -40.0, 40.0, -25.0, 25.0, -17.5, 17.5, target):
                with self.subTest(target=target, initial=initial):
                    yaw = initial
                    trajectory = []
                    for _ in range(8):
                        normalized = yaw * YAW_OBSERVATION_SCALE / MAX_YAW_DEG
                        action = fixed_yaw_rate_action(normalized, target, clip=clip)
                        next_yaw = clip(yaw + 10.0 * action, -MAX_YAW_DEG, MAX_YAW_DEG)
                        self.assertLessEqual(abs(next_yaw - yaw), 10.0)
                        self.assertLessEqual(abs(next_yaw - target), abs(yaw - target))
                        yaw = next_yaw
                        trajectory.append(yaw)
                    self.assertAlmostEqual(yaw, target)
                    self.assertEqual(trajectory[-2:], [target, target])
                    self.assertEqual(fixed_yaw_rate_action(
                        yaw * YAW_OBSERVATION_SCALE / MAX_YAW_DEG, target, clip=clip), 0.0)
                    if initial == 0.0:
                        self.assertEqual(trajectory[:3], [target / 2, target, target])
                        self.assertTrue(all(abs(value) <= TARGET_YAW_DEG for value in trajectory))

    @unittest.skipIf(np is None, "NumPy unavailable locally; scalar shared-function regression remains mandatory")
    def test_fixed_action_numpy_array_matches_81_turbine_closed_loop(self):
        targets = np.where(np.arange(81) % 2 == 0, TARGET_YAW_DEG, -TARGET_YAW_DEG)
        yaw = np.resize(np.array([0., -40., 40., 17.5, -25.], dtype=np.float32), (2, 81))
        for _ in range(8):
            action = fixed_yaw_rate_action(yaw * YAW_OBSERVATION_SCALE / MAX_YAW_DEG,
                                           targets, clip=np.clip)
            self.assertEqual(action.shape, yaw.shape)
            yaw = np.clip(yaw + 10. * action, -MAX_YAW_DEG, MAX_YAW_DEG)
        np.testing.assert_array_equal(yaw, np.broadcast_to(targets, yaw.shape))
        np.testing.assert_array_equal(fixed_yaw_rate_action(yaw * .1, targets, clip=np.clip),
                                      np.zeros_like(yaw))

    def test_actual_yaw_metrics_reject_old_saturation_and_nonzero_executed_rate(self):
        rows = [{"iteration": i + 1, "mean_abs_yaw_deg": 20.,
                 "mean_abs_yaw_speed_deg_per_s": 0.} for i in range(11)]
        evidence, checks = fixed_action_evidence(rows)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(evidence), 11)
        for key, bad_value in (("mean_abs_yaw_deg", 40.),
                               ("mean_abs_yaw_speed_deg_per_s", .1),
                               ("mean_abs_yaw_deg", float("nan"))):
            bad = copy.deepcopy(rows)
            bad[-1][key] = bad_value
            self.assertFalse(all(fixed_action_evidence(bad)[1].values()))
        self.assertFalse(all(fixed_action_evidence(rows[:-1])[1].values()))
        self.assertFalse(all(fixed_action_evidence([{} for _ in rows])[1].values()))

    def test_final_checkpoint_requires_every_environment_exact_work_and_completed_barrier(self):
        path = "/batch/p4/profile_full_state.process0.pkl"
        checkpoint = {"full_state_process_local": path, "full_state_sha256": "a" * 64,
                      "path_process0_only": "/batch/p4/profile_actor.pkl"}
        io = [{"path": path, "sha256": "a" * 64, "bytes": 123, "finished_seconds": 3500.,
               "global_interactions": 3072, "completed_collector_blocks": 12,
               "history_blocks": 11, "process_id": 0, "num_processes": 4,
               "payload_keys": ["actor_states", "plane_source_states", "plane_cursors", "learner",
                                "configuration", "history", "process_id", "num_processes",
                                "global_interactions", "completed_collector_iterations_including_warmup"]}]
        cursors = [i * 20000 + 12300 for i in range(1, 9)]
        states = [{"path": path, "plane_cursors": cursors,
                   "local_les_steps": [12300] * 8, "local_transition_clocks": [500] * 8,
                   "local_final_yaw_max_abs_deg": [0.] * 8,
                   "learner_updates": 3072, "learner_replay_size": 3072}]
        barriers = [{"name": "microbatch-checkpoint-written", "completed": True,
                     "started_seconds": 3501.}]
        actors = [{"path": checkpoint["path_process0_only"], "global_interactions": 3072,
                   "sha256": "b" * 64, "bytes": 12}]
        kwargs = dict(rank=0, topology=4, local_environments=8, global_interactions=3072,
                      expected_final_cursors=cursors)
        self.assertTrue(all(checkpoint_completion_checks(checkpoint, io, states, barriers, actors, **kwargs).values()))
        for key, value in (("local_les_steps", [12300] * 7),
                           ("local_transition_clocks", [500] * 7 + [492]),
                           ("local_final_yaw_max_abs_deg", [0.] * 7 + [20.]),
                           ("learner_updates", 2816), ("plane_cursors", cursors[:-1])):
            bad_states = copy.deepcopy(states)
            bad_states[0][key] = value
            self.assertFalse(all(checkpoint_completion_checks(checkpoint, io, bad_states, barriers, actors, **kwargs).values()))
        self.assertFalse(all(checkpoint_completion_checks(checkpoint, io, states, [], actors, **kwargs).values()))
        self.assertFalse(all(checkpoint_completion_checks(checkpoint, io, states,
            [dict(barriers[0], started_seconds=3499.)], actors, **kwargs).values()))
        bad_io = [dict(io[0], path=path.replace(".process", "_rolling0.process"))]
        self.assertFalse(all(checkpoint_completion_checks(checkpoint, bad_io, states, barriers, actors, **kwargs).values()))

    def test_work_includes_real_reset_and_not_fabricated_earlier_controls(self):
        work = contract()
        self.assertEqual(CLOCK_OFFSET, 404)
        self.assertEqual(CONTROLS, 96)
        self.assertEqual(work["global_collected_interactions"], 3072)
        self.assertEqual(work["global_learner_updates"], 3072)
        self.assertEqual(work["les_steps_over_all_environments"], 393600)
        self.assertEqual(work["initial_burnin_interactions"], 0)
        boundaries = [(CLOCK_OFFSET + (i + 1) * 8) // 500
                      - (CLOCK_OFFSET + i * 8) // 500 for i in range(12)]
        self.assertEqual(boundaries, [0] * 11 + [1])

    def test_all_global_inputs_and_seeds_match_across_topologies(self):
        for topology in (1, 2, 4):
            local = 32 // topology
            seeds, cursors = [], []
            for rank in range(topology):
                seeds.extend(rank_seed(11, rank, local, i) for i in range(local))
                cursors.extend((i + 1) * 20000 + rank * local * 20000 for i in range(local))
            self.assertEqual(seeds, list(range(11, 43)))
            self.assertEqual(cursors, list(range(20000, 640001, 20000)))

    def test_argv_matches_sample_count_updates_physics_and_cap(self):
        original = ["python", "-u", "codes/jax_mole_microbatch_distributed.py",
                    "--sample-size", "64", "--local-environments", "8",
                    "--replay-capacity", "262144", "--reset-interactions", "150",
                    "--collection-steps", "8", "--episode-interactions", "500",
                    "--save-initial-checkpoint"]
        for topology in (1, 2, 4):
            for rank in range(topology):
                argv = profile_argv(original, topology=topology, rank=rank, root=Path("/batch/profile"))
                self.assertEqual(int(flag(argv, "--sample-size")) * topology, 256)
                self.assertEqual(int(flag(argv, "--local-environments")) * topology, 32)
                self.assertEqual(flag(argv, "--replay-capacity"), "262144")
                self.assertEqual(flag(argv, "--reset-interactions"), "150")
                self.assertEqual(flag(argv, "--episode-interactions"), "500")
                self.assertEqual(flag(argv, "--burnin-interactions"), "0")
                self.assertEqual(int(flag(argv, "--iterations")), 11)
                self.assertNotIn("--save-initial-checkpoint", argv)
                self.assertEqual(float(flag(argv, "--stop-after-seconds")), 18000 / topology - 600)
        self.assertEqual(contract()["topology_gpu_hour_cap"], 5.)
        self.assertEqual(contract()["total_gpu_hour_cap"], 15.)
        self.assertEqual(CHECKPOINT_SAVE_RESERVE_SECONDS, 600.)
        self.assertEqual(contract()["checkpoint_save_reserve_seconds"], 600.)
        self.assertFalse(contract()["automatic_retry"])

    def test_legacy_profile_contract_is_rejected_before_reuse(self):
        validate_contract(contract())
        legacy = dict(contract(), schema="paper1_n81_bounded_computation_profile_v1")
        with self.assertRaises(ValueError):
            validate_contract(legacy)

    def test_input_horizon_includes_reset_and_rejects_short_archive(self):
        cursors = list(range(20000, 640001, 20000))
        result = profile_horizon(cursors, 2700000)
        self.assertEqual(result["required_archive_end_exclusive"], 652300)
        self.assertEqual(result["planes_consumed_per_environment"], 12300)
        with self.assertRaises(ValueError):
            profile_horizon(cursors, 640000 + 96 * 50)

    def test_capacity_fallback_must_use_common_E(self):
        work = contract(16)
        self.assertEqual(work["global_collected_interactions"], 1536)
        for topology in (1, 2, 4):
            argv = profile_argv(["python", "-u", "trainer.py"], topology=topology,
                                rank=0, root=Path("/batch/smaller"), global_environments=16)
            self.assertEqual(int(flag(argv, "--local-environments")) * topology, 16)
            self.assertEqual(int(flag(argv, "--sample-size")) * topology, 256)
        with self.assertRaises(ValueError):
            contract(18)

    def test_prepare_uses_unchanged_chunks_and_new_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            manifest = {"schema_version": 1, "status": "complete", "dtype": "float32",
                        "plane_shape": [3, 41, 483], "completed_planes": 2700000,
                        "member_offsets": list(range(20000, 160001, 20000)),
                        "provenance": {"target_turbines": 81},
                        "chunks": [{"path": "chunks/chunk0.npy", "sha256": "unchanged", "bytes": 23}]}
            manifest_path = source / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            before = sha(manifest_path)
            recipes = root / "recipes.json"
            recipes.write_text(json.dumps({"training_runs": [{"ranks": [{"argv": ["python", "-u", "trainer.py"]}]}]}))
            args = SimpleNamespace(source_archive=source, recipes=recipes,
                                   output_root=root / "profile", global_environments=32)
            prepare(args)
            self.assertEqual(sha(manifest_path), before)
            protocol = json.loads((args.output_root / "PROFILE.json").read_text())
            for rank in protocol["ranks"]:
                self.assertEqual(rank["hard_wall_seconds"], 18000 / rank["topology"])
            for topology in (1, 2, 4):
                view = json.loads((args.output_root / ("p%d" % topology) / "archive" / "manifest.json").read_text())
                self.assertEqual(len(view["member_offsets"]), 32 // topology)
                self.assertEqual(view["chunks"][0]["sha256"], "unchanged")
                self.assertEqual(view["chunks"][0]["path"], str(source / "chunks/chunk0.npy"))
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_summary_uses_slowest_rank_and_rejects_failed_or_different_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root / "PROFILE.json"
            protocol.write_text(json.dumps(contract()))
            for topology in (1, 2, 4):
                directory = root / ("p%d" % topology)
                directory.mkdir()
                for rank in range(topology):
                    seconds = float(rank + 2)
                    report = {"schema": SCHEMA, "status": "pass", "checks": {"fixture": True},
                              "profile_protocol_sha256": sha(protocol),
                              "configuration": contract(), "normal_block_seconds": [seconds] * 10,
                              "normal_collector_seconds": [seconds / 2] * 10,
                              "normal_learner_and_sync_seconds": [seconds / 2] * 10,
                              "wrapper_total_seconds": 100.,
                              "reset_blocks": [{"wallclock_seconds": 40.}],
                              "complete_checkpoint_io": [{"seconds": 3.}],
                              "checkpoint_host_conversion_timing": [{"seconds": 2.}],
                              "actor_checkpoint_io": [], "global_barriers": [],
                              "fixed_action_physical_evidence_by_warm_block": [],
                              "memory_stats": {"peak_bytes_in_use": 100}}
                    (directory / ("rank%d.profile.json" % rank)).write_text(json.dumps(report))
                    (directory / ("rank%d.trainer.json" % rank)).write_text(json.dumps(
                        {"history": [{"mean_farm_power_mw": 90., "mean_abs_yaw_deg": 20.,
                                      "mean_abs_yaw_speed_deg_per_s": 0.} for _ in range(11)]}))
            args = SimpleNamespace(protocol=protocol, output=root / "summary.json")
            summarize(args)
            result = json.loads(args.output.read_text())
            self.assertEqual(result["topologies"]["2"]["normal_block_seconds_median"], 3.)
            self.assertAlmostEqual(result["topologies"]["2"]["regular_block_speedup"], 2 / 3)
            bad_path = root / "p4" / "rank3.profile.json"
            bad = json.loads(bad_path.read_text())
            bad["status"] = "incomplete_or_failed"
            bad_path.write_text(json.dumps(bad))
            args.output = root / "bad_summary.json"
            with self.assertRaises(ValueError):
                summarize(args)
            bad["status"] = "pass"
            bad["schema"] = "paper1_n81_bounded_computation_profile_v1"
            bad_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                summarize(args)
            bad["schema"] = SCHEMA
            bad["configuration"] = contract(16)
            bad_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                summarize(args)
            bad["configuration"] = contract()
            bad_path.write_text(json.dumps(bad))
            trainer_path = root / "p4" / "rank3.trainer.json"
            wrong = json.loads(trainer_path.read_text())
            wrong["history"][-1]["mean_abs_yaw_deg"] = 40.
            trainer_path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                summarize(args)


if __name__ == "__main__":
    unittest.main()
