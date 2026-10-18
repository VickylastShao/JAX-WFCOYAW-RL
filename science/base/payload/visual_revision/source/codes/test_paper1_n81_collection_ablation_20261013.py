"""CPU-only contract checks; do not allocate an LES state or start a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import paper1_n81_collection_ablation_20261013 as component


class CollectionContractTests(unittest.TestCase):
    def test_exact_cadence_and_budget(self):
        c = component.contract()
        self.assertEqual(c["new_interactions_per_branch"], 512)
        self.assertEqual(c["new_learner_updates_per_branch"], 512)
        self.assertEqual(c["final_global_interactions"], 3584)
        self.assertEqual(c["pair_gpu_hour_cap"], 4)
        self.assertFalse(c["physical_reset_in_retained_blocks"])

    def test_branch_changes_only_dispatch_metadata_and_output(self):
        original = ["python", "-u", "codes/jax_mole_microbatch_distributed.py",
                    "--algorithm", "graph_sac_v2", "--rhs-assembly", "tensor",
                    "--collection-steps", "8", "--sample-size", "256"]
        argv = component.branch_argv(original, "/new", "/parent", "single_jit8")
        for option, expected in (("--collection-steps", "8"), ("--sample-size", "256"),
                                 ("--local-environments", "32"), ("--iterations", "2"),
                                 ("--rhs-assembly", "tensor"), ("--planned-global-interactions", "3584")):
            self.assertEqual(argv[argv.index(option) + 1], expected)
        self.assertNotIn("--resume-learner-only", argv)
        self.assertNotIn("--resume-allow-implementation-change", argv)

    def test_eight_dispatches_keep_same_inputs_rng_and_replay_order(self):
        calls = []
        summaries = ("mean_farm_power_mw", "mean_reward", "mean_angle_penalty",
                     "mean_abs_yaw_deg", "mean_abs_yaw_speed_deg_per_s",
                     "inlet_interval_delta_rms", "episode_reset")

        def one(state, planes, start):
            calls.append((state, planes, start))
            metrics = {key: float(state + 1) for key in summaries}
            metrics.update(replay_size=state + 1, sac_updates=0,
                           raw_transitions={"action": [state], "sequence_id": [state]})
            return state + 1, metrics

        wrapper = component.SingleJitEight(one, lambda xs: sum(xs) / len(xs),
                                           lambda xs: sum(xs, []))
        state, metrics = wrapper(0, "same_planes", 12300)
        self.assertEqual(state, 8)
        self.assertEqual(calls, [(i, "same_planes", 12300) for i in range(8)])
        self.assertEqual(metrics["mean_reward"], 4.5)
        self.assertEqual(metrics["replay_size"], 8)
        self.assertEqual(metrics["sac_updates"], 0)
        self.assertEqual(metrics["raw_transitions"]["action"], list(range(8)))

    def test_factory_does_not_change_learner_cadence_or_jit_host_loop(self):
        factory_calls = []
        jit_calls = []

        def factory(case, config, updates_per_iteration=0, collection_steps_per_iteration=8):
            factory_calls.append((updates_per_iteration, collection_steps_per_iteration))
            return lambda *args: args

        def jit(function, **kwargs):
            jit_calls.append(kwargs)
            return function

        split = component.collector_factory(factory, jit, "single_jit8", sum, sum)
        wrapped = split(None, None)
        self.assertEqual(factory_calls, [(0, 1)])
        self.assertEqual(jit_calls, [{"donate_argnums": (0,)}])
        routed = component.selective_jit(jit)(wrapped, donate_argnums=(0,))
        self.assertIs(routed, wrapped)
        self.assertEqual(len(jit_calls), 1)
        with self.assertRaises(ValueError):
            split(None, None, updates_per_iteration=1)
        with self.assertRaises(ValueError):
            split(None, None, collection_steps_per_iteration=1)

    def test_protocol_identity_and_scalar_contract_are_frozen(self):
        value = component.contract()
        value["wrapper_sha256"] = component.sha256(component.__file__)
        component.validate_protocol(value)
        value["collection_steps"] = 1
        with self.assertRaises(ValueError):
            component.validate_protocol(value)

    def test_cli_prepare_dry_run_requires_no_jax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = {"schema": "paper1_n81_bounded_computation_profile_v2",
                       "ranks": [{"topology": 1, "rank": 0, "trainer_argv": ["python", "-u", "codes/trainer.py"]}]}
            receipt = {"status": "pass", "topology": 1, "checks": {"a": True},
                       "complete_checkpoint_io": [{"sha256": component.PARENT_SHA256,
                                                  "bytes": component.PARENT_BYTES, "global_interactions": 3072}]}
            (root / "profile.json").write_text(json.dumps(profile))
            receipt["profile_protocol_sha256"] = component.sha256(root / "profile.json")
            (root / "receipt.json").write_text(json.dumps(receipt))
            result = subprocess.run([sys.executable, str(Path(component.__file__)), "prepare",
                                     "--profile-protocol", str(root / "profile.json"),
                                     "--profile-rank-result", str(root / "receipt.json"),
                                     "--parent", "/does/not/need/to/exist/for/preparation",
                                     "--output-root", str(root / "out"), "--dry-run"],
                                    check=True, capture_output=True, text=True)
            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["status"], "dry_run")
            self.assertEqual(len(parsed["protocol"]["branches"]), 2)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
