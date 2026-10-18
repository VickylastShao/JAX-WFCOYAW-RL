"""NumPy/stdlib regressions for identity-bound N81 confirmation summaries.

All data below are synthetic accounting fixtures, never LES research evidence.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from mole_large_scale_metrics import paired_gain, score_trace
from summarize_mole_n81_final_evaluation import (
    N81_LAYOUT_SHA256, read_json, render_markdown, sha256_file, summarize,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


class SummaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.batch = Path(cls.temporary.name)
        cls.plan = {
            "task_id": "synthetic-summary-contracts",
            "actor_final": {"sha256": "a" * 64, "actual_global_interactions": 1000448},
            "sources": [{"source_id": name, "seed": 100 + index,
                         "source_state_sha256": str(index + 1) * 64,
                         "manifest_sha256": str(index + 3) * 64, "completed_planes": 520000}
                        for index, name in enumerate(("B", "C"))],
            "evaluation": {"burnin": 150, "scored_interactions": 2000,
                           "control_dt": 10., "les_steps_per_control": 50,
                           "primary_confidence_interval": False,
                           "negative_gains_retained": True, "minimum_positive_gain_threshold": None,
                           "training_seeds_evaluated": 1, "independent_sources": 2},
            "contexts": [],
        }
        cls.results = {}
        for source_index, source in enumerate(cls.plan["sources"]):
            for window, start in enumerate((20000, 147500, 275000, 402500), 1):
                context_id = "n81_%s_w%d" % (source["source_id"], window)
                context = {"context_id": context_id, "source_id": source["source_id"],
                           "window_start": start, "window_end_exclusive": start + 107500}
                cls.plan["contexts"].append(context)
                directory = cls.batch / "contexts" / context_id
                directory.mkdir(parents=True)
                (directory / "common_state.pkl").write_bytes(("synthetic-common-" + context_id).encode())
                common_hash = sha256_file(directory / "common_state.pkl")
                baseline = 20. * window * (1 if source["source_id"] == "B" else 4)
                percent = [-10, 10, 10, 15][window - 1] if source["source_id"] == "B" else -5
                result = {"status": "complete", "context_id": context_id,
                          "turbines": 81, "layout_sha256": N81_LAYOUT_SHA256,
                          "source_state": {"seed": source["seed"], "sha256": source["source_state_sha256"]},
                          "archive_manifest_sha256": source["manifest_sha256"],
                          "window_start": start, "window_end_exclusive": start + 107500,
                          "common_state_sha256": common_hash, "scores": {}, "controllers": {}, "pairs": {}}
                for branch in ("zero", "rl_r1"):
                    mean_power = baseline * 3600 / 20000 * (1 if branch == "zero" else 1 + percent / 100)
                    # A zero-mean fluctuation keeps the requested exact energy.
                    fluctuation = (np.arange(2000) % 2 * 2 - 1) * (source_index * 4 + window) * 1e-4
                    farm_power = (mean_power + fluctuation)[:, None]
                    turbines = np.repeat((farm_power / 81)[..., None], 81, axis=2)
                    yaw = np.zeros((2000, 1, 81))
                    if branch == "rl_r1":
                        yaw[:] = window * 0.1
                    trace = {"farm_power_mw": farm_power, "turbine_power_mw": turbines, "yaw_deg": yaw}
                    path = directory / (branch + ".npz")
                    np.savez(path, **trace)
                    result["scores"][branch] = score_trace(trace, np.zeros((1, 81)))
                    actor = None if branch == "zero" else {
                        "sha256": "a" * 64,
                        "metadata": {"actual_global_interactions": 1000448,
                                     "policy_architecture": "graph_sac_v2", "action_parameterization": "per_turbine"}}
                    result["controllers"][branch] = {
                        "trace": "/batch/contexts/" + context_id + "/" + branch + ".npz",
                        "trace_sha256": sha256_file(path), "actor": actor,
                        "common_state_sha256": common_hash, "scoring_cursor": start + 7500,
                        "final_cursor_exclusive": start + 107500}
                    if branch == "zero":
                        burnin_path = directory / "common_burnin.npz"
                        np.savez(burnin_path, **{key: value[:150] for key, value in trace.items()})
                        result["common_burnin"] = {
                            "trace": "/batch/contexts/" + context_id + "/common_burnin.npz",
                            "trace_sha256": sha256_file(burnin_path), "controls": 150,
                            "duration_seconds": 1500, "initial_plane_cursor": start,
                            "final_plane_cursor_exclusive": start + 7500, "excluded_from_scoring": True}
                result["pairs"]["rl_r1"] = paired_gain(result["scores"]["rl_r1"], result["scores"]["zero"])
                cls.results[context_id] = result

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        write_json(self.batch / "PLAN.json", self.plan)
        for context_id, result in self.results.items():
            write_json(self.batch / "contexts" / context_id / "RESULT.json", result)

    def change_result(self, mutate, context_id="n81_B_w1"):
        result = copy.deepcopy(self.results[context_id])
        mutate(result)
        write_json(self.batch / "contexts" / context_id / "RESULT.json", result)

    def assert_invalid(self, phrase):
        summary = summarize(self.batch)
        self.assertEqual(summary["status"], "invalid")
        self.assertIsNone(summary["primary_equal_source_gain_percent"])
        self.assertTrue(any(phrase in item["error"] for item in summary["invalid_contexts"]), summary)

    def test_complete_source_weighting_keeps_negative_windows_and_sources(self):
        summary = summarize(self.batch)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["validated_trajectories"], 16)
        self.assertAlmostEqual(summary["sources"][0]["generation_gain_percent"], 10)
        self.assertAlmostEqual(summary["sources"][1]["generation_gain_percent"], -5)
        self.assertAlmostEqual(summary["primary_equal_source_gain_percent"], 2.5)
        self.assertAlmostEqual(summary["pooled_secondary"]["generation_gain_percent"], -2)
        self.assertAlmostEqual(summary["pooled_secondary"]["absolute_generation_gain_mwh"], -20)
        self.assertEqual(summary["negative_sources"], ["C"])
        self.assertEqual(len(summary["negative_contexts"]), 5)
        self.assertIsNone(summary["primary_confidence_interval"])
        self.assertFalse(summary["bootstrap_performed"])
        self.assertEqual(len(summary["sources"][0]["rl_yaw_travel_deg_per_turbine"]), 81)
        self.assertIn("负收益来源：C", render_markdown(summary))

    def test_missing_context_is_incomplete_with_no_full_estimate(self):
        (self.batch / "contexts/n81_C_w4/RESULT.json").unlink()
        summary = summarize(self.batch)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["missing_contexts"], ["n81_C_w4"])
        self.assertEqual(summary["validated_trajectories"], 14)
        self.assertIsNone(summary["primary_equal_source_gain_percent"])
        self.assertEqual(summary["sources"], [])

    def test_context_source_manifest_and_window_mismatch_are_rejected(self):
        mutations = [
            (lambda r: r.update(context_id="n81_B_w2"), "identity"),
            (lambda r: r["source_state"].update(seed=999), "source seed/state"),
            (lambda r: r["source_state"].update(sha256="f" * 64), "source seed/state"),
            (lambda r: r.update(archive_manifest_sha256="e" * 64), "manifest"),
            (lambda r: r.update(window_end_exclusive=127499), "window"),
        ]
        for mutate, phrase in mutations:
            with self.subTest(phrase=phrase):
                self.change_result(mutate)
                self.assert_invalid(phrase)

    def test_branch_actor_common_hash_cursor_and_duration_are_rejected(self):
        mutations = [
            (lambda r: r["controllers"]["rl_r1"]["actor"].update(sha256="b" * 64), "Actor"),
            (lambda r: r["controllers"]["rl_r1"]["actor"]["metadata"].update(actual_global_interactions=524288), "Actor"),
            (lambda r: r["controllers"]["rl_r1"].update(common_state_sha256="f" * 64), "common state"),
            (lambda r: r["controllers"]["rl_r1"].update(scoring_cursor=27501), "cursor"),
            (lambda r: r["scores"]["rl_r1"].update(duration_seconds=4000), "duration"),
            (lambda r: r["pairs"]["rl_r1"].update(generation_gain_percent=50), "scored trace"),
        ]
        for mutate, phrase in mutations:
            with self.subTest(phrase=phrase):
                self.change_result(mutate)
                self.assert_invalid(phrase)

    def test_trace_and_common_file_hash_are_independently_checked(self):
        self.change_result(lambda r: r["controllers"]["rl_r1"].update(trace_sha256="f" * 64))
        self.assert_invalid("trace hash")
        self.change_result(lambda r: r.update(common_state_sha256="f" * 64))
        self.assert_invalid("common state hash")

    def test_common_burnin_is_bound_and_excluded(self):
        mutations = [
            (lambda r: r["common_burnin"].update(controls=400), "burnin duration"),
            (lambda r: r["common_burnin"].update(final_plane_cursor_exclusive=27501), "burnin duration"),
            (lambda r: r["common_burnin"].update(excluded_from_scoring=False), "burnin duration"),
            (lambda r: r["common_burnin"].update(trace_sha256="f" * 64), "burnin trace hash"),
        ]
        for mutate, phrase in mutations:
            with self.subTest(phrase=phrase):
                self.change_result(mutate)
                self.assert_invalid(phrase)

    def test_duplicate_context_and_trace_path_are_not_counted(self):
        plan = copy.deepcopy(self.plan)
        plan["contexts"][1] = copy.deepcopy(plan["contexts"][0])
        write_json(self.batch / "PLAN.json", plan)
        self.assert_invalid("distinct contexts")
        write_json(self.batch / "PLAN.json", self.plan)
        self.change_result(lambda r: r["controllers"]["rl_r1"].update(trace=r["controllers"]["zero"]["trace"]))
        self.assert_invalid("trace path")

    def test_hardlinked_trajectory_is_rejected_but_equal_independent_traces_are_retained(self):
        directory = self.batch / "contexts/n81_B_w1"
        zero, rl = directory / "zero.npz", directory / "rl_r1.npz"
        original = rl.read_bytes()
        result = copy.deepcopy(self.results["n81_B_w1"])
        result["scores"]["rl_r1"] = copy.deepcopy(result["scores"]["zero"])
        result["pairs"]["rl_r1"] = paired_gain(result["scores"]["zero"], result["scores"]["zero"])
        result["controllers"]["rl_r1"]["trace_sha256"] = result["controllers"]["zero"]["trace_sha256"]
        write_json(directory / "RESULT.json", result)
        try:
            rl.unlink()
            os.link(zero, rl)
            self.assert_invalid("file/inode")
            rl.unlink()
            shutil.copyfile(zero, rl)
            summary = summarize(self.batch)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["windows"][0]["gain"]["generation_gain_percent"], 0)
            self.assertEqual(len(summary["identical_trace_content_groups"]), 1)
        finally:
            rl.unlink()
            rl.write_bytes(original)

    def test_confirmatory_ci_and_pilot_horizon_cannot_silently_replace_contract(self):
        plan = copy.deepcopy(self.plan)
        plan["evaluation"]["primary_confidence_interval"] = True
        write_json(self.batch / "PLAN.json", plan)
        self.assert_invalid("confidence interval")
        plan["evaluation"]["primary_confidence_interval"] = False
        plan["evaluation"]["scored_interactions"] = 400
        write_json(self.batch / "PLAN.json", plan)
        self.assert_invalid("150/2000")

    def test_nonfinite_json_and_duplicate_keys_are_rejected(self):
        path = self.batch / "contexts/n81_B_w1/RESULT.json"
        path.write_text('{"status":"complete","context_id":"n81_B_w1","x":NaN}')
        self.assert_invalid("nonfinite JSON")
        path.write_text('{"status":"complete","status":"complete"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            read_json(path)


if __name__ == "__main__":
    unittest.main()
