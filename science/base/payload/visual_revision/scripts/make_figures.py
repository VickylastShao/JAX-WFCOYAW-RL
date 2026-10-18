#!/usr/bin/env python3
"""Regenerate five N81 figures from verified saved results, using CPU only.

The four mechanism layouts reuse the frozen original plotting code after
independent checks of their energy and yaw inputs. Figure bytes can differ
across Matplotlib/font versions; numeric agreement is the acceptance rule.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys


REFERENCES = {
    "scripts/frozen_plotting.py": "61fe9778fa99ef52923c65cb4e8592fe59e0f59d7cbce0c4ec347e9cef6cf2d3",
    "data/reference_mechanism_summary.json": "dc15aa052c61db36ee7c9b080460dc57497f09667af226e24617bceec1293083",
    "data/reference_P2_summary.json": "fe3f7ae2f30d16531dd8617a516e5eab05f7aa2bdacf2992913a51dac26a0f71",
}
NAMES = ("row_energy", "time_block_gain", "yaw_motion", "scaling_breakdown", "learning_curve_A")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root / "data") or output.is_relative_to(root / "source"):
        raise ValueError("figure outputs must not overwrite frozen package inputs")
    for name in NAMES:
        for suffix in ("pdf", "png"):
            if (output / f"{name}.{suffix}").exists():
                raise FileExistsError(output / f"{name}.{suffix}")
    for name, expected in REFERENCES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen figure input SHA differs: " + name)
    process = subprocess.run([sys.executable, str(root / "scripts/reanalyse.py"), "--root", str(root)],
                             text=True, capture_output=True, check=True)
    analysis = json.loads(process.stdout)
    if analysis["status"] != "pass":
        raise ValueError("L1 independent reanalysis did not pass")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def equal(left, right, label):
        if not np.allclose(left, right, rtol=1e-11, atol=1e-8):
            raise ValueError("figure statistic differs: " + label)

    summary = json.loads((root / "data/reference_mechanism_summary.json").read_text())
    metrics = analysis["metrics"]
    equal(metrics["equal_source_gain_percent"], summary["primary_equal_source_gain_percent"], "primary_gain")
    rates = []
    patterns = []
    for window in summary["windows"]:
        source, number = window["source_id"], window["window_number"]
        context = metrics["BC"][source]["windows"][number - 1]
        equal(context["gain_percent"], window["gain_percent"], "window_gain")
        equal(context["blocks_percent"], [b["gain_percent"] for b in window["time_blocks_500_controls"]], "fixed_blocks")
        with np.load(root / f"data/BC/{source}_w{number}/rl_r1.npz", allow_pickle=False) as trace:
            yaw = trace["yaw_deg"].astype(np.float64).reshape(2000, 81)
        increments = np.diff(np.concatenate([np.zeros((1, 81)), yaw]), axis=0)
        rates.append(np.abs(increments / 10).ravel())
        means = yaw.mean(axis=0)
        patterns.append(means)
        equal(means, window["yaw"]["mean_yaw_deg_per_turbine"], "yaw_mean")
        equal(means.reshape(9, 9).mean(axis=1), window["row_mean_yaw_deg"], "row_yaw")
        equal(np.abs(increments).sum(axis=0).mean() / (20000 / 3600),
              window["yaw"]["mean_command_travel_deg_per_turbine_hour"], "yaw_travel")
        tolerance = 4 * float(np.spacing(np.float32(40)))
        equal(np.mean(np.abs(yaw) >= 40 - tolerance), window["yaw"]["strict_angle_saturation_fraction"], "strict_40")
        equal(np.mean(np.abs(yaw) >= 38), window["yaw"]["near_angle_limit_fraction_abs_at_least_38_deg"], "near_38")
    equal(np.mean(patterns, axis=0).reshape(9, 9).mean(axis=1),
          summary["average_pattern"]["row_mean_yaw_deg"], "mean_row_pattern")
    for source in summary["sources"]:
        actual = metrics["BC"][source["source_id"]]
        equal(actual["gain_percent"], source["gain_percent"], "source_gain")
        # Integrate the row inputs directly so a saved summary is never
        # accepted as the numerical source of the mechanism figures.
        zero_rows, policy_rows = np.zeros(9), np.zeros(9)
        for number in range(1, 5):
            for controller, rows in (("zero", zero_rows), ("rl_r1", policy_rows)):
                with np.load(root / f'data/BC/{source["source_id"]}_w{number}/{controller}.npz', allow_pickle=False) as trace:
                    turbine = trace["turbine_power_mw"].astype(np.float64).reshape(2000, 9, 9)
                rows += turbine.sum(axis=(0, 2)) * 10 / 3600
        equal(policy_rows - zero_rows, source["row_gain_mwh"], "row_delta")
        equal(100 * (policy_rows / zero_rows - 1), source["row_gain_percent"], "row_gain")
    timing = json.loads((root / "data/timing/SCALING_SUMMARY.json").read_text())
    if timing != summary["efficiency"]:
        raise ValueError("scaling figure input differs from frozen timing source")
    spec = importlib.util.spec_from_file_location("n81_frozen_plotting", root / "scripts/frozen_plotting.py")
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    plotting.plots(summary, np.concatenate(rates), output)

    # Use the independently recomputed four-node curve, including the
    # nonmonotone later nodes, and validate its reference values first.
    p2reference = json.loads((root / "data/reference_P2_summary.json").read_text())
    nodes = metrics["P2"]
    for actual, recorded in zip(nodes, p2reference["nodes"], strict=True):
        equal(actual["interactions"], recorded["actual_training_interactions"], "P2_node")
        equal(actual["energy_MWh"], recorded["modeled_gross_ADM_energy_mwh"], "P2_energy")
        equal(actual["gain_percent"], recorded["modeled_gross_ADM_energy_gain_percent"], "P2_gain")
    x = [n["interactions"] for n in nodes]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(x, [n["gain_percent"] for n in nodes], "o-", color="#2b6f9c")
    axes[0].axhline(0, color="#555555", linewidth=.8)
    axes[0].set(xlabel="Actual global training interactions", ylabel="Modeled gross energy gain (%)",
                title="(a) All four saved-policy nodes")
    travel = []
    for n in nodes:
        file = "zero.npz" if n["interactions"] == 0 else f'rl_step{n["interactions"]}.npz'
        with np.load(root / "data/P2" / file, allow_pickle=False) as trace:
            yaw = trace["yaw_deg"].astype(np.float64).reshape(2000, 81)
        value = np.abs(np.diff(np.concatenate([np.zeros((1, 81)), yaw]), axis=0)).sum(axis=0).mean() / (20000/3600)
        travel.append(value)
    equal(travel, [n["command_equivalent_travel_deg_per_turbine_hour"] for n in p2reference["nodes"]], "P2_travel")
    axes[1].plot(x, travel, "s-", color="#c36b2a")
    axes[1].set(xlabel="Actual global training interactions", ylabel="Command travel (deg/turbine/h)",
                title="(b) Command motion on the same window")
    for axis in axes:
        axis.set_xticks(x, ["0", "131,072", "524,288", "1,000,448"], rotation=20)
    figure.text(.07, .01, "One source-A window; final budget actor retained. No checkpoint reselection.", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, 1))
    plotting.save_figure(figure, output, "learning_curve_A")
    print(json.dumps({"status": "pass", "gpu_used": False, "numeric_inputs_verified": True,
                      "figures": [str(output / (name + ".pdf")) for name in NAMES],
                      "plotting_environment": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                      "scope": "five figures regenerated from saved traces; no LES or full-state replay"}, indent=2))


if __name__ == "__main__":
    main()
