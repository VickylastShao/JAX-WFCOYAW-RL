#!/usr/bin/env python3
"""Aggregate and adjudicate the N=9 WP7 independent holdout matrix."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mole_n9_wp6_contract import payload_sha256_without_hash
from mole_n9_wp7_contract import MIDPOINT_DIRECTIONS_DEG, TRAINING_DIRECTIONS_DEG, WP7_OFFSETS


SELECTED_ACTOR_SHA256 = "a0474f942c93a296121cfff7c317e861dfbb05369f5cbd42c39aaaca495703b9"
CONTROLLER_ROLES = {
    "zero_yaw": "formal_reference",
    "neutral_actor_step32768": "formal_learning_baseline",
    "static_les_lookup": "formal_open_loop_baseline",
    "selected_policy_direction_mean": "formal_open_loop_baseline",
    "selected_realtime_policy": "formal_controller",
    "selected_probe_mean_ablation": "formal_state_ablation",
    "cross_precursor_action_replay": "formal_feedback_ablation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a1", "a2", "combined"), required=True)
    parser.add_argument("--results-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("purpose") != "mole_n9_wp7_formal_direction_result":
        raise ValueError(f"unexpected WP7 purpose in {path}")
    if result.get("payload_sha256_without_hash_field") != payload_sha256_without_hash(result):
        raise ValueError(f"payload hash mismatch in {path}")
    return result


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reports = [
        load_result(path)
        for directory in args.results_dir
        for path in sorted(directory.glob("*.json"))
    ]
    expected_directions = (
        TRAINING_DIRECTIONS_DEG
        if args.stage == "a1"
        else MIDPOINT_DIRECTIONS_DEG
        if args.stage == "a2"
        else TRAINING_DIRECTIONS_DEG + MIDPOINT_DIRECTIONS_DEG
    )
    expected_stages = (
        {"wp7-a1"}
        if args.stage == "a1"
        else {"wp7-a2"}
        if args.stage == "a2"
        else {"wp7-a1", "wp7-a2"}
    )
    by_direction = {
        float(report["configuration"]["relative_wind_angle_deg"]): report
        for report in reports
    }
    matrix_complete = (
        len(reports) == len(expected_directions)
        and set(by_direction) == set(expected_directions)
        and {report["configuration"]["evaluation_stage"] for report in reports}
        == expected_stages
    )

    report_contracts = []
    rows: list[dict[str, Any]] = []
    controllers: list[dict[str, Any]] = []
    for controller_id, role in CONTROLLER_ROLES.items():
        direction_means: dict[str, float] = {}
        controller_windows: list[dict[str, Any]] = []
        for direction in expected_directions:
            report = by_direction.get(direction)
            if report is None:
                continue
            gains = []
            for window in report["windows"]:
                metrics = window["controllers"][controller_id]
                gain = float(metrics["gain_percent"])
                gains.append(gain)
                controller_windows.append(metrics)
                rows.append(
                    {
                        "direction_deg": direction,
                        "start_plane": int(window["start_plane"]),
                        "controller_id": controller_id,
                        "gain_percent": gain,
                        "mean_farm_power_mw": float(metrics["mean_farm_power_mw"]),
                        "mean_turbine_power_mw": metrics["mean_turbine_power_mw"],
                        "mean_yaw_deg": metrics["mean_yaw_deg"],
                        "mean_abs_yaw_deg": float(metrics["mean_abs_yaw_deg"]),
                        "mean_yaw_travel_deg_per_turbine": float(metrics["mean_yaw_travel_deg_per_turbine"]),
                        "peak_yaw_rate_deg_per_s": float(metrics["peak_yaw_rate_deg_per_s"]),
                        "action_saturation_fraction": float(metrics["action_saturation_fraction"]),
                    }
                )
            direction_means[str(direction)] = float(np.mean(gains))
        if direction_means:
            controllers.append(
                {
                    "controller_id": controller_id,
                    "evidence_role": role,
                    "equal_direction_mean_gain_percent": float(np.mean(list(direction_means.values()))),
                    "worst_direction_mean_gain_percent": float(min(direction_means.values())),
                    "negative_direction_count": int(sum(value < -0.001 for value in direction_means.values())),
                    "negative_window_count": int(
                        sum(
                            row["gain_percent"] < -0.001
                            for row in rows
                            if row["controller_id"] == controller_id
                        )
                    ),
                    "mean_abs_yaw_deg": float(np.mean([item["mean_abs_yaw_deg"] for item in controller_windows])),
                    "mean_yaw_travel_deg_per_turbine": float(
                        np.mean([item["mean_yaw_travel_deg_per_turbine"] for item in controller_windows])
                    ),
                    "peak_yaw_rate_deg_per_s": float(
                        max(item["peak_yaw_rate_deg_per_s"] for item in controller_windows)
                    ),
                    "action_saturation_fraction": float(
                        np.mean([item["action_saturation_fraction"] for item in controller_windows])
                    ),
                    "direction_mean_gain_percent": direction_means,
                }
            )

    for report in reports:
        report_contracts.append(
            report.get("status") == "pass"
            and report.get("selection_performed") is False
            and report.get("sealed_material_accessed") is True
            and report["identity"]["source_state_sha256"]
            == reports[0]["identity"]["source_state_sha256"]
            and report["identity"]["selected_actor_sha256"] == SELECTED_ACTOR_SHA256
            and report["configuration"]["offsets"] == list(WP7_OFFSETS)
            and len(report["windows"]) == 2
            and all(
                set(window["controllers"]) == set(CONTROLLER_ROLES)
                and finite(window["controllers"])
                and window["cross_replay_binding"]["status"] == "pass"
                for window in report["windows"]
            )
        )

    by_id = {item["controller_id"]: item for item in controllers}
    realtime = by_id.get("selected_realtime_policy", {})
    all_realtime_directions_positive = bool(realtime) and all(
        value > 0.0 for value in realtime["direction_mean_gain_percent"].values()
    )
    stage_gate_pass = (
        matrix_complete
        and all(report_contracts)
        and float(realtime.get("equal_direction_mean_gain_percent", -math.inf)) > 0.0
        and (args.stage == "a1" or all_realtime_directions_positive)
    )
    paired = {}
    if realtime:
        for baseline in (
            "static_les_lookup",
            "selected_policy_direction_mean",
            "selected_probe_mean_ablation",
            "cross_precursor_action_replay",
        ):
            paired[f"selected_realtime_policy_minus_{baseline}"] = {
                "equal_direction_mean_gain_difference_percentage_points": (
                    realtime["equal_direction_mean_gain_percent"]
                    - by_id[baseline]["equal_direction_mean_gain_percent"]
                )
            }

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "mole_n9_wp7_stage_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if stage_gate_pass else "fail",
        "stage": args.stage,
        "checks": {
            "matrix_complete": matrix_complete,
            "all_reports_pass_contract": all(report_contracts),
            "selected_realtime_equal_direction_mean_positive": float(
                realtime.get("equal_direction_mean_gain_percent", -math.inf)
            )
            > 0.0,
            "selected_realtime_all_direction_means_positive": all_realtime_directions_positive,
        },
        "stage_gate_pass": stage_gate_pass,
        "independent_precursor_seed": 20260904,
        "statistical_unit": "independent_precursor_seed",
        "inferential_interval_reported": False,
        "selected_actor_sha256": SELECTED_ACTOR_SHA256,
        "controllers": controllers,
        "paired_comparisons": paired,
        "window_rows": rows,
        "selection_performed": False,
        "sealed_material_accessed": True,
        "evidence_boundary": (
            "One independent precursor seed with two deterministic windows; report the seed-level effect without a population confidence interval."
        ),
    }
    result["payload_sha256_without_hash_field"] = payload_sha256_without_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "stage": args.stage, "output": str(args.output)}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
