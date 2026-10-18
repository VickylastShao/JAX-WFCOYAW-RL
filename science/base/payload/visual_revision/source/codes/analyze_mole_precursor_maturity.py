#!/usr/bin/env python3
"""Apply a predeclared engineering stationarity screen to precursor runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCALAR_GATES = {
    "mean_hub_speed_m_per_s": {"cv": 0.03, "relative_drift": 0.05},
    "mean_hub_turbulence_intensity": {"cv": 0.10, "relative_drift": 0.15},
    "hub_turbulent_kinetic_energy_m2_per_s2": {
        "cv": 0.12,
        "relative_drift": 0.20,
    },
    "mean_wall_friction_velocity_proxy_m_per_s": {
        "cv": 0.06,
        "relative_drift": 0.10,
    },
}
CROSS_SEED_RELATIVE_DIFFERENCE_GATES = {
    "mean_hub_speed_m_per_s": 0.05,
    "mean_hub_turbulence_intensity": 0.15,
    "hub_turbulent_kinetic_energy_m2_per_s2": 0.20,
    "mean_wall_friction_velocity_proxy_m_per_s": 0.10,
}
SPECTRUM_L1_GATE = 0.20
DIVERGENCE_RMS_GATE = 1.0e-5


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_difference(left: float, right: float) -> float:
    denominator = max(0.5 * (abs(left) + abs(right)), 1.0e-20)
    return abs(left - right) / denominator


def _scalar_window(
    values: np.ndarray, completed_steps: np.ndarray
) -> dict[str, float]:
    mean = float(np.mean(values))
    cv = float(np.std(values, ddof=1) / max(abs(mean), 1.0e-20))
    normalized_steps = (completed_steps - completed_steps[0]) / max(
        completed_steps[-1] - completed_steps[0], 1.0
    )
    full_window_slope = float(np.polyfit(normalized_steps, values, 1)[0])
    relative_drift = abs(full_window_slope) / max(
        abs(mean), 1.0e-20
    )
    return {
        "mean": mean,
        "coefficient_of_variation": cv,
        "relative_linear_drift": float(relative_drift),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _mean_spectrum(history: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.mean(
        np.asarray([row["metrics"][key] for row in history]), axis=0
    )


def _cadence_history(
    payload: dict[str, Any], minimum_points: int = 0
) -> list[dict[str, Any]]:
    diagnostic_cadence = payload["configuration"]["chunk_steps"]
    invocation_rows = [
        row
        for row in payload.get("chunks_executed_this_invocation", [])
        if row.get("chunk_steps") == diagnostic_cadence and "metrics" in row
    ]
    if len(invocation_rows) >= minimum_points:
        return invocation_rows
    return [
        row
        for row in payload["diagnostic_history"]
        if row["completed_steps"] % diagnostic_cadence == 0
    ]


def analyze_run(
    path: Path,
    final_window_points: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    payload = _load(path)
    history = payload["diagnostic_history"]
    diagnostic_cadence = payload["configuration"]["chunk_steps"]
    required = 2 * final_window_points
    cadence_history = _cadence_history(payload, minimum_points=required)
    if len(cadence_history) < required:
        raise ValueError(
            f"{path} has {len(cadence_history)} cadence-aligned diagnostics; "
            f"at least {required} required"
        )
    previous = cadence_history[-required:-final_window_points]
    final = cadence_history[-final_window_points:]
    final_steps = np.asarray([row["completed_steps"] for row in final])
    scalars: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, gates in SCALAR_GATES.items():
        values = np.asarray([row["metrics"][key] for row in final])
        summary = _scalar_window(values, final_steps)
        scalars[key] = summary
        checks[f"{key}_cv"] = (
            summary["coefficient_of_variation"] <= gates["cv"]
        )
        checks[f"{key}_drift"] = (
            summary["relative_linear_drift"]
            <= gates["relative_drift"]
        )
    spectra: dict[str, Any] = {}
    for key in (
        "hub_streamwise_spectrum_normalized",
        "hub_spanwise_spectrum_normalized",
    ):
        previous_mean = _mean_spectrum(previous, key)
        final_mean = _mean_spectrum(final, key)
        distance = float(np.sum(np.abs(final_mean - previous_mean)))
        spectra[key] = {"previous_to_final_mean_l1": distance}
        checks[f"{key}_window_l1"] = distance <= SPECTRUM_L1_GATE
    endpoint = history[-1]
    divergence = np.asarray(
        [row["metrics"]["interior_divergence_rms"] for row in final]
        + [endpoint["metrics"]["interior_divergence_rms"]]
    )
    finite = all(row["metrics"]["finite"] for row in final) and endpoint[
        "metrics"
    ]["finite"]
    checks["final_window_finite"] = finite
    checks["final_window_divergence"] = bool(
        np.max(divergence) <= DIVERGENCE_RMS_GATE
    )
    return (
        {
            "path": str(path),
            "seed": payload["configuration"]["seed"],
            "completed_steps": history[-1]["completed_steps"],
            "diagnostic_cadence_steps": diagnostic_cadence,
            "final_window_steps": [row["completed_steps"] for row in final],
            "scalars": scalars,
            "spectra": spectra,
            "maximum_final_divergence_rms": float(np.max(divergence)),
        },
        checks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-window-points", type=int, default=5)
    parser.add_argument(
        "--evidence-boundary",
        default=(
            "Engineering screen declared before the 1,910,468-step results; "
            "it is not a proof of statistical stationarity or equality to "
            "the authors' private precursor ensemble."
        ),
    )
    args = parser.parse_args()
    if args.final_window_points < 2:
        raise ValueError("final-window-points must be at least 2")

    runs = []
    checks: dict[str, bool] = {}
    for index, path in enumerate(args.runs):
        run, run_checks = analyze_run(path, args.final_window_points)
        runs.append(run)
        checks.update(
            {f"run{index}_{key}": value for key, value in run_checks.items()}
        )

    cross_seed: dict[str, Any] = {"scalars": {}, "spectra": {}}
    for key, gate in CROSS_SEED_RELATIVE_DIFFERENCE_GATES.items():
        left = runs[0]["scalars"][key]["mean"]
        right = runs[1]["scalars"][key]["mean"]
        difference = _relative_difference(left, right)
        cross_seed["scalars"][key] = {
            "relative_difference": difference,
            "gate": gate,
        }
        checks[f"cross_seed_{key}"] = difference <= gate

    source_payloads = [_load(path) for path in args.runs]
    for key in (
        "hub_streamwise_spectrum_normalized",
        "hub_spanwise_spectrum_normalized",
    ):
        spectra = []
        for payload in source_payloads:
            cadence_history = _cadence_history(payload)
            final = cadence_history[-args.final_window_points :]
            spectra.append(_mean_spectrum(final, key))
        distance = float(np.sum(np.abs(spectra[0] - spectra[1])))
        cross_seed["spectra"][key] = {
            "mean_spectrum_l1": distance,
            "gate": SPECTRUM_L1_GATE,
        }
        checks[f"cross_seed_{key}"] = distance <= SPECTRUM_L1_GATE

    result = {
        "schema_version": 1,
        "purpose": "mole_precursor_engineering_stationarity_screen",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "evidence_boundary": args.evidence_boundary,
        "gates": {
            "scalar": SCALAR_GATES,
            "cross_seed_relative_difference": (
                CROSS_SEED_RELATIVE_DIFFERENCE_GATES
            ),
            "spectrum_l1": SPECTRUM_L1_GATE,
            "divergence_rms": DIVERGENCE_RMS_GATE,
            "final_window_points": args.final_window_points,
        },
        "checks": checks,
        "runs": runs,
        "cross_seed": cross_seed,
    }
    canonical = json.dumps(result, indent=2, sort_keys=True)
    result["payload_sha256_without_hash_field"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
