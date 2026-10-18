#!/usr/bin/env python3
"""CPU analysis of frozen N81 trajectories; no model runs or policy selection.

Energy is modeled gross ADM extraction, not electrical generator output. Yaw motion/rate is a command equivalent
derived from successive recorded control-boundary angles, not measured drive
motion. Physical sources, rather than four windows, are the replicate units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMA = "paper1_n81_existing_mechanism_analysis_20261013_v1"
DT = 10.0
N = 2000
TURBINES = 81
ENERGY_FACTOR = DT / 3600.0
ANGLE_LIMIT = 40.0
ANGLE_TOL = float(4 * np.spacing(np.float32(ANGLE_LIMIT)))
RATE_TOL = ANGLE_TOL / DT
COLORS = {"B": "#2b6f9c", "C": "#c36b2a"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def gain(zero, controlled):
    if float(zero) <= 0:
        raise ValueError("nonpositive reference energy")
    return 100.0 * (float(controlled) / float(zero) - 1.0)


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}


def window_energy(trace, start=0, stop=N):
    return float(np.sum(trace["farm_power_mw"][start:stop], dtype=np.float64) * ENERGY_FACTOR)


def reversal_counts(increments):
    # Removing changes below four FP32 ULPs of the 40-degree limit avoids
    # interpreting numerical signed noise as a command reversal. Zeros do not
    # interrupt the previous nonzero direction. It is not a fatigue threshold.
    counts = []
    for column in increments.T:
        nonzero = column[np.abs(column) > ANGLE_TOL]
        counts.append(int(np.sum(np.sign(nonzero[1:]) != np.sign(nonzero[:-1]))))
    return np.asarray(counts, dtype=np.int64)


def yaw_metrics(yaw, normalized_action):
    previous = np.vstack((np.zeros((1, TURBINES)), yaw[:-1]))
    increments = yaw - previous
    equivalent_rate = increments / DT
    travel = np.sum(np.abs(increments), axis=0)
    hours = yaw.shape[0] * DT / 3600.0
    mean = np.mean(yaw, axis=0)
    time_sd = np.std(yaw, axis=0, ddof=0)
    reversals = reversal_counts(increments)
    commanded_end = np.clip(previous + DT * normalized_action, -ANGLE_LIMIT, ANGLE_LIMIT)
    static_ms = float(np.mean(mean ** 2))
    temporal_variance = float(np.mean(time_sd ** 2))
    return {
        "yaw_travel_deg_per_turbine": travel.tolist(),
        "mean_command_travel_deg_per_turbine_hour": float(np.mean(travel) / hours),
        "command_reversal_counts_per_turbine": reversals.tolist(),
        "mean_command_reversals_per_turbine_hour": float(np.mean(reversals) / hours),
        "mean_yaw_deg_per_turbine": mean.tolist(),
        "temporal_yaw_sd_deg_per_turbine": time_sd.tolist(),
        "mean_temporal_yaw_sd_deg": float(np.mean(time_sd)),
        "mean_absolute_yaw_deg": float(np.mean(np.abs(yaw))),
        "mean_absolute_command_equivalent_rate_deg_s": float(np.mean(np.abs(equivalent_rate))),
        "maximum_command_equivalent_rate_deg_s": float(np.max(np.abs(equivalent_rate))),
        "maximum_absolute_yaw_deg": float(np.max(np.abs(yaw))),
        "strict_angle_saturation_fraction": float(np.mean(np.abs(yaw) >= ANGLE_LIMIT - ANGLE_TOL)),
        "near_angle_limit_fraction_abs_at_least_38_deg": float(np.mean(np.abs(yaw) >= 38.0)),
        "strict_angle_saturation_fraction_per_turbine": np.mean(np.abs(yaw) >= ANGLE_LIMIT - ANGLE_TOL, axis=0).tolist(),
        "command_rate_near_limit_fraction_abs_at_least_0_95_deg_s": float(np.mean(np.abs(equivalent_rate) >= 0.95)),
        "command_reconstruction_max_abs_error_deg": float(np.max(np.abs(commanded_end - yaw))),
        "squared_angle_fraction_of_temporal_mean_pattern": static_ms / (static_ms + temporal_variance),
        "temporal_mean_pattern_fraction_definition": "mean_i(mean_t(yaw_ti)^2) / mean_ti(yaw_ti^2); geometry descriptor, not fraction of energy gain explained",
        "mean_pattern_mean_square_deg2": static_ms,
        "mean_temporal_variance_deg2": temporal_variance,
        "angle_comparison_tolerance_deg": ANGLE_TOL,
        "rate_comparison_tolerance_deg_s": RATE_TOL,
        "physical_drive_energy_or_fatigue_measured": False,
    }, equivalent_rate


def temporal_sources(root):
    result = {}
    all_rows = {}
    for source, name in (("A", "n81_train_A"), ("B", "n81_heldout_B"), ("C", "n81_heldout_C")):
        directory = root / "source" / "archives" / name / "temporal"
        paths = sorted(directory.glob("block*.json"))
        rows = [json.loads(path.read_text()) for path in paths]
        if not rows or any(rows[i]["stop"] != rows[i + 1]["start"] for i in range(len(rows) - 1)):
            raise ValueError("temporal source block coverage is absent or discontinuous")
        durations = np.array([row["duration_seconds"] for row in rows])
        means = np.array([row["mean_hub_u_m_per_s"] for row in rows])
        ti = np.array([row["mean_spatial_ti"] for row in rows])
        result[source] = {
            "archive": name, "blocks": len(rows), "duration_seconds": float(durations.sum()),
            "hub_mean_u_m_s": float(np.average(means, weights=durations)),
            "hub_block_mean_range_m_s": [float(means.min()), float(means.max())],
            "hub_first_block_mean_u_m_s": float(means[0]), "hub_last_block_mean_u_m_s": float(means[-1]),
            "hub_first_to_last_change_m_s": float(means[-1] - means[0]),
            "mean_spatial_ti_proxy": float(np.average(ti, weights=durations)),
            "spatial_ti_block_range": [float(ti.min()), float(ti.max())],
            "plane_mean_hub_time_sd_range_m_s": [float(min(row["std_plane_mean_hub_u_m_per_s"] for row in rows)),
                                                  float(max(row["std_plane_mean_hub_u_m_per_s"] for row in rows))],
            "evidence_boundary": "spatial TI is not temporal point TI; block/subblock means are correlated along one precursor source; no stationarity or CV acceptance gate",
            "block_records": [{"start": row["start"], "stop": row["stop"],
                               "mean_hub_u_m_per_s": row["mean_hub_u_m_per_s"],
                               "mean_spatial_ti": row["mean_spatial_ti"]} for row in rows],
            "input_hashes": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        }
        all_rows[source] = rows
    return result, all_rows


def overlap_hub_proxy(rows, start, stop):
    weighted_u = weighted_ti = covered = 0.0
    for row in rows:
        for sub in row["subblocks"]:
            left = row["start"] + sub["local_start"]
            right = row["start"] + sub["local_stop"]
            width = max(0, min(stop, right) - max(start, left))
            weighted_u += width * sub["mean_hub_u_m_per_s"]
            weighted_ti += width * sub["mean_spatial_ti"]
            covered += width
    if covered != stop - start:
        raise ValueError("scoring cursor is not covered by temporal subblock records")
    return {"mean_hub_u_proxy_m_s": weighted_u / covered,
            "mean_spatial_ti_proxy": weighted_ti / covered,
            "start_plane": start, "stop_plane_exclusive": stop,
            "definition": "overlap-weighted 5000-plane subblock mean; partial subblock overlaps are approximate, not exact scoring-window hub means"}


def analyze(args):
    root = args.existing
    evaluation = root / "evaluation"
    old = json.loads((evaluation / "EVALUATION_SUMMARY.json").read_text())
    plan = json.loads((evaluation / "PLAN.json").read_text())
    if old["status"] != "complete" or plan["evaluation"]["control_dt"] != DT:
        raise ValueError("frozen complete evaluation with dt=10s required")
    rank_paths = [root / "source/runs/n81_train_r1" / f"rank{i}.json" for i in range(4)]
    ranks = [json.loads(path.read_text()) for path in rank_paths]
    layout = ranks[0]["configuration"]["layout_contract"]
    positions = np.array(layout["turbine_positions_m"], dtype=np.float64)
    if positions.shape != (TURBINES, 3):
        raise ValueError("actual turbine roster differs")
    row_x = np.unique(positions[:, 0])
    groups = [np.flatnonzero(positions[:, 0] == x) for x in row_x]
    if len(groups) != 9 or any(len(group) != 9 for group in groups):
        raise ValueError("actual streamwise coordinates do not resolve nine rows")
    temporal, temporal_rows = temporal_sources(root)
    windows = []
    hashes = []
    rates_for_plot = []
    gain_matrices = []
    checks = {"all_trace_shapes_and_values_finite": True,
              "all_inputs_match_frozen_hashes": True,
              "actual_positions_match_all_context_layout_ids": True,
              "same_inlet_trace_in_paired_branches": True,
              "yaw_command_reconstruction": True,
              "all_window_scores_reproduced": True,
              "farm_sum_matches_per_turbine_sum": True,
              "all_zero_reference_commands_zero": True}
    for source in ("B", "C"):
        for number in range(1, 5):
            name = f"n81_{source}_w{number}"
            directory = evaluation / "contexts" / name
            result_path = directory / "RESULT.json"
            result = json.loads(result_path.read_text())
            expected = next(window for window in old["windows"] if window["context_id"] == name)
            checks["actual_positions_match_all_context_layout_ids"] &= result["layout_sha256"] == layout["declared_layout_sha256"]
            if sha256(result_path) != expected["result_sha256"]:
                raise ValueError("frozen context result hash differs: " + name)
            traces = {}
            for mode in ("zero", "rl_r1"):
                path = directory / (mode + ".npz")
                digest = sha256(path)
                if digest != result["controllers"][mode]["trace_sha256"]:
                    raise ValueError("frozen trace hash differs: " + str(path))
                hashes.append({"path": str(path), "sha256": digest})
                traces[mode] = load_trace(path)
                shapes = {"farm_power_mw": (N, 1), "turbine_power_mw": (N, 1, TURBINES), "yaw_deg": (N, 1, TURBINES)}
                checks["all_trace_shapes_and_values_finite"] &= all(traces[mode][key].shape == shape for key, shape in shapes.items())
                checks["all_trace_shapes_and_values_finite"] &= all(np.isfinite(value).all() for value in traces[mode].values())
                checks["all_window_scores_reproduced"] &= abs(window_energy(traces[mode]) - result["scores"][mode]["generation_energy_mwh"]) < 1.0e-9
                farm = traces[mode]["farm_power_mw"][:, 0]
                per = traces[mode]["turbine_power_mw"][:, 0]
                checks["farm_sum_matches_per_turbine_sum"] &= bool(np.allclose(farm, per.sum(1), atol=2e-5, rtol=1e-6))
            zero, rl = traces["zero"], traces["rl_r1"]
            checks["same_inlet_trace_in_paired_branches"] &= np.array_equal(zero["inlet_mean_u"], rl["inlet_mean_u"]) and np.array_equal(zero["inlet_delta_rms"], rl["inlet_delta_rms"])
            checks["all_zero_reference_commands_zero"] &= bool(np.all(zero["yaw_deg"] == 0))
            yaw, rates = yaw_metrics(rl["yaw_deg"][:, 0], rl["normalized_action"][:, 0])
            checks["yaw_command_reconstruction"] &= yaw["command_reconstruction_max_abs_error_deg"] <= ANGLE_TOL
            rates_for_plot.append(np.abs(rates).reshape(-1))
            ez = window_energy(zero)
            er = window_energy(rl)
            blocks = []
            for index, start in enumerate(range(0, N, 500)):
                stop = start + 500
                bz, br = window_energy(zero, start, stop), window_energy(rl, start, stop)
                blocks.append({"block": index + 1, "start_control_inclusive": start, "stop_control_exclusive": stop,
                               "zero_energy_mwh": bz, "rl_energy_mwh": br, "gain_percent": gain(bz, br),
                               "absolute_gain_mwh": br - bz,
                               "zero_mean_power_mw": bz / (500 * DT / 3600),
                               "rl_mean_power_mw": br / (500 * DT / 3600),
                               "exact_plane_mean_inlet_u_m_s": float(np.mean(zero["inlet_mean_u"][start:stop]))})
            gain_matrices.append([block["gain_percent"] for block in blocks])
            turbine_z = np.sum(zero["turbine_power_mw"][:, 0], axis=0) * ENERGY_FACTOR
            turbine_r = np.sum(rl["turbine_power_mw"][:, 0], axis=0) * ENERGY_FACTOR
            rowz = np.array([turbine_z[group].sum() for group in groups])
            rowr = np.array([turbine_r[group].sum() for group in groups])
            lz, lr = window_energy(zero, 1000), window_energy(rl, 1000)
            cursor = result["controllers"]["zero"]["scoring_cursor"]
            if cursor != result["controllers"]["rl_r1"]["scoring_cursor"]:
                raise ValueError("paired scoring cursor differs")
            burn_path = directory / "common_burnin.npz"
            if sha256(burn_path) != result["common_burnin"]["trace_sha256"]:
                raise ValueError("common burnin trace hash differs")
            burn = load_trace(burn_path)["farm_power_mw"][:, 0]
            windows.append({"context_id": name, "source_id": source, "window_number": number,
                            "zero_energy_mwh": ez, "rl_energy_mwh": er,
                            "gain_percent": gain(ez, er), "absolute_gain_mwh": er - ez,
                            "time_blocks_500_controls": blocks,
                            "last1000": {"zero_energy_mwh": lz, "rl_energy_mwh": lr, "gain_percent": gain(lz, lr)},
                            "row_zero_energy_mwh": rowz.tolist(), "row_rl_energy_mwh": rowr.tolist(),
                            "row_gain_mwh": (rowr - rowz).tolist(),
                            "row_gain_percent": (100 * (rowr / rowz - 1)).tolist(),
                            "row_mean_yaw_deg": [float(np.mean(np.array(yaw["mean_yaw_deg_per_turbine"])[group])) for group in groups],
                            "row_mean_per_turbine_temporal_yaw_sd_deg": [float(np.mean(np.array(yaw["temporal_yaw_sd_deg_per_turbine"])[group])) for group in groups],
                            "yaw": yaw,
                            "inlet_exact_plane_mean_u_m_s": float(np.mean(zero["inlet_mean_u"])),
                            "inlet_exact_plane_mean_u_time_sd_m_s": float(np.std(zero["inlet_mean_u"])),
                            "inlet_hub_subblock_proxy": overlap_hub_proxy(temporal_rows[source], cursor, cursor + N * 50),
                            "common_burnin": {"first50_mean_power_mw": float(np.mean(burn[:50])),
                                              "last50_mean_power_mw": float(np.mean(burn[-50:])),
                                              "excluded_from_scoring": True},
                            "paired_evaluation_wall_seconds": result["elapsed_wall_seconds"]})
    sources = []
    for source in ("B", "C"):
        contexts = [window for window in windows if window["source_id"] == source]
        ez = sum(window["zero_energy_mwh"] for window in contexts)
        er = sum(window["rl_energy_mwh"] for window in contexts)
        rowz = np.sum([window["row_zero_energy_mwh"] for window in contexts], axis=0)
        rowr = np.sum([window["row_rl_energy_mwh"] for window in contexts], axis=0)
        lastz = sum(window["last1000"]["zero_energy_mwh"] for window in contexts)
        lastr = sum(window["last1000"]["rl_energy_mwh"] for window in contexts)
        block_gains = [gain(sum(window["time_blocks_500_controls"][i]["zero_energy_mwh"] for window in contexts),
                            sum(window["time_blocks_500_controls"][i]["rl_energy_mwh"] for window in contexts)) for i in range(4)]
        sources.append({"source_id": source, "zero_energy_mwh": ez, "rl_energy_mwh": er,
                        "gain_percent": gain(ez, er), "last1000_gain_percent": gain(lastz, lastr),
                        "last1000_zero_energy_mwh": lastz, "last1000_rl_energy_mwh": lastr,
                        "row_zero_energy_mwh": rowz.tolist(), "row_rl_energy_mwh": rowr.tolist(),
                        "row_gain_mwh": (rowr - rowz).tolist(), "row_gain_percent": (100 * (rowr / rowz - 1)).tolist(),
                        "aggregated_block_gains_percent": block_gains,
                        "first500_share_of_total_absolute_gain": sum(window["time_blocks_500_controls"][0]["absolute_gain_mwh"] for window in contexts) / (er - ez),
                        "last1000_share_of_total_absolute_gain": (lastr - lastz) / (er - ez),
                        "mean_command_travel_deg_per_turbine_hour": float(np.mean([window["yaw"]["mean_command_travel_deg_per_turbine_hour"] for window in contexts])),
                        "mean_command_reversals_per_turbine_hour": float(np.mean([window["yaw"]["mean_command_reversals_per_turbine_hour"] for window in contexts])),
                        "strict_angle_saturation_fraction": float(np.mean([window["yaw"]["strict_angle_saturation_fraction"] for window in contexts])),
                        "first_to_last_context": {"gain_change_percentage_points": contexts[-1]["gain_percent"] - contexts[0]["gain_percent"],
                                                  "hub_mean_proxy_change_m_s": contexts[-1]["inlet_hub_subblock_proxy"]["mean_hub_u_proxy_m_s"] - contexts[0]["inlet_hub_subblock_proxy"]["mean_hub_u_proxy_m_s"],
                                                  "exact_plane_mean_inlet_change_m_s": contexts[-1]["inlet_exact_plane_mean_u_m_s"] - contexts[0]["inlet_exact_plane_mean_u_m_s"]}})
    equal_gain = float(np.mean([source["gain_percent"] for source in sources]))
    checks["source_gains_and_primary_reproduced"] = abs(equal_gain - old["primary_equal_source_gain_percent"]) < 1e-10
    pooled_rows = np.sum([source["row_gain_mwh"] for source in sources], axis=0)
    means = np.array([window["yaw"]["mean_yaw_deg_per_turbine"] for window in windows])
    consensus = means.mean(axis=0)
    cosines = []
    pair_rms = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            cosines.append(float(np.dot(means[i], means[j]) / (np.linalg.norm(means[i]) * np.linalg.norm(means[j]))))
            pair_rms.append(float(np.sqrt(np.mean((means[i] - means[j]) ** 2))))
    scaling = json.loads((root / "efficiency/efficiency/SCALING_SUMMARY.json").read_text())
    allocation = json.loads(args.allocation_receipt.read_text())
    terminal_bundle = json.loads(args.efficiency_terminal_receipts.read_text())
    terminal_documents = json.loads(terminal_bundle["result"]["stdout"])["documents"]
    if any(terminal_documents[f"p{p}_COMPLETE.json"]["status"] != "complete" for p in (1, 2, 4)):
        raise ValueError("each topology needs a complete allocation receipt")
    efficiency_cost = {str(p): terminal_documents[f"p{p}_COMPLETE.json"]["allocated_gpu_hours"] for p in (1, 2, 4)}
    job_seconds_cost = sum(job["elapsed_seconds"] for job in allocation["jobs"].values()) / 3600
    costs = {
        "formal_training_interactions": ranks[0]["configuration"]["total_global_transitions_including_compile_warmup"],
        "formal_training_SAC_updates": int(ranks[0]["final_metrics"]["sac_updates"]),
        "formal_training_critical_rank_hours": max(rank["timing"]["total_process_wallclock_seconds"] for rank in ranks) / 3600,
        "formal_training_rank_timer_gpu_hours": sum(rank["timing"]["total_process_wallclock_seconds"] for rank in ranks) / 3600,
        "formal_normal_block_median_seconds": max(rank["timing"]["steady_state_iteration_seconds_median"] for rank in ranks),
        "formal_cost_boundary": "rank timers exclude container startup/independent precursor generation; supervisor allocation is a separate cost measure",
        "evaluation_allocated_gpu_hours_including_A400_pilot": json.loads((evaluation / "WORKFLOW_RESULT.json").read_text())["allocated_gpu_hours"],
        "BC_eight_paired_context_wall_sum_gpu_hours": sum(window["paired_evaluation_wall_seconds"] for window in windows) / 3600,
        "efficiency_allocated_gpu_hours": allocation["allocated_gpu_hours"],
        "efficiency_allocated_gpu_hours_by_topology": efficiency_cost,
        "allocation_receipt_sha256": sha256(args.allocation_receipt),
        "efficiency_terminal_receipts_sha256": sha256(args.efficiency_terminal_receipts),
        "status_job_elapsed_timer_gpu_hours": job_seconds_cost,
        "job_timer_minus_allocation_gpu_hours": job_seconds_cost - allocation["allocated_gpu_hours"],
        "timer_note": "Supervisor COMPLETE allocation receipts are authoritative; individual job elapsed_seconds are sampled at slightly different instants and are recorded separately, not substituted.",
    }
    if abs(sum(efficiency_cost.values()) - allocation["allocated_gpu_hours"]) > 1e-8:
        raise ValueError("efficiency allocation ledger does not reconcile")
    negative_blocks = [{"context_id": window["context_id"], "block": block["block"], "gain_percent": block["gain_percent"]}
                       for window in windows for block in window["time_blocks_500_controls"] if block["gain_percent"] < 0]
    summary = {
        "schema": SCHEMA, "status": "pass" if all(checks.values()) else "verification_failed", "checks": checks,
        "analysis_script_sha256": sha256(__file__), "energy_measure": "modeled gross ADM extracted energy MWh (thrust * filtered_disk_speed); no generator electrical conversion or actuator electrical energy modeled",
        "primary_equal_source_gain_percent": equal_gain,
        "equal_source_last1000_gain_percent": float(np.mean([source["last1000_gain_percent"] for source in sources])),
        "pooled_zero_energy_mwh": sum(source["zero_energy_mwh"] for source in sources),
        "pooled_rl_energy_mwh": sum(source["rl_energy_mwh"] for source in sources),
        "pooled_absolute_gain_mwh": float(sum(source["rl_energy_mwh"] - source["zero_energy_mwh"] for source in sources)),
        "pooled_row_integrated_absolute_gain_mwh": float(pooled_rows.sum()),
        "row_integrated_minus_farm_integrated_gain_mwh": float(pooled_rows.sum() - sum(source["rl_energy_mwh"] - source["zero_energy_mwh"] for source in sources)),
        "first_row_absolute_gain_mwh": float(pooled_rows[0]),
        "downstream_eight_rows_absolute_gain_mwh": float(pooled_rows[1:].sum()),
        "positive_windows": sum(window["gain_percent"] > 0 for window in windows),
        "positive_500_control_blocks": sum(block["gain_percent"] > 0 for window in windows for block in window["time_blocks_500_controls"]),
        "negative_500_control_blocks": negative_blocks,
        "window_gain_range_percent": [min(window["gain_percent"] for window in windows), max(window["gain_percent"] for window in windows)],
        "block_gain_range_percent": [float(np.min(gain_matrices)), float(np.max(gain_matrices))],
        "last1000_window_gain_range_percent": [min(window["last1000"]["gain_percent"] for window in windows), max(window["last1000"]["gain_percent"] for window in windows)],
        "layout_source": str(rank_paths[0]), "layout_source_sha256": sha256(rank_paths[0]),
        "layout": {"declared_layout_sha256": layout["declared_layout_sha256"], "streamwise_x_m": row_x.tolist(),
                   "row_turbine_indices": [group.tolist() for group in groups], "actual_positions_m": positions.tolist(),
                   "coordinate_convention": "LES x streamwise, y vertical, z crossflow; groups obtained from actual x coordinates"},
        "sources": sources, "windows": windows, "temporal_precursor_sources": temporal,
        "average_pattern": {"mean_yaw_deg_per_turbine": consensus.tolist(),
                            "row_mean_yaw_deg": [float(consensus[group].mean()) for group in groups],
                            "cross_context_mean_pattern_pair_cosine_range": [min(cosines), max(cosines)],
                            "cross_context_mean_pattern_pair_rms_difference_deg_range": [min(pair_rms), max(pair_rms)],
                            "cross_context_mean_pattern_sd_deg_mean_over_turbines": float(np.mean(np.std(means, axis=0))),
                            "mean_per_turbine_within_context_temporal_sd_deg": float(np.mean([window["yaw"]["mean_temporal_yaw_sd_deg"] for window in windows])),
                            "temporal_mean_squared_angle_fraction_range": [min(window["yaw"]["squared_angle_fraction_of_temporal_mean_pattern"] for window in windows),
                                                                         max(window["yaw"]["squared_angle_fraction_of_temporal_mean_pattern"] for window in windows)],
                            "conclusion": "Nonconstant commands coexist with a reproducible spatial mean pattern. Neither observation demonstrates benefit of feedback over a frozen mean pattern; a controlled intervention would be required."},
        "efficiency": scaling, "costs": costs, "input_trace_hashes": hashes,
        "statistical_contract": {"independent_physical_sources": 2, "training_optimization_seeds": 1,
                                 "windows_per_source": 4, "windows_are_iid": False,
                                 "confidence_interval_or_p_value": None,
                                 "block_segmentation": "prospectively fixed four contiguous 500-control segments; all retained",
                                 "tail_sensitivity": "last 1000 controls descriptive; all 2000 remain primary"},
        "flow_fields_or_probe_time_series_available": False,
        "limits": ["Fixed aligned layout and nominal wind condition; B/C are turbulent-source repetition, not wind-regime generalization.",
                   "Exact plane-mean inlet speed from trajectories and overlap-weighted hub-speed proxy from source records are different quantities.",
                   "Time-segment positivity rules out gain confined to startup, but does not establish statistically stationary inflow.",
                   "Row-energy redistribution is consistent with wake coordination; flow-field deflection is not directly observed.",
                   "Command-equivalent rates/reversals are not physical yaw-drive wear or net-energy measurements.",
                   "Matched-work scaling uses fixed collector actions and a synthetic episode clock; not full-training time-to-gain."]}
    if not all(checks.values()):
        raise ValueError("mechanism verification failed: " + str(checks))
    write_new(args.summary, summary)
    plots(summary, np.concatenate(rates_for_plot), args.figures)
    write_new(args.report, report(summary, args.summary, args.figures))
    print(json.dumps({"status": "pass", "primary_gain_percent": equal_gain,
                      "tail_gain_percent": summary["equal_source_last1000_gain_percent"],
                      "positive_blocks": summary["positive_500_control_blocks"],
                      "summary": str(args.summary), "report": str(args.report)}, indent=2))


def save_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = directory / (name + "." + extension)
        if path.exists():
            raise FileExistsError("figure output must be new: " + str(path))
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plots(summary, absolute_rates, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    sources = summary["sources"]
    windows = summary["windows"]
    rows = np.arange(1, 10)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for source in sources:
        axes[0].plot(rows, source["row_gain_percent"], marker="o", color=COLORS[source["source_id"]], label="Source " + source["source_id"])
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set(xlabel="Streamwise turbine row", ylabel="Modeled gross energy change (%)", title="(a) Source-level row redistribution")
    axes[0].legend(frameon=False)
    change = np.sum([source["row_gain_mwh"] for source in sources], axis=0)
    axes[1].bar(rows, change, color=["#b54b4b" if value < 0 else "#508f79" for value in change])
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(xlabel="Streamwise turbine row", ylabel="Modeled gross energy difference (MWh)", title="(b) All scored windows combined")
    axes[1].text(0.03, 0.97, "Upstream: %.1f MWh\nDownstream: +%.1f MWh\nNet: +%.1f MWh" %
                 (change[0], change[1:].sum(), change.sum()), transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    figure.tight_layout()
    save_figure(figure, directory, "row_energy")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values = np.array([[block["gain_percent"] for block in window["time_blocks_500_controls"]] for window in windows])
    colored = axis.imshow(values, vmin=0, vmax=22, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(4), ["0–500", "500–1000", "1000–1500", "1500–2000"])
    axis.set_yticks(range(8), [window["source_id"] + str(window["window_number"]) for window in windows])
    axis.set(xlabel="Scored control interval (10 s per control)", ylabel="Context within physical source",
             title="All preselected 500-control segments; no segments removed")
    axis.axhline(3.5, color="white", linewidth=2)
    for i in range(8):
        for j in range(4):
            axis.text(j, i, "%.2f" % values[i, j], ha="center", va="center", color="white" if values[i, j] > 13 else "black", fontsize=10)
    figure.colorbar(colored, ax=axis, label="Modeled gross energy gain (%)")
    figure.text(0.08, 0.01, "Two independent physical sources; four correlated contexts per source. No iid-window confidence interval.", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, directory, "time_block_gain")

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    for window in windows:
        axes[0, 0].plot(rows, window["row_mean_yaw_deg"], color=COLORS[window["source_id"]], alpha=0.35, linewidth=1)
    axes[0, 0].plot(rows, summary["average_pattern"]["row_mean_yaw_deg"], "ko-", markersize=4, label="Eight-context mean pattern")
    axes[0, 0].set(xlabel="Streamwise row", ylabel="Time-mean commanded yaw (deg)", title="(a) Reproducible spatial mean pattern")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].hist(absolute_rates, bins=np.linspace(0, 1, 31), weights=np.full(absolute_rates.size, 100 / absolute_rates.size),
                    color="#508f79", edgecolor="white", linewidth=0.3)
    axes[0, 1].set(xlabel="Absolute command-equivalent yaw rate (deg/s)", ylabel="Turbine-control samples (%)", title="(b) Dynamic command motion")
    labels = [window["source_id"] + str(window["window_number"]) for window in windows]
    x = np.arange(8)
    travel = np.array([window["yaw"]["mean_command_travel_deg_per_turbine_hour"] for window in windows])
    axes[1, 0].scatter(x, travel, color=[COLORS[window["source_id"]] for window in windows], s=45)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(xlabel="Context", ylabel="Mean command travel (deg/turbine/h)", title="(c) Command travel; no drive model")
    axes[1, 0].set_ylim(0, max(travel) * 1.12)
    saturated = 100 * np.array([window["yaw"]["strict_angle_saturation_fraction"] for window in windows])
    near = 100 * np.array([window["yaw"]["near_angle_limit_fraction_abs_at_least_38_deg"] for window in windows])
    axes[1, 1].bar(x - 0.17, saturated, 0.34, color="#b54b4b", label="At 40 deg (4 ULP tolerance)")
    axes[1, 1].bar(x + 0.17, near, 0.34, color="#d59c61", label="Absolute yaw ≥38 deg")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set(xlabel="Context", ylabel="Turbine-control samples (%)", title="(d) Limit occupancy")
    axes[1, 1].set_ylim(0, max(near) * 1.35)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure, directory, "yaw_motion")

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    topology = summary["efficiency"]["topologies"]
    collect = np.array([topology[str(p)]["normal_collector_seconds_median"] for p in (1, 2, 4)])
    learner = np.array([topology[str(p)]["normal_learner_including_communication_seconds_median"] for p in (1, 2, 4)])
    x = np.arange(3)
    axes[0, 0].bar(x - 0.18, collect, 0.36, label="LES collection", color="#2b6f9c")
    axes[0, 0].bar(x + 0.18, learner, 0.36, label="Learner + communication", color="#c36b2a")
    axes[0, 0].set_xticks(x, ["1", "2", "4"])
    axes[0, 0].set(xlabel="GPUs", ylabel="Median regular-block time (s)", title="(a) Matched 256-transition blocks")
    axes[0, 0].legend(frameon=False, fontsize=8)
    total = np.array([topology[str(p)]["total_including_cold_reset_learner_sync_and_checkpoint_seconds"] for p in (1, 2, 4)])
    reset = np.array([topology[str(p)]["reset_block_seconds"] for p in (1, 2, 4)])
    axes[0, 1].bar(x, reset / 60, label="Whole block containing reset150", color="#a08aad")
    axes[0, 1].bar(x, (total - reset) / 60, bottom=reset / 60, label="Other measured time", color="#81a6b6")
    axes[0, 1].set_xticks(x, ["1", "2", "4"])
    axes[0, 1].set(xlabel="GPUs", ylabel="Bounded profile wall time (min)", title="(b) Profile-specific reset exposure")
    axes[0, 1].legend(frameon=False, fontsize=8)
    for i, value in enumerate(total / 60):
        axes[0, 1].text(i, value + 2, "%.1f" % value, ha="center", fontsize=9)
    axes[0, 1].set_ylim(0, max(total / 60) * 1.15)
    gpu_cost = np.array([summary["costs"]["efficiency_allocated_gpu_hours_by_topology"][str(p)] for p in (1, 2, 4)])
    axes[1, 0].bar(x, gpu_cost, color="#508f79")
    axes[1, 0].set_xticks(x, ["1", "2", "4"])
    axes[1, 0].set(xlabel="GPUs", ylabel="Allocated GPU hours", title="(c) Resource cost increases with scaling")
    for i, value in enumerate(gpu_cost):
        axes[1, 0].text(i, value + 0.03, "%.3f" % value, ha="center", fontsize=9)
    axes[1, 0].set_ylim(0, max(gpu_cost) * 1.15)
    axes[1, 1].plot([1, 2, 4], collect[0] / collect, "o-", color="#2b6f9c", label="Collection")
    axes[1, 1].plot([1, 2, 4], learner[0] / learner, "s-", color="#c36b2a", label="Learner + communication")
    axes[1, 1].plot([1, 2, 4], [1, 2, 4], "k--", alpha=0.4, label="Linear scaling")
    axes[1, 1].set_xticks([1, 2, 4])
    axes[1, 1].set(xlabel="GPUs", ylabel="Speedup relative to one GPU", title="(d) Different component scaling")
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.text(0.07, 0.01, "Fixed collector commands and synthetic clock 404. Complete-profile speedup is not full-training time-to-gain.", fontsize=8)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(figure, directory, "scaling_breakdown")


def report(summary, summary_path, figures):
    sources = summary["sources"]
    windows = summary["windows"]
    motion = [window["yaw"] for window in windows]
    pattern = summary["average_pattern"]
    lines = ["# 81风机现有数据机制分析", "", "日期：2026-10-13。CPU复算完成，没有新增LES或训练。", "",
             "## 复算与证据身份", "",
             "16条评分轨迹、8条共同预热轨迹均按冻结RESULT的SHA复核。实际风机坐标取自正式rank0的layout_contract，并核对八个RESULT的布局SHA；按实际流向x坐标分排，没有按数组形状猜测风机顺序。功率单位为MW，每控制10 s，积分得到模型毛ADM气动提取能量MWh。此功率等于推力乘以滤波盘面速度，没有模拟发电机转换效率；原始RESULT沿用generation字段名，不将其解释为电能或净发电量。",
             "", "| 来源 | 零偏航MWh | R1 MWh | 全2000步收益 | 后1000步收益 |", "|---|---:|---:|---:|---:|"]
    for source in sources:
        lines.append("| %s | %.6f | %.6f | %.4f%% | %.4f%% |" % (source["source_id"], source["zero_energy_mwh"], source["rl_energy_mwh"], source["gain_percent"], source["last1000_gain_percent"]))
    lines.extend(["", "两来源等权主收益为**%.6f%%**，复现原评价汇总；后1000步的两来源等权收益为**%.6f%%**，仅作描述性敏感性。两来源才是独立物理单位，四窗/来源及其时间段均相关。仅一个优化种子；没有构造八窗口iid置信区间或人口泛化p值。" % (summary["primary_equal_source_gain_percent"], summary["equal_source_last1000_gain_percent"]),
                  "", "## 时间分解与启动过渡", "",
                  "预先固定的32个500控制子段全部正收益，范围%.4f%%至%.4f%%；八个后1000步窗口也全部为正，范围%.4f%%至%.4f%%。这直接排除‘收益只在评分初期出现’的解释，并不证明严格平稳。共同150控制预热仍排除在所有能量主指标之外。" % (*summary["block_gain_range_percent"], *summary["last1000_window_gain_range_percent"]),
                  "", "| 来源 | 第一500步占总新增能量 | 后1000步占总新增能量 | 第一窗至第四窗收益变化/百分点 |", "|---|---:|---:|---:|"])
    for source in sources:
        lines.append("| %s | %.2f%% | %.2f%% | %+.4f |" % (source["source_id"], 100 * source["first500_share_of_total_absolute_gain"], 100 * source["last1000_share_of_total_absolute_gain"], source["first_to_last_context"]["gain_change_percentage_points"]))
    lines.extend(["", "## 逐排能量机制", "",
                  "第一排合计损失**%.6f MWh**，下游八排合计增加**%.6f MWh**，总净增加**%.6f MWh**。逐排先积分和场站功率先求和的FP32归约顺序不同，两个净能量口径相差%.9g MWh，保留该差异，主指标继续采用场站功率积分。两个来源每排的方向一致：第一排损失、其余八排增加。它与协同偏航的上游损失—下游恢复机制一致。没有完整流场或逐点探针时间序列，不能把能量分配图写成直接观察到的尾流偏转或湍流混合。" % (summary["first_row_absolute_gain_mwh"], summary["downstream_eight_rows_absolute_gain_mwh"], summary["pooled_absolute_gain_mwh"], summary["row_integrated_minus_farm_integrated_gain_mwh"]),
                  "", "| 排 | B来源逐排变化 | C来源逐排变化 |", "|---|---:|---:|"])
    for i in range(9):
        lines.append("| %d | %+.3f%% | %+.3f%% |" % (i + 1, sources[0]["row_gain_percent"][i], sources[1]["row_gain_percent"][i]))
    lines.extend(["", "## 动作代价、限幅与固定模式替代解释", "",
                  "所有偏航都是控制边界记录的指令角。差分/10 s给出command-equivalent rate，不能当作实际执行器瞬时角速度；反转数也不是疲劳模型。限幅判定容差为40°处FP32的四个ULP，即%.9g°，仅排除数值量化误差。" % ANGLE_TOL,
                  "", "| 来源 | 平均指令行程/°每台每小时 | 平均指令反转/次每台每小时 | 严格40°限幅占比 |", "|---|---:|---:|---:|"])
    for source in sources:
        lines.append("| %s | %.3f | %.3f | %.4f%% |" % (source["source_id"], source["mean_command_travel_deg_per_turbine_hour"], source["mean_command_reversals_per_turbine_hour"], 100 * source["strict_angle_saturation_fraction"]))
    lines.extend(["", "八窗逐机时间标准差的平均为%.4f°，窗口范围%.4f°至%.4f°，指令确实随时间变化。平均排偏航依次为%s°。不同窗口平均模式的余弦相似度为%.6f至%.6f，逐机RMS差异为%.4f°至%.4f°；共同空间模式具有很强重复性。" % (pattern["mean_per_turbine_within_context_temporal_sd_deg"], min(item["mean_temporal_yaw_sd_deg"] for item in motion), max(item["mean_temporal_yaw_sd_deg"] for item in motion), ", ".join("%.2f" % x for x in pattern["row_mean_yaw_deg"]), *pattern["cross_context_mean_pattern_pair_cosine_range"], *pattern["cross_context_mean_pattern_pair_rms_difference_deg_range"]),
                  "", "时间平均模式占偏航角均方的%.2f%%至%.2f%%。这是角度分解，不是收益贡献分解。现有证据允许‘稳定空间偏航模式叠加瞬态调整’；不能排除平均模式本身已经产生主要收益，也不能从动作非恒定推导动态反馈具有因果增益。若本文保留‘动态反馈优于固定控制’贡献，应另外比较冻结时间平均偏航或严格动作回放；当前基本主线不必自动增加此实验。" % tuple(100 * x for x in pattern["temporal_mean_squared_angle_fraction_range"]),
                  "", "高行程和约175次每小时的指令反转是需要说明的代价。40°最大值实际只对应约%.3f%%至%.3f%%的机组—控制样本，不能把出现40°推断为长期饱和；也不能据此宣称现场可部署、低磨损或净收益。" % (100 * min(item["strict_angle_saturation_fraction"] for item in motion), 100 * max(item["strict_angle_saturation_fraction"] for item in motion)),
                  "", "## 入口漂移及两种均速口径", "",
                  "trace的inlet_mean_u是完整入口平面上流向速度的平均，约8.1 m/s；temporal文件的mean_hub_u是轮毂高度的截面均速，约7.35 m/s。不能混用两者。A/B/C的空间TI代理约0.107，不是固定测点的时间湍流强度。源块和子块沿同一湍流轨迹相关。", "",
                  "| 来源 | 入口轮毂高度均速/m s⁻¹ | 20k块均速范围/m s⁻¹ | 首尾块变化/m s⁻¹ | 平均空间TI代理 |", "|---|---:|---:|---:|---:|"])
    for name, source in summary["temporal_precursor_sources"].items():
        lines.append("| %s | %.6f | %.6f～%.6f | %+.6f | %.6f |" % (name, source["hub_mean_u_m_s"], *source["hub_block_mean_range_m_s"], source["hub_first_to_last_change_m_s"], source["mean_spatial_ti_proxy"]))
    lines.extend(["", "评分片段轮毂均速采用5000-plane子块的重叠加权代理；跨子块部分按子块均值近似，不冒充评分窗精确轮毂均速。B/C第一窗至第四窗完整平面均速都下降，但B收益上升、C收益下降，入口均速的单一漂移不能充分解释全部收益差异。没有对相关窗口做因果回归或站稳验收。",
                  "", "## 计算效率与成本口径", "",
                  "1/2/4卡完整匹配剖析分别为199.332/103.195/57.047 min，总加速1/1.932/3.494；常规256交互块中位417.284/222.983/125.876 s。采样四卡加速约3.985，学习与通信约1.568；学习与通信成为主要扩展瓶颈。短剖析的重置块占54%～60%，正式500控制协议不同。独立入口生成、成熟Replay与完整训练time-to-gain均不由该比值确认。",
                  "", "真实分配成本1/2/4卡分别为%.6f/%.6f/%.6f GPUh，四卡较单卡增加%.4f%%。正式训练完成%d真实LES交互和%d SAC更新，临界rank时间%.6f h，四rank内部计时合计%.6f GPUh；内部计时与监督器容器分配GPUh不混用。B/C八对上下文墙时合计%.6f GPUh；评价监督器连A400试段合计%.6f GPUh。" % (*[summary["costs"]["efficiency_allocated_gpu_hours_by_topology"][str(p)] for p in (1, 2, 4)], (summary["costs"]["efficiency_allocated_gpu_hours_by_topology"]["4"] / summary["costs"]["efficiency_allocated_gpu_hours_by_topology"]["1"] - 1) * 100, summary["costs"]["formal_training_interactions"], summary["costs"]["formal_training_SAC_updates"], summary["costs"]["formal_training_critical_rank_hours"], summary["costs"]["formal_training_rank_timer_gpu_hours"], summary["costs"]["BC_eight_paired_context_wall_sum_gpu_hours"], summary["costs"]["evaluation_allocated_gpu_hours_including_A400_pilot"]),
                  "", "## 对论文的直接影响", "",
                  "已支持百万次81台直接LES训练、两个独立入口实现下的有益策略、持久正收益与能量在上下游的再分配，以及限定匹配剖析的计算扩展性。尚未支持瞬态反馈相对固定模式的额外收益、训练种子稳定性、数值离散无关性、现场驱动负担或净收益。新稿应以可扩展LES–RL执行框架与有益策略为主线；不能把空间模式高度重复或大量反转本身包装为新的反馈机制。",
                  "", "## 产物与复现", "", f"数据摘要：`{summary_path.resolve()}`。脚本不改变输入或选择策略。输出已存在时拒绝覆盖。", "",
                  "运行：", "", "```bash", "/home/shao/.local/share/statistical-analysis/.venv/bin/python codes/paper1_n81_mechanism_analysis_20261013.py", "```", "",
                  "图件（PDF为论文版本，PNG为检查副本）："])
    for name in ("row_energy", "time_block_gain", "yaw_motion", "scaling_breakdown"):
        lines.append(f"- [{name}.pdf]({(figures / (name + '.pdf')).resolve()})")
    return "\n".join(lines) + "\n"


def main():
    repo = Path(__file__).resolve().parents[1]
    audit = repo / "audit/paper1_n81_evidence_closure_20261013"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=audit / "existing")
    parser.add_argument("--summary", type=Path, default=audit / "MECHANISM_SUMMARY.json")
    parser.add_argument("--report", type=Path, default=audit / "MECHANISM_REPORT.zh-CN.md")
    parser.add_argument("--figures", type=Path, default=repo / "paper1_les_system/n81_20261013/figures")
    parser.add_argument("--allocation-receipt", type=Path,
                        default=repo / "audit/paper1_n81_efficiency_corrected_20261012/verified_remote_results_20261013/STATUS.json")
    parser.add_argument("--efficiency-terminal-receipts", type=Path,
                        default=repo / "audit/paper1_n81_efficiency_corrected_20261012/PROGRESS_CHECK_20261013_001010.json")
    args = parser.parse_args()
    if args.summary.exists() or args.report.exists():
        parser.error("summary/report must be new; existing evidence is not overwritten")
    analyze(args)


if __name__ == "__main__":
    main()
