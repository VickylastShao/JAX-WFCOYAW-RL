#!/usr/bin/env python3
"""Independently integrate saved N81 traces; no NumPy/JAX, LES or Actor execution."""
from __future__ import annotations

import argparse
import array
import ast
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile

sys.dont_write_bytecode = True
from verify_manifest import package_path, sha256, verify


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(root: Path, name: str) -> dict:
    return json.loads(package_path(root, name).read_text(encoding="utf-8"))


def npz_array(path: Path, key: str, expected_shape: tuple) -> array.array:
    """Read numeric little-endian float32 C-order NPY without importing NumPy."""
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate NPZ member")
        with archive.open(key + ".npy") as stream:
            require(stream.read(6) == b"\x93NUMPY", "invalid NPY magic")
            version = tuple(stream.read(2))
            require(version in ((1, 0), (2, 0), (3, 0)), "unsupported NPY version")
            width = 2 if version == (1, 0) else 4
            header_size = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
            require(header_size <= 65536, "oversized NPY header")
            header = ast.literal_eval(stream.read(header_size).decode("utf-8" if version == (3, 0) else "latin1"))
            require(header.get("descr") == "<f4" and header.get("fortran_order") is False, "unexpected NPY dtype/order")
            require(header.get("shape") == expected_shape, f"unexpected {path.name}:{key} shape")
            data = array.array("f")
            require(data.itemsize == 4, "platform float array must be 4 bytes")
            data.frombytes(stream.read())
            if sys.byteorder != "little":
                data.byteswap()
            require(len(data) == math.prod(expected_shape), "NPY payload size differs")
            require(all(math.isfinite(x) for x in data), "nonfinite NPZ value")
            return data


def gain(zero: float, controlled: float) -> float:
    require(zero > 0, "reference energy must be positive")
    return 100.0 * (controlled / zero - 1.0)


def energy(power, dt: float) -> float:
    return math.fsum(power) * dt / 3600.0


def row_energy(turbine_power, groups: list, n: int, dt: float) -> list:
    return [math.fsum(turbine_power[t * 81 + i] for t in range(n) for i in group) * dt / 3600.0 for group in groups]


def yaw_metrics(yaw, action, n: int, dt: float, tolerance: float) -> dict:
    travel, reversals, means, temporal_variances = [], [], [], []
    reconstruction_error = 0.0
    for i in range(81):
        angles = [yaw[t * 81 + i] for t in range(n)]
        previous = [0.0] + angles[:-1]
        increments = [a - b for a, b in zip(angles, previous)]
        travel.append(math.fsum(abs(x) for x in increments))
        signs = [1 if x > 0 else -1 for x in increments if abs(x) > tolerance]
        reversals.append(sum(a != b for a, b in zip(signs, signs[1:])))
        mean = statistics.fmean(angles)
        means.append(mean)
        temporal_variances.append(math.fsum((x - mean) ** 2 for x in angles) / n)
        for t, (angle, prev) in enumerate(zip(angles, previous)):
            predicted = max(-40.0, min(40.0, prev + dt * action[t * 81 + i]))
            reconstruction_error = max(reconstruction_error, abs(predicted - angle))
    require(reconstruction_error <= tolerance, "yaw/action command reconstruction differs")
    hours = n * dt / 3600.0
    static_ms = statistics.fmean(x * x for x in means)
    temporal_variance = statistics.fmean(temporal_variances)
    return {"mean_command_travel_deg_per_turbine_hour": statistics.fmean(travel) / hours,
            "mean_command_reversals_per_turbine_hour": statistics.fmean(reversals) / hours,
            "strict_angle_saturation_fraction": sum(abs(x) >= 40.0 - tolerance for x in yaw) / len(yaw),
            "near_angle_limit_fraction_abs_at_least_38_deg": sum(abs(x) >= 38.0 for x in yaw) / len(yaw),
            "mean_yaw_deg_per_turbine": means,
            "squared_angle_fraction_of_temporal_mean_pattern": static_ms / (static_ms + temporal_variance) if static_ms + temporal_variance else 0.0,
            "command_reconstruction_max_abs_error_deg": reconstruction_error,
            "physical_drive_energy_or_fatigue_measured": False}


def load_trace(root: Path, folder: str, controller: str, n: int, dt: float, groups: list, tolerance: float) -> dict:
    path = package_path(root, folder + "/" + controller + ".npz")
    result = read_json(root, folder + "/RESULT.json")
    require(result["status"] == "complete", "source result is incomplete")
    record = result["controllers"][controller]
    require(sha256(path) == record["trace_sha256"], "original trace receipt SHA differs")
    power = npz_array(path, "farm_power_mw", (n, 1))
    turbine = npz_array(path, "turbine_power_mw", (n, 1, 81))
    yaw = npz_array(path, "yaw_deg", (n, 1, 81))
    # Zero-reference files deliberately omit normalized_action; the controller
    # contract specifies zero command. Never substitute for a learned trace.
    action = array.array("f", [0.0]) * (n * 81) if controller == "zero" else npz_array(path, "normalized_action", (n, 1, 81))
    for t in range(n):
        require(math.isclose(power[t], math.fsum(turbine[t * 81:(t + 1) * 81]), rel_tol=1e-6, abs_tol=2e-5), "farm/turbine sum differs")
    value = energy(power, dt)
    require(math.isclose(value, result["scores"][controller]["generation_energy_mwh"], rel_tol=1e-12, abs_tol=1e-9), "recorded energy differs")
    if controller == "zero":
        require(all(x == 0 for x in yaw) and all(x == 0 for x in action), "zero reference has yaw/action")
    return {"power": power, "rows_MWh": row_energy(turbine, groups, n, dt), "energy_MWh": value,
            "yaw": yaw_metrics(yaw, action, n, dt, tolerance),
            "inlet_mean_u": npz_array(path, "inlet_mean_u", (n, 1)),
            "inlet_delta_rms": npz_array(path, "inlet_delta_rms", (n, 1))}


def paired(zero: dict, controlled: dict) -> None:
    require(zero["inlet_mean_u"] == controlled["inlet_mean_u"] and zero["inlet_delta_rms"] == controlled["inlet_delta_rms"], "paired inlet traces differ")


def compare_subset(actual, expected, path="metrics", atol=1e-8, rtol=1e-12) -> int:
    count = 0
    if isinstance(expected, dict):
        require(isinstance(actual, dict), path + ": expected mapping")
        for key, value in expected.items():
            require(key in actual, path + ": missing key " + key)
            count += compare_subset(actual[key], value, path + "." + key, atol, rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ": list length differs")
        for i, value in enumerate(expected):
            count += compare_subset(actual[i], value, path + f"[{i}]", atol, rtol)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), f"{path}: value {actual!r} differs from {expected!r}")
        count += 1
    else:
        require(actual == expected, path + ": value differs")
        count += 1
    return count


def analyse(root: Path) -> dict:
    manifest = verify(root, root / "DATA_MANIFEST.json")
    require(manifest["status"] == "pass", "data manifest failed: " + json.dumps(manifest["failures"]))
    cfg = read_json(root, "configs/analysis_expected.json")
    require(cfg["schema"] == "n81_l1_analysis_contract_v1", "analysis schema differs")
    dt, tol = cfg["control_dt_seconds"], cfg["yaw_fp32_tolerance_deg"]
    require(dt == 10.0 and tol == 4 * 2 ** -18, "physical units/tolerance contract differs")
    groups = cfg["streamwise_row_turbine_indices"]
    require(len(groups) == 9 and all(len(g) == 9 for g in groups) and sorted(i for g in groups for i in g) == list(range(81)), "row roster differs")
    out = {"BC": {}, "P2": [], "P4": {}}
    block_gains, late_window_gains, all_rows = [], [], []
    for source in ("B", "C"):
        windows = []
        for number in range(1, 5):
            folder = f"data/BC/{source}_w{number}"
            zero = load_trace(root, folder, "zero", 2000, dt, groups, tol)
            rl = load_trace(root, folder, "rl_r1", 2000, dt, groups, tol)
            paired(zero, rl)
            blocks = [gain(energy(zero["power"][k:k+500], dt), energy(rl["power"][k:k+500], dt)) for k in range(0, 2000, 500)]
            row_delta = [r-z for z,r in zip(zero["rows_MWh"], rl["rows_MWh"])]
            late_z, late_r = energy(zero["power"][1000:], dt), energy(rl["power"][1000:], dt)
            window = {"window": number, "zero_MWh": zero["energy_MWh"], "policy_MWh": rl["energy_MWh"],
                      "gain_percent": gain(zero["energy_MWh"], rl["energy_MWh"]), "blocks_percent": blocks,
                      "last_1000_zero_MWh": late_z, "last_1000_policy_MWh": late_r,
                      "last_1000_gain_percent": gain(late_z, late_r), "row_delta_MWh": row_delta, "yaw": rl["yaw"]}
            windows.append(window)
            block_gains.extend(blocks)
            late_window_gains.append(window["last_1000_gain_percent"])
            all_rows.append(row_delta)
        z = math.fsum(w["zero_MWh"] for w in windows)
        r = math.fsum(w["policy_MWh"] for w in windows)
        out["BC"][source] = {"zero_MWh": z, "policy_MWh": r, "gain_percent": gain(z, r),
                             "last_1000_source_gain_percent": gain(math.fsum(w["last_1000_zero_MWh"] for w in windows), math.fsum(w["last_1000_policy_MWh"] for w in windows)),
                             "mean_command_travel_deg_per_turbine_hour": statistics.fmean(w["yaw"]["mean_command_travel_deg_per_turbine_hour"] for w in windows),
                             "mean_command_reversals_per_turbine_hour": statistics.fmean(w["yaw"]["mean_command_reversals_per_turbine_hour"] for w in windows),
                             "strict_angle_saturation_fraction": statistics.fmean(w["yaw"]["strict_angle_saturation_fraction"] for w in windows),
                             "windows": windows}
    out["equal_source_gain_percent"] = statistics.fmean(out["BC"][s]["gain_percent"] for s in ("B", "C"))
    out["block_gain_range_percent"] = [min(block_gains), max(block_gains)]
    out["positive_500_control_blocks"] = sum(g > 0 for g in block_gains)
    out["positive_windows"] = sum(w["gain_percent"] > 0 for s in ("B", "C") for w in out["BC"][s]["windows"])
    out["last_1000_equal_source_gain_percent"] = statistics.fmean(out["BC"][s]["last_1000_source_gain_percent"] for s in ("B", "C"))
    out["last_1000_window_mean_percent"] = statistics.fmean(late_window_gains)
    out["row_energy_difference_MWh"] = [math.fsum(rows[i] for rows in all_rows) for i in range(9)]
    farm_delta = math.fsum(out["BC"][s]["policy_MWh"] - out["BC"][s]["zero_MWh"] for s in ("B", "C"))
    out["pooled_farm_energy_difference_MWh"] = farm_delta
    out["row_sum_minus_farm_difference_MWh"] = math.fsum(out["row_energy_difference_MWh"]) - farm_delta
    zero = load_trace(root, "data/P2", "zero", 2000, dt, groups, tol)
    for step in (0, 131072, 524288, 1000448):
        rl = zero if step == 0 else load_trace(root, "data/P2", "rl_step"+str(step), 2000, dt, groups, tol)
        paired(zero, rl)
        out["P2"].append({"interactions": step, "energy_MWh": rl["energy_MWh"], "gain_percent": gain(zero["energy_MWh"], rl["energy_MWh"]),
                          "command_travel_deg_per_turbine_hour": rl["yaw"]["mean_command_travel_deg_per_turbine_hour"]})
    for label, les_dt in (("dt02", .2), ("dt01", .1)):
        folder = "data/P4/" + label
        contract = read_json(root, folder + "/RESULT.json")["contract"]
        require(contract["les_dt_seconds"] == les_dt and contract["control_dt_seconds"] == dt and contract["scored_controls"] == 400, "P4 dt/control contract differs")
        zero = load_trace(root, folder, "zero", 400, dt, groups, tol)
        rl = load_trace(root, folder, "rl_final", 400, dt, groups, tol)
        paired(zero, rl)
        out["P4"][label] = {"zero_MWh": zero["energy_MWh"], "policy_MWh": rl["energy_MWh"], "gain_percent": gain(zero["energy_MWh"], rl["energy_MWh"]),
                               "row_gains_percent": [gain(z,r) for z,r in zip(zero["rows_MWh"], rl["rows_MWh"])]}
    out["P4"]["gain_difference_pp"] = out["P4"]["dt01"]["gain_percent"] - out["P4"]["dt02"]["gain_percent"]
    out["P4"]["row_gain_difference_pp"] = [f-c for c,f in zip(out["P4"]["dt02"]["row_gains_percent"], out["P4"]["dt01"]["row_gains_percent"])]
    scaling = read_json(root, "data/timing/SCALING_SUMMARY.json")["topologies"]
    times = {g: v["total_including_cold_reset_learner_sync_and_checkpoint_seconds"] for g,v in scaling.items()}
    out["scaling_from_saved_timers"] = {}
    ordinary1 = statistics.median(scaling["1"]["normal_block_critical_rank_seconds"])
    for g, data in scaling.items():
        blocks = data["normal_block_critical_rank_seconds"]
        require(data["actual_global_interactions"] == 3072 and len(blocks) == 10, "fixed-work profile contract differs")
        out["scaling_from_saved_timers"][g] = {"seconds": times[g], "speedup": times["1"]/times[g], "ordinary_sample_count": len(blocks),
                 "ordinary_median_seconds": statistics.median(blocks), "ordinary_range_seconds": [min(blocks), max(blocks)],
                 "ordinary_speedup": ordinary1/statistics.median(blocks),
                 "critical_path_seconds_times_GPU_count_over_3600": float(g)*times[g]/3600.0,
                 "saved_sum_rank_timer_GPUh": data["rank_accounted_gpu_hours"]}
    out["P3_saved_report"] = {"full_state_comparison_reexecuted": False}
    p3summary = read_json(root, "data/P3/SUMMARY.json")
    for mode in ("scan8", "single_jit8"):
        data = read_json(root, f"data/P3/{mode}.component.json")
        collector = data["normal_collector_seconds"]
        require(len(collector) == len(data["normal_block_seconds"]) == 2, "P3 block count differs")
        median = statistics.median(collector)
        require(math.isclose(median, p3summary["collector_median_seconds"][mode], abs_tol=1e-10), "P3 saved median differs")
        out["P3_saved_report"][mode] = {"collector_seconds": collector, "collector_median_seconds": median,
                                      "block_seconds": data["normal_block_seconds"], "learner_seconds": data["normal_learner_seconds"]}
    eq = p3summary["state_equivalence"]
    leaves = eq["leaves"]
    out["P3_saved_report"]["leaf_count_in_saved_report"] = len(leaves)
    out["P3_saved_report"]["saved_leaf_exact_finite_pass_count"] = sum(l["exact"] and l["finite"] and l["pass"] for l in leaves)
    out["P3_saved_report"]["saved_state_equivalence_pass"] = eq["pass"]
    out["P3_saved_report"]["collector_speedup_single_over_scan"] = out["P3_saved_report"]["single_jit8"]["collector_median_seconds"]/out["P3_saved_report"]["scan8"]["collector_median_seconds"]
    expected = read_json(root, "RESULTS_EXPECTED.json")
    checked = compare_subset(out, expected["metrics"])
    return {"status": "pass", "scope": "L1 saved-output CPU arithmetic; no LES, Actor inference, or full-state P3 replay", "gpu_used": False,
            "manifest_verification": manifest, "expected_values_checked": checked,
            "numerical_comparison": {"absolute_tolerance": 1e-8, "relative_tolerance": 1e-12, "purpose": "arithmetic replay tolerance, not scientific acceptance threshold"},
            "metrics": out}


def self_test(root: Path) -> dict:
    """Run a fresh-copy replay and reject meaningful tampering in temp files."""
    import shutil
    import subprocess
    import tempfile

    original = verify(root, root / "DATA_MANIFEST.json")
    require(original["status"] == "pass", "original manifest must pass before self-test")
    files = list(read_json(root, "DATA_MANIFEST.json")["files"])
    files += ["DATA_MANIFEST.json", "RESULTS_EXPECTED.json", "scripts/reanalyse.py", "scripts/verify_manifest.py"]
    source_sha = {name: sha256(package_path(root, name)) for name in files}
    tests = []
    with tempfile.TemporaryDirectory(prefix="n81-l1-portable-") as name:
        temporary = Path(name)
        isolated = temporary / "independent-copy"
        cwd = temporary / "unrelated-working-directory"
        cwd.mkdir()
        for relative in files:
            target = isolated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package_path(root, relative), target)

        def run(script: str, should_pass: bool, label: str):
            command = [sys.executable, str(isolated / "scripts" / script), "--root", str(isolated)]
            completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
            require((completed.returncode == 0) == should_pass, label + ": unexpected exit status")
            stream = completed.stdout if completed.returncode == 0 or completed.stdout.strip() else completed.stderr
            report = json.loads(stream)
            require(report["status"] == ("pass" if should_pass else "fail"), label + ": unexpected result status")
            tests.append({"test": label, "expected_exit_zero": should_pass, "actual_exit_code": completed.returncode,
                          "status": "pass", "failure_reason": report.get("error") or report.get("failures"),
                          "expected_values_checked": report.get("expected_values_checked")})

        run("verify_manifest.py", True, "fresh_copy_manifest_from_unrelated_cwd")
        run("reanalyse.py", True, "fresh_copy_full_reanalysis_from_unrelated_cwd")
        data = isolated / "data/BC/B_w1/rl_r1.npz"
        saved = data.read_bytes()
        changed = bytes([saved[0] ^ 1]) + saved[1:]
        data.write_bytes(changed)
        run("reanalyse.py", False, "tampered_NPZ_rejected_before_arithmetic")
        data.write_bytes(saved)
        manifest_path = isolated / "DATA_MANIFEST.json"
        original_manifest = manifest_path.read_text(encoding="utf-8")
        changed_manifest = json.loads(original_manifest)
        first = next(iter(changed_manifest["files"]))
        changed_manifest["files"][first]["bytes"] += 1
        manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
        run("verify_manifest.py", False, "wrong_manifest_byte_count_rejected")
        changed_manifest = json.loads(original_manifest)
        changed_manifest["files"]["../outside-package"] = {"bytes": 0, "sha256": "0" * 64}
        manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
        run("verify_manifest.py", False, "manifest_path_escape_rejected")
        manifest_path.write_text(original_manifest, encoding="utf-8")
        expected_path = isolated / "RESULTS_EXPECTED.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        expected["metrics"]["equal_source_gain_percent"] += 0.5
        expected_path.write_text(json.dumps(expected), encoding="utf-8")
        run("reanalyse.py", False, "wrong_expected_gain_rejected")
    require(all(sha256(package_path(root, p)) == s for p,s in source_sha.items()), "self-test mutated the source package")
    return {"status": "pass", "scope": "L1 fresh-copy CPU replay and integrity/arithmetic negative controls only", "gpu_used": False,
            "temporary_copy_removed": True, "source_package_unchanged": True, "manifest_files_tested": original["files"], "tests": tests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Write JSON here; default stdout")
    parser.add_argument("--self-test", action="store_true", help="Fresh-copy replay plus tamper rejection tests; temporary files only")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        result = self_test(root) if args.self_test else analyse(root)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            dest = args.output.resolve()
            require(not dest.is_relative_to(root / "data") and not dest.is_relative_to(root / "configs"), "output must not overwrite frozen data/configs")
            require(dest not in (root / "DATA_MANIFEST.json", root / "RESULTS_EXPECTED.json"), "output must not overwrite expected contract")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError, struct.error, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
