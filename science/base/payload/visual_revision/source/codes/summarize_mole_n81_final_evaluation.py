"""CPU-only, identity-checked summary of the frozen N81 R1 confirmation.

No policy loading, threshold tuning, or bootstrap confidence interval is done.
RESULT identities are checked against PLAN; saved traces are independently
hashed and scored. Missing or invalid contexts never produce a full estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np

from mole_large_scale_metrics import paired_gain, score_trace


N81_LAYOUT_SHA256 = "67a41a873c604be2b7fce5d3ded7f26edd105d0e7b5c02a75b35876d3f2d841f"
STATISTICAL_LIMITATIONS = [
    "Only two independent physical precursor sources B/C; four windows per source are correlated contexts, not independent physical replicates.",
    "Only the predeclared final R1 Actor and one optimization seed are evaluated; training-seed stability is not established.",
    "No confirmatory bootstrap confidence interval is reported: the 400-control A pilot does not qualify slow inlet correlation.",
    "Results describe the specified LES inflow and fixed windows; strict statistical stationarity and population generalization are not established.",
    "Generation energy is gross electricity; actuator energy is not modeled. Zero is the physical reference; no minimum positive-gain gate is imposed.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_plan(plan):
    evaluation = plan["evaluation"]
    require(evaluation.get("burnin") == 150 and evaluation.get("scored_interactions") == 2000
            and evaluation.get("control_dt") == 10 and evaluation.get("les_steps_per_control") == 50,
            "PLAN does not match the preregistered 150/2000 control horizon")
    require(evaluation.get("primary_confidence_interval") is False,
            "PLAN must explicitly preregister no confirmatory confidence interval")
    require(evaluation.get("negative_gains_retained") is True
            and evaluation.get("minimum_positive_gain_threshold") is None,
            "PLAN must retain negatives and impose no positive-gain gate")
    require(evaluation.get("training_seeds_evaluated") == 1
            and evaluation.get("independent_sources") == 2,
            "PLAN must identify one learner seed and two physical sources")
    actor = plan["actor_final"]
    require(valid_sha(actor.get("sha256")) and actor.get("actual_global_interactions") == 1000448,
            "PLAN lacks the final fixed-budget Actor identity")
    sources = {source["source_id"]: source for source in plan["sources"]}
    require(len(sources) == len(plan["sources"]) and {"B", "C"} <= set(sources),
            "PLAN has duplicate or missing sources")
    for name in ("B", "C"):
        source = sources[name]
        require(valid_sha(source.get("source_state_sha256")) and valid_sha(source.get("manifest_sha256")),
                "PLAN source lacks hashes: " + name)
        require(isinstance(source.get("seed"), int) and not isinstance(source.get("seed"), bool),
                "PLAN source lacks an integer seed: " + name)
    require(sources["B"]["seed"] != sources["C"]["seed"]
            and sources["B"]["source_state_sha256"] != sources["C"]["source_state_sha256"],
            "B/C cannot share a seed or source state")
    contexts = plan["contexts"]
    ids = [context["context_id"] for context in contexts]
    require(len(contexts) == 8 and len(set(ids)) == 8, "PLAN requires eight distinct contexts")
    for name in ("B", "C"):
        group = [context for context in contexts if context["source_id"] == name]
        require(len(group) == 4, "PLAN requires four windows per source")
        require(sorted(context["window_start"] for context in group) == [20000, 147500, 275000, 402500],
                "PLAN changed the predeclared window starts")
    for context in contexts:
        require(re.fullmatch(r"[A-Za-z0-9_-]+", context["context_id"]) is not None,
                "unsafe context_id")
        require(context["source_id"] in ("B", "C"), "confirmation context is not B/C")
        require(context["window_end_exclusive"] == context["window_start"] + 107500,
                "PLAN window does not cover burnin and scoring")
        require(context["window_end_exclusive"] <= sources[context["source_id"]]["completed_planes"],
                "PLAN window exceeds its source archive")
    return sources


def _close(actual, expected, label):
    values = np.asarray(actual, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    require(values.shape == target.shape and np.isfinite(values).all()
            and np.allclose(values, target, rtol=1e-7, atol=1e-7),
            "saved metric disagrees with independently scored trace: " + label)


def _artifact(batch, context_dir, recorded, filename):
    require(isinstance(recorded, str), "missing trace path")
    raw = Path(recorded)
    if raw.is_absolute():
        path = batch / raw.relative_to("/batch") if raw.is_relative_to("/batch") else raw
    else:
        path = context_dir / raw
    expected = (context_dir / filename).resolve()
    require(path.resolve() == expected and expected.parent == context_dir.resolve(),
            "trace path is outside its expected context/branch")
    return expected


def validate_context(batch, plan, sources, expected, seen_files, hash_groups):
    context_id = expected["context_id"]
    directory = batch / "contexts" / context_id
    result_path = directory / "RESULT.json"
    result = read_json(result_path)
    source = sources[expected["source_id"]]
    require(result.get("status") == "complete" and result.get("context_id") == context_id,
            "wrong context status or identity")
    require(result.get("turbines") == 81 and result.get("layout_sha256") == N81_LAYOUT_SHA256,
            "wrong turbine count or layout")
    require(result.get("source_state", {}).get("sha256") == source["source_state_sha256"]
            and result.get("source_state", {}).get("seed") == source["seed"],
            "source seed/state identity mismatch")
    require(result.get("archive_manifest_sha256") == source["manifest_sha256"],
            "archive manifest identity mismatch")
    require(result.get("window_start") == expected["window_start"]
            and result.get("window_end_exclusive") == expected["window_end_exclusive"],
            "window identity mismatch")
    require(set(result.get("controllers", {})) == {"zero", "rl_r1"}
            and set(result.get("scores", {})) == {"zero", "rl_r1"}
            and set(result.get("pairs", {})) == {"rl_r1"},
            "expected exactly one zero and one final R1 branch")
    common_hash = result.get("common_state_sha256")
    require(valid_sha(common_hash) and sha256_file(directory / "common_state.pkl") == common_hash,
            "common state hash mismatch")
    scoring_cursor = expected["window_start"] + 150 * 50
    burnin = result.get("common_burnin", {})
    require(burnin.get("controls") == 150 and burnin.get("duration_seconds") == 1500
            and burnin.get("initial_plane_cursor") == expected["window_start"]
            and burnin.get("final_plane_cursor_exclusive") == scoring_cursor
            and burnin.get("excluded_from_scoring") is True,
            "common burnin duration/cursor/exclusion contract mismatch")
    burnin_path = _artifact(batch, directory, burnin.get("trace"), "common_burnin.npz")
    require(valid_sha(burnin.get("trace_sha256"))
            and sha256_file(burnin_path) == burnin["trace_sha256"], "common burnin trace hash mismatch")
    with np.load(burnin_path, allow_pickle=False) as arrays:
        burnin_trace = {name: arrays[name] for name in ("farm_power_mw", "turbine_power_mw", "yaw_deg")}
    require(burnin_trace["farm_power_mw"].shape == (150, 1)
            and burnin_trace["turbine_power_mw"].shape == (150, 1, 81)
            and burnin_trace["yaw_deg"].shape == (150, 1, 81), "common burnin trace shape mismatch")
    require(np.all(burnin_trace["yaw_deg"] == 0), "common burnin was not zero yaw")
    # Check finite physics and farm/turbine accounting without adding burnin
    # energy to the scored window or deriving an effect-based acceptance gate.
    score_trace(burnin_trace, np.zeros((1, 81)))
    scores, identities, files, content = {}, {}, [], []
    for branch in ("zero", "rl_r1"):
        record = result["controllers"][branch]
        require(record.get("common_state_sha256") == common_hash
                and record.get("scoring_cursor") == scoring_cursor
                and record.get("final_cursor_exclusive") == expected["window_end_exclusive"],
                "branch common state or scoring cursor mismatch: " + branch)
        if branch == "zero":
            require(record.get("actor") is None, "zero branch contains an Actor")
        else:
            identity = record.get("actor", {})
            metadata = identity.get("metadata", {})
            require(identity.get("sha256") == plan["actor_final"]["sha256"]
                    and metadata.get("actual_global_interactions") == 1000448
                    and metadata.get("policy_architecture") == "graph_sac_v2"
                    and metadata.get("action_parameterization") == "per_turbine",
                    "R1 is not the frozen final fixed-budget Actor")
        path = _artifact(batch, directory, record.get("trace"), branch + ".npz")
        digest = sha256_file(path)
        require(valid_sha(record.get("trace_sha256")) and digest == record["trace_sha256"],
                "trace hash mismatch: " + branch)
        stat = path.stat()
        file_identity = (stat.st_dev, stat.st_ino)
        require(file_identity not in seen_files and file_identity not in files,
                "duplicated trajectory file/inode")
        with np.load(path, allow_pickle=False) as arrays:
            trace = {name: arrays[name] for name in ("farm_power_mw", "turbine_power_mw", "yaw_deg")}
        require(trace["farm_power_mw"].shape == (2000, 1)
                and trace["turbine_power_mw"].shape == (2000, 1, 81)
                and trace["yaw_deg"].shape == (2000, 1, 81),
                "trace shape is not the full one-environment confirmation window")
        require(np.all(trace["farm_power_mw"] >= 0) and np.all(trace["turbine_power_mw"] >= 0),
                "negative physical generation in trace")
        if branch == "zero":
            require(np.all(trace["yaw_deg"] == 0), "zero-yaw branch contains yaw actuation")
        score = score_trace(trace, np.zeros((1, 81)), control_dt=10., max_angle=40., max_rate=1.)
        require(score["rate_limit_satisfied"] and score["angle_limit_satisfied"],
                "actuator constraint violation: " + branch)
        saved = result["scores"][branch]
        require(saved.get("controls") == 2000 and saved.get("duration_seconds") == 20000,
                "saved scoring duration mismatch: " + branch)
        for key, value in score.items():
            if isinstance(value, (float, int, list)) and not isinstance(value, bool):
                _close(saved.get(key), value, branch + "." + key)
            else:
                require(saved.get(key) == value, "saved scoring contract mismatch: " + branch + "." + key)
        scores[branch] = score
        files.append(file_identity)
        content.append((digest, {"context_id": context_id, "branch": branch}))
        identities[branch] = {"trace": str(path.relative_to(batch)), "trace_sha256": digest,
                              "common_state_sha256": common_hash, "scoring_cursor": scoring_cursor,
                              "final_cursor_exclusive": expected["window_end_exclusive"]}
    gain = paired_gain(scores["rl_r1"], scores["zero"])
    for key, value in gain.items():
        _close(result["pairs"]["rl_r1"].get(key), value, "pair." + key)
    seen_files.update(files)
    for digest, identity in content:
        hash_groups.setdefault(digest, []).append(identity)
    return {"context_id": context_id, "source_id": expected["source_id"],
            "source_seed": source["seed"], "source_state_sha256": source["source_state_sha256"],
            "archive_manifest_sha256": source["manifest_sha256"],
            "window_start": expected["window_start"], "window_end_exclusive": expected["window_end_exclusive"],
            "common_burnin": burnin,
            "result_sha256": sha256_file(result_path), "trajectories": identities,
            "scores": scores, "gain": gain, "negative_gain": gain["generation_gain_percent"] < 0}


def _source_summary(name, windows):
    zero = math.fsum(window["scores"]["zero"]["generation_energy_mwh"] for window in windows)
    rl = math.fsum(window["scores"]["rl_r1"]["generation_energy_mwh"] for window in windows)
    require(zero > 0, "source baseline energy must be positive")
    travel = np.sum([window["scores"]["rl_r1"]["yaw_travel_deg_per_turbine"] for window in windows], axis=0)
    turbine_delta = np.sum([np.asarray(window["scores"]["rl_r1"]["turbine_generation_energy_mwh"])
                            - window["scores"]["zero"]["turbine_generation_energy_mwh"] for window in windows], axis=0)
    return {"source_id": name, "contexts": [window["context_id"] for window in windows],
            "baseline_generation_mwh": zero, "rl_generation_mwh": rl,
            "absolute_generation_gain_mwh": rl - zero, "generation_gain_percent": 100 * (rl - zero) / zero,
            "negative_gain": rl < zero,
            "negative_contexts": [window["context_id"] for window in windows if window["negative_gain"]],
            "turbine_generation_gain_mwh": turbine_delta.tolist(),
            "rl_yaw_travel_deg_per_turbine": travel.tolist(),
            "rl_mean_total_yaw_travel_deg_per_turbine": float(travel.mean()),
            "rl_peak_yaw_rate_deg_s": max(window["scores"]["rl_r1"]["peak_yaw_rate_deg_s"] for window in windows),
            "rl_maximum_abs_yaw_deg": max(window["scores"]["rl_r1"]["maximum_abs_yaw_deg"] for window in windows)}


def summarize(batch):
    batch = Path(batch).resolve()
    plan_path = batch / "PLAN.json"
    summary = {"schema_version": 1, "status": "invalid", "task_id": None,
               "plan_sha256": sha256_file(plan_path), "expected_contexts": 8, "expected_trajectories": 16,
               "primary_confidence_interval": None, "bootstrap_performed": False,
               "statistical_limitations": STATISTICAL_LIMITATIONS,
               "missing_contexts": [], "invalid_contexts": [], "windows": [],
               "primary_equal_source_gain_percent": None, "pooled_secondary": None, "sources": []}
    try:
        plan = read_json(plan_path)
        sources = validate_plan(plan)
        summary.update(task_id=plan.get("task_id"), actor_final=plan["actor_final"],
                       preregistered_confidence_interval_reason=plan["evaluation"].get("confidence_interval_reason"))
    except (ValueError, KeyError, TypeError, OSError) as error:
        summary["invalid_contexts"].append({"context_id": "PLAN", "error": str(error)})
        return summary
    seen_files, hash_groups = set(), {}
    for context in plan["contexts"]:
        context_id = context["context_id"]
        if not (batch / "contexts" / context_id / "RESULT.json").is_file():
            summary["missing_contexts"].append(context_id)
            continue
        try:
            window = validate_context(batch, plan, sources, context, seen_files, hash_groups)
            summary["windows"].append(window)
        except (ValueError, KeyError, TypeError, OSError) as error:
            summary["invalid_contexts"].append({"context_id": context_id, "error": str(error)})
    summary["validated_contexts"] = len(summary["windows"])
    summary["validated_trajectories"] = len(seen_files)
    summary["identical_trace_content_groups"] = [
        {"trace_sha256": digest, "trajectories": identities}
        for digest, identities in hash_groups.items() if len(identities) > 1]
    if summary["missing_contexts"] or summary["invalid_contexts"]:
        summary["status"] = "incomplete" if summary["missing_contexts"] else "invalid"
        return summary
    require(len(seen_files) == 16, "confirmation must contain sixteen distinct trajectory files")
    summary["sources"] = [_source_summary(name, [window for window in summary["windows"] if window["source_id"] == name])
                          for name in ("B", "C")]
    summary["primary_equal_source_gain_percent"] = math.fsum(source["generation_gain_percent"] for source in summary["sources"]) / 2
    total_zero = math.fsum(source["baseline_generation_mwh"] for source in summary["sources"])
    total_rl = math.fsum(source["rl_generation_mwh"] for source in summary["sources"])
    summary["pooled_secondary"] = {"baseline_generation_mwh": total_zero, "rl_generation_mwh": total_rl,
                                   "absolute_generation_gain_mwh": total_rl - total_zero,
                                   "generation_gain_percent": 100 * (total_rl - total_zero) / total_zero}
    summary["negative_contexts"] = [window["context_id"] for window in summary["windows"] if window["negative_gain"]]
    summary["negative_sources"] = [source["source_id"] for source in summary["sources"] if source["negative_gain"]]
    summary["status"] = "complete"
    return summary


def render_markdown(summary):
    lines = ["# 81风机R1独立评价", "", "状态：" + summary["status"] + "。",
             f"已核验{summary.get('validated_contexts', 0)}/8个窗口、{summary.get('validated_trajectories', 0)}/16条轨迹。", ""]
    if summary["status"] == "complete":
        lines += [f"主指标（B/C来源级能量收益等权平均）：**{summary['primary_equal_source_gain_percent']:.6f}%**。",
                  f"次指标（全部窗口合并能量比）：{summary['pooled_secondary']['generation_gain_percent']:.6f}%；"
                  f"绝对发电量增减{summary['pooled_secondary']['absolute_generation_gain_mwh']:.6f} MWh。", "",
                  "| 来源 | 零偏航 MWh | R1 MWh | 增减 MWh | 收益 % |", "|---|---:|---:|---:|---:|"]
        for source in summary["sources"]:
            lines.append(f"| {source['source_id']} | {source['baseline_generation_mwh']:.6f} | {source['rl_generation_mwh']:.6f} | "
                         f"{source['absolute_generation_gain_mwh']:.6f} | {source['generation_gain_percent']:.6f} |")
        lines += ["", "| 窗口 | 来源 | 增减 MWh | 收益 % | R1最大偏航角 ° | R1峰值速率 °/s |", "|---|---|---:|---:|---:|---:|"]
        for window in summary["windows"]:
            score = window["scores"]["rl_r1"]
            lines.append(f"| {window['context_id']} | {window['source_id']} | {window['gain']['absolute_generation_gain_mwh']:.6f} | "
                         f"{window['gain']['generation_gain_percent']:.6f} | {score['maximum_abs_yaw_deg']:.6f} | {score['peak_yaw_rate_deg_s']:.6f} |")
        lines += ["", "负收益来源：" + ("、".join(summary["negative_sources"]) or "无") + "。",
                  "负收益窗口：" + ("、".join(summary["negative_contexts"]) or "无") + "。"]
    else:
        lines.append("评价尚未完整通过身份与轨迹核验；不计算或宣称完整主指标。")
        if summary["missing_contexts"]:
            lines += ["", "缺失窗口：" + "、".join(summary["missing_contexts"]) + "。"]
        if summary["invalid_contexts"]:
            lines += ["", "未通过核验："]
            lines += ["- " + item["context_id"] + "：" + item["error"] for item in summary["invalid_contexts"]]
    lines += ["", "仅两根独立物理来源、一个训练种子；同源四窗口存在相关性。400步A开发段不足以确认入口慢相关，"
              "本轮不提供确认性bootstrap置信区间，不据此声称总体显著性或训练稳定性。",
              "收益指发电量增减，未扣除偏航执行器耗电；所有负收益保留。完整执行器行程与逐风机能量见EVALUATION_SUMMARY.json。",
              "", "身份、文件哈希和指标由独立汇总器核对；Actor SHA在RESULT与冻结PLAN之间绑定，不加载pickle或重新调参。", ""]
    return "\n".join(lines)


def write_atomic(path, text):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.batch)
    write_atomic(args.batch / "EVALUATION_SUMMARY.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_atomic(args.batch / "RESULTS.md", render_markdown(summary))
    print(json.dumps({"status": summary["status"], "validated_contexts": summary.get("validated_contexts", 0),
                      "missing_contexts": summary["missing_contexts"], "invalid_contexts": summary["invalid_contexts"],
                      "primary_equal_source_gain_percent": summary["primary_equal_source_gain_percent"]}, allow_nan=False))
    return {"complete": 0, "incomplete": 2, "invalid": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
