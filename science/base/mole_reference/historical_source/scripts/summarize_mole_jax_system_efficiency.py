#!/usr/bin/env python3
"""Summarize the staged Mole-aligned JAX efficiency experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-validation", type=Path, action="append", required=True)
    parser.add_argument("--final-process0", type=Path, required=True)
    parser.add_argument("--final-process1", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--online-evaluation", type=Path)
    parser.add_argument("--framework-comparison", type=Path)
    parser.add_argument("--paper-hours", type=float, default=18.5)
    parser.add_argument("--paper-nodes", type=int, default=34)
    parser.add_argument("--archive-plane-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_payload(document: dict[str, Any]) -> str:
    payload = dict(document)
    expected = payload.pop("payload_sha256_without_hash_field")
    actual = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual != expected:
        raise ValueError("payload SHA-256 mismatch")
    return actual


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if len(args.stage_validation) != 3:
        raise ValueError("exactly three stage validations are required")
    stages = [load(path) for path in args.stage_validation]
    final_documents = [load(args.final_process0), load(args.final_process1)]
    final_hashes = [verify_payload(document) for document in final_documents]
    expected_targets = [32768, 100352, 1000448]
    if [stage["expected_global_interactions"] for stage in stages] != expected_targets:
        raise ValueError("stage validations are not ordered S1, S2, S3")

    critical_seconds = sum(
        stage["critical_path_wallclock_seconds"] for stage in stages
    )
    critical_hours = critical_seconds / 3600.0
    paper_speed_ratio = args.paper_hours / critical_hours
    final_metrics = [document["final_metrics"] for document in final_documents]
    final_cursors = [document["configuration"]["plane_cursors"] for document in final_documents]
    h1 = {
        "staged_critical_path_below_paper_hours": critical_hours < args.paper_hours,
    }
    h3 = {
        "all_stage_validations_passed": all(stage["status"] == "pass" for stage in stages),
        "both_final_processes_passed": all(
            document["status"] == "pass" for document in final_documents
        ),
        "all_embedded_checks_passed": all(
            all(document["checks"].values()) for document in final_documents
        ),
        "expected_final_replay_size": all(
            metric["replay_size"] == 1000448 for metric in final_metrics
        ),
        "expected_final_sac_updates": all(
            metric["sac_updates"] == 968704 for metric in final_metrics
        ),
        "parameters_finite": all(
            metric["parameters_finite"] == 1.0 for metric in final_metrics
        ),
        "actor_replicas_synchronized": all(
            metric["actor_replica_checksum_delta"] == 0.0
            for metric in final_metrics
        ),
        "replay_replicas_synchronized": all(
            metric["replay_replica_checksum_delta"] == 0.0
            for metric in final_metrics
        ),
        "archive_consumed_without_wrap": all(
            max(cursors) == args.archive_plane_count
            and min(cursors) >= 0
            and len(set(cursors)) == len(cursors)
            for cursors in final_cursors
        ),
    }

    evaluation = None
    h2 = {
        "available_evaluation_integrity": False,
        "available_mean_gain_at_least_four_percent": False,
        "available_clustered_ci_lower_bound_positive": False,
        "mole_224_episode_sample_count_matched": False,
    }
    if args.evaluation is not None:
        evaluation_document = load(args.evaluation)
        evaluation_hash = verify_payload(evaluation_document)
        sac = evaluation_document["controllers"]["sac"]
        h2 = {
            "available_evaluation_integrity": evaluation_document["status"] == "pass",
            "available_mean_gain_at_least_four_percent": (
                sac["mean_paired_gain_percent"] >= 4.0
            ),
            "available_clustered_ci_lower_bound_positive": (
                sac["clustered_bootstrap_ci95_percent"][0] > 0.0
            ),
            "mole_224_episode_sample_count_matched": (
                sac["environment_count"] == 224
            ),
        }
        evaluation = {
            "source": str(args.evaluation),
            "payload_sha256": evaluation_hash,
            "evidence_grade": evaluation_document["evidence_grade"],
            "sac": sac,
            "static_bo": evaluation_document["controllers"]["static_bo"],
        }

    framework = None
    h4 = {"same_gpu_jax_faster_than_torchrl": False}
    if args.framework_comparison is not None:
        framework_document = load(args.framework_comparison)
        framework_hash = verify_payload(framework_document)
        h4 = {
            "same_gpu_jax_faster_than_torchrl": bool(
                framework_document["checks"]["jax_faster_than_torchrl"]
            ),
            "framework_comparison_passed": framework_document["status"] == "pass",
        }
        framework = {
            "source": str(args.framework_comparison),
            "payload_sha256": framework_hash,
            "measurement": framework_document["measurement"],
        }

    online_evaluation = None
    online_transfer_checks = {
        "online_evaluation_integrity": False,
        "online_mean_gain_at_least_four_percent": False,
        "online_clustered_ci_lower_bound_positive": False,
    }
    if args.online_evaluation is not None:
        online_document = load(args.online_evaluation)
        online_hash = verify_payload(online_document)
        online_sac = online_document["controllers"]["sac"]
        online_transfer_checks = {
            "online_evaluation_integrity": online_document["status"] == "pass",
            "online_mean_gain_at_least_four_percent": (
                online_sac["mean_paired_gain_percent"] >= 4.0
            ),
            "online_clustered_ci_lower_bound_positive": (
                online_sac["clustered_bootstrap_ci95_percent"][0] > 0.0
            ),
        }
        online_evaluation = {
            "source": str(args.online_evaluation),
            "payload_sha256": online_hash,
            "evidence_grade": online_document["evidence_grade"],
            "sac": online_sac,
            "static_bo": online_document["controllers"]["static_bo"],
        }

    checks = {
        "h1_runtime": all(h1.values()),
        "h2_available_performance": all(
            value for name, value in h2.items() if not name.startswith("mole_224")
        ),
        "h2_publication_sample_count": h2["mole_224_episode_sample_count_matched"],
        "h3_training_validity": all(h3.values()),
        "h4_framework_attribution": all(h4.values()),
        "online_transfer_performance": all(online_transfer_checks.values()),
    }
    numerical_values = [
        metric[name]
        for metric in final_metrics
        for name in ("actor_loss", "critic_loss", "alpha", "actor_abs_max", "critic_abs_max")
    ]
    if not all(math.isfinite(value) for value in numerical_values):
        raise ValueError("non-finite final metrics")

    result = {
        "schema_version": 1,
        "purpose": "mole_jax_system_efficiency_final_summary",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if checks["h1_runtime"] and checks["h3_training_validity"] else "fail",
        "checks": checks,
        "hypotheses": {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "online_transfer": online_transfer_checks,
        },
        "timing": {
            "stage_critical_path_hours": [
                stage["critical_path_wallclock_hours"] for stage in stages
            ],
            "staged_critical_path_seconds": critical_seconds,
            "staged_critical_path_hours": critical_hours,
            "paper_reported_hours": args.paper_hours,
            "paper_hours_over_jax_hours_ratio": paper_speed_ratio,
            "wallclock_reduction_percent": 100.0
            * (args.paper_hours - critical_hours)
            / args.paper_hours,
        },
        "resource_accounting": {
            "jax_gpu_count": 2,
            "jax_gpu_hours": 2.0 * critical_hours,
            "paper_reported_node_count": args.paper_nodes,
            "paper_reported_node_hours": args.paper_nodes * args.paper_hours,
            "comparability_boundary": (
                "GPU-hours and ARCHER2 node-hours are descriptive resource counts, "
                "not equivalent compute or energy units."
            ),
        },
        "final_training": {
            "global_interactions": 1000448,
            "sac_updates": 968704,
            "process_metrics": final_metrics,
            "archive_plane_count_per_host": args.archive_plane_count,
            "final_plane_cursors_per_host": final_cursors,
        },
        "evaluation": evaluation,
        "online_transfer_evaluation": online_evaluation,
        "framework_attribution": framework,
        "sources": {
            "stage_validations": [str(path) for path in args.stage_validation],
            "final_process_results": [str(args.final_process0), str(args.final_process1)],
            "final_process_payload_sha256": final_hashes,
        },
        "claim_boundary": (
            "H1 and H3 support a measured system-level efficiency result. "
            "The framework-only attribution requires H4, and Mole-equivalent "
            "policy evidence requires 224 independent evaluation episodes."
        ),
    }
    canonical = json.dumps(result, indent=2, sort_keys=True)
    result["payload_sha256_without_hash_field"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
