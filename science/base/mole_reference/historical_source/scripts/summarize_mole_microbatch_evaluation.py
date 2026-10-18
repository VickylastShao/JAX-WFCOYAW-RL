#!/usr/bin/env python3
"""Combine two held-out Mole microbatch evaluations with clustered CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-result", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_payload(document: dict[str, Any]) -> str:
    payload = dict(document)
    expected = payload.pop("payload_sha256_without_hash_field")
    actual = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual != expected:
        raise ValueError("evaluation payload SHA-256 mismatch")
    return actual


def percentile_interval(samples: np.ndarray) -> list[float]:
    return np.quantile(samples, (0.025, 0.975)).tolist()


def main() -> int:
    args = parse_args()
    if len(args.host_result) != 2:
        raise ValueError("exactly two host results are required")
    if args.bootstrap_samples < 1000:
        raise ValueError("at least 1000 bootstrap samples are required")
    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.host_result
    ]
    payload_hashes = [verify_payload(document) for document in documents]
    rng = np.random.default_rng(args.seed)

    configuration = documents[0]["configuration"]
    source_hashes = [document["source_precursor_sha256"] for document in documents]
    zero_clusters = [
        np.asarray(
            document["controllers"]["zero"]["mean_power_mw_per_environment"],
            dtype=np.float64,
        )
        for document in documents
    ]
    controllers: dict[str, Any] = {
        "zero": {
            "mean_power_mw": float(np.mean(np.concatenate(zero_clusters))),
            "environment_count": int(sum(cluster.size for cluster in zero_clusters)),
        }
    }
    for name in ("static_bo", "sac"):
        controlled_clusters = [
            np.asarray(
                document["controllers"][name]["mean_power_mw_per_environment"],
                dtype=np.float64,
            )
            for document in documents
        ]
        gain_clusters = [
            100.0 * (controlled / zero - 1.0)
            for controlled, zero in zip(
                controlled_clusters, zero_clusters, strict=True
            )
        ]
        gains = np.concatenate(gain_clusters)
        iid_indices = rng.integers(
            0, gains.size, (args.bootstrap_samples, gains.size)
        )
        iid_means = np.mean(gains[iid_indices], axis=1)
        clustered_means = np.empty(args.bootstrap_samples, dtype=np.float64)
        for sample in range(args.bootstrap_samples):
            selected_clusters = rng.integers(0, len(gain_clusters), len(gain_clusters))
            selected_values = []
            for cluster_index in selected_clusters:
                cluster = gain_clusters[int(cluster_index)]
                indices = rng.integers(0, cluster.size, cluster.size)
                selected_values.append(cluster[indices])
            clustered_means[sample] = np.mean(np.concatenate(selected_values))
        cluster_means = [float(np.mean(cluster)) for cluster in gain_clusters]
        controllers[name] = {
            "mean_power_mw": float(np.mean(np.concatenate(controlled_clusters))),
            "mean_paired_gain_percent": float(np.mean(gains)),
            "median_paired_gain_percent": float(np.median(gains)),
            "paired_gain_percent_per_environment": gains.tolist(),
            "positive_gain_fraction": float(np.mean(gains > 0.0)),
            "iid_paired_bootstrap_ci95_percent": percentile_interval(iid_means),
            "clustered_bootstrap_ci95_percent": percentile_interval(clustered_means),
            "mean_gain_percent_per_base_trajectory": cluster_means,
            "environment_count": int(gains.size),
            "base_trajectory_count": len(gain_clusters),
        }

    sac = controllers["sac"]
    h2_checks = {
        "mean_gain_at_least_four_percent": sac["mean_paired_gain_percent"] >= 4.0,
        "clustered_ci_lower_bound_positive": (
            sac["clustered_bootstrap_ci95_percent"][0] > 0.0
        ),
        "evaluation_length_matches_mole": configuration["eval_interactions"] == 2000,
        "evaluation_sample_count_matches_mole": sac["environment_count"] == 224,
    }
    integrity_checks = {
        "both_evaluations_passed": all(
            document["status"] == "pass" for document in documents
        ),
        "configuration_identical": all(
            document["configuration"] == configuration for document in documents[1:]
        ),
        "checkpoint_metadata_identical": (
            documents[0]["checkpoint_metadata"]
            == documents[1]["checkpoint_metadata"]
        ),
        "distinct_base_trajectory_sources": len(set(source_hashes)) == 2,
        "eight_paired_environments": sac["environment_count"] == 8,
        "all_values_finite": bool(
            np.all(
                np.isfinite(
                    np.concatenate(
                        [
                            np.asarray(
                                controllers[name].get(
                                    "paired_gain_percent_per_environment", []
                                ),
                                dtype=np.float64,
                            )
                            for name in ("static_bo", "sac")
                        ]
                    )
                )
            )
        ),
    }
    result = {
        "schema_version": 1,
        "purpose": "mole_heldout_microbatch_evaluation_summary",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if all(integrity_checks.values()) else "fail",
        "integrity_checks": integrity_checks,
        "hypothesis_h2_checks": h2_checks,
        "hypothesis_h2_status": "pass" if all(h2_checks.values()) else "fail",
        "evidence_grade": "provisional_two_base_trajectories_eight_states",
        "configuration": configuration,
        "controllers": controllers,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "iid_unit": "paired precursor state",
            "cluster_unit": "generated base precursor trajectory",
        },
        "limitations": [
            "Only two independently generated base trajectories and eight held-out "
            "states are available.",
            "The clustered confidence interval is unstable with only two clusters.",
            "This evaluation cannot be represented as the article's 224 independent "
            "evaluation episodes.",
        ],
        "sources": {
            "host_results": [str(path) for path in args.host_result],
            "result_payload_sha256": payload_hashes,
            "precursor_sha256": source_hashes,
        },
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
