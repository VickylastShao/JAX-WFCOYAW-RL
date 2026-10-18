#!/usr/bin/env python3
"""Validate and compare paired same-GPU JAX and TorchRL SAC benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax", type=Path, required=True)
    parser.add_argument("--torchrl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_payload(document: dict[str, Any]) -> str:
    payload = dict(document)
    expected = payload.pop("payload_sha256_without_hash_field")
    actual = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual != expected:
        raise ValueError("benchmark payload SHA-256 mismatch")
    return actual


def main() -> int:
    args = parse_args()
    jax_result = json.loads(args.jax.read_text(encoding="utf-8"))
    torchrl_result = json.loads(args.torchrl.read_text(encoding="utf-8"))
    jax_hash = verify_payload(jax_result)
    torchrl_hash = verify_payload(torchrl_result)

    checks = {
        "jax_passed": jax_result["status"] == "pass",
        "torchrl_passed": torchrl_result["status"] == "pass",
        "framework_labels_correct": (
            jax_result["framework"] == "jax"
            and torchrl_result["framework"] == "torchrl"
        ),
        "configuration_identical": (
            jax_result["configuration"] == torchrl_result["configuration"]
        ),
        "jax_cuda_backend": jax_result["environment"].get("backend") == "gpu",
        "torchrl_cuda_backend": bool(torchrl_result["environment"].get("cuda")),
        "parameters_finite": (
            jax_result["measurement"]["parameters_finite"]
            and torchrl_result["measurement"]["parameters_finite"]
        ),
    }
    jax_rate = jax_result["measurement"]["updates_per_second"]
    torchrl_rate = torchrl_result["measurement"]["updates_per_second"]
    speedup = jax_rate / torchrl_rate
    checks["jax_faster_than_torchrl"] = speedup > 1.0
    status = "pass" if all(checks.values()) else "fail"
    comparison = {
        "schema_version": 1,
        "purpose": "mole_aligned_sac_same_gpu_framework_comparison",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "checks": checks,
        "measurement": {
            "jax_updates_per_second": jax_rate,
            "torchrl_updates_per_second": torchrl_rate,
            "jax_samples_per_second": jax_result["measurement"]["samples_per_second"],
            "torchrl_samples_per_second": torchrl_result["measurement"][
                "samples_per_second"
            ],
            "jax_over_torchrl_update_speedup": speedup,
        },
        "environments": {
            "jax": jax_result["environment"],
            "torchrl": torchrl_result["environment"],
        },
        "sources": {
            "jax": str(args.jax),
            "torchrl": str(args.torchrl),
            "jax_payload_sha256": jax_hash,
            "torchrl_payload_sha256": torchrl_hash,
        },
    }
    canonical = json.dumps(comparison, indent=2, sort_keys=True)
    comparison["payload_sha256_without_hash_field"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
