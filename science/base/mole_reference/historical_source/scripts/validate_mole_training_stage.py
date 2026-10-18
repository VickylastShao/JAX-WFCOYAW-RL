#!/usr/bin/env python3
"""Validate paired results from one staged Mole-aligned JAX training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


GLOBAL_BATCH = 512
MINIMUM_REPLAY_SIZE = 32_000
UPDATES_PER_BATCH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process0", type=Path, required=True)
    parser.add_argument("--process1", type=Path, required=True)
    parser.add_argument("--expected-interactions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def payload_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    expected = payload.pop("payload_sha256_without_hash_field")
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    actual = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError("result payload SHA-256 does not match")
    return actual


def expected_sac_updates(interactions: int) -> int:
    batches = interactions // GLOBAL_BATCH
    first_update_batch = math.ceil(MINIMUM_REPLAY_SIZE / GLOBAL_BATCH)
    active_batches = max(0, batches - first_update_batch + 1)
    return active_batches * UPDATES_PER_BATCH


def validate_one(document: dict[str, Any], interactions: int) -> list[str]:
    errors: list[str] = []
    configuration = document["configuration"]
    final_metrics = document["final_metrics"]
    expected_configuration = {
        "effective_global_sample_size": 256,
        "global_transitions_per_iteration": GLOBAL_BATCH,
        "inlet_mode": "time_planes",
        "local_environments": 16,
        "microbatch": 1,
        "random_action_transitions_global": 32_000,
        "replay_capacity_per_host": 1_000_448,
        "replay_mode": "global",
        "sample_size_per_host": 128,
        "updates_per_iteration": UPDATES_PER_BATCH,
        "zero_yaw_burnin_interactions_per_environment": 150,
    }
    if document["status"] != "pass":
        errors.append("status is not pass")
    for name, passed in document["checks"].items():
        if not passed:
            errors.append(f"check failed: {name}")
    for name, expected in expected_configuration.items():
        if configuration.get(name) != expected:
            errors.append(
                f"configuration {name}={configuration.get(name)!r}, "
                f"expected {expected!r}"
            )
    if configuration["total_global_transitions_including_compile_warmup"] != interactions:
        errors.append("global interaction count does not match target")
    if configuration["terminated_early_nonfinite"]:
        errors.append("run terminated on non-finite values")
    if final_metrics["replay_size"] != interactions:
        errors.append("final replay size does not match interaction count")
    expected_updates = expected_sac_updates(interactions)
    if final_metrics["sac_updates"] != expected_updates:
        errors.append(
            f"SAC updates={final_metrics['sac_updates']}, "
            f"expected {expected_updates}"
        )
    for name, value in final_metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            errors.append(f"non-finite final metric: {name}")
    if document["checkpoint"]["full_state_sha256"] is None:
        errors.append("full-state checkpoint SHA-256 is missing")
    payload_sha256(document)
    return errors


def main() -> int:
    args = parse_args()
    if args.expected_interactions % GLOBAL_BATCH:
        raise ValueError("expected interactions must be divisible by 512")
    documents = [
        json.loads(args.process0.read_text(encoding="utf-8")),
        json.loads(args.process1.read_text(encoding="utf-8")),
    ]
    errors = [validate_one(document, args.expected_interactions) for document in documents]
    cross_errors: list[str] = []
    for name in (
        "actor_replica_checksum_delta",
        "replay_replica_checksum_delta",
        "sac_updates",
        "replay_size",
    ):
        left = documents[0]["final_metrics"][name]
        right = documents[1]["final_metrics"][name]
        if left != right:
            cross_errors.append(f"cross-process mismatch for {name}: {left} != {right}")
    left_hashes = documents[0]["checkpoint"]["metadata"].get(
        "implementation_sha256"
    )
    right_hashes = documents[1]["checkpoint"]["metadata"].get(
        "implementation_sha256"
    )
    if left_hashes is not None and left_hashes != right_hashes:
        cross_errors.append("implementation SHA-256 maps differ across hosts")

    process_seconds = [
        document["timing"]["total_process_wallclock_seconds"]
        for document in documents
    ]
    summary = {
        "schema_version": 1,
        "purpose": "mole_jax_training_stage_validation",
        "status": "pass" if not any(errors) and not cross_errors else "fail",
        "expected_global_interactions": args.expected_interactions,
        "expected_sac_updates": expected_sac_updates(args.expected_interactions),
        "process_wallclock_seconds": process_seconds,
        "critical_path_wallclock_seconds": max(process_seconds),
        "critical_path_wallclock_hours": max(process_seconds) / 3600.0,
        "process_errors": errors,
        "cross_process_errors": cross_errors,
        "result_payload_sha256": [payload_sha256(document) for document in documents],
        "implementation_sha256_in_result": left_hashes,
        "implementation_hash_enforcement": (
            "full_checkpoint_resume_configuration"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
