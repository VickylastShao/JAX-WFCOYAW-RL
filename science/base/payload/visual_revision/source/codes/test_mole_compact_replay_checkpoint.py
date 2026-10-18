#!/usr/bin/env python3
"""Round-trip the compact replay checkpoint at small or production capacity."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import ReplayState
from jax_mole_microbatch_distributed import (
    FULL_CHECKPOINT_SCHEMA_VERSION,
    _compact_replay_to_host,
    _compact_replay_to_sharded_device,
    _load_full_checkpoint,
    _save_full_checkpoint_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=17)
    parser.add_argument("--active-size", type=int, default=5)
    parser.add_argument("--observation-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zero-data",
        action="store_true",
        help="Avoid patterned host initialization for production-size tests.",
    )
    return parser.parse_args()


def _array(
    shape: tuple[int, ...], active_size: int, scale: float, zero: bool
) -> jax.Array:
    host = np.zeros(shape, dtype=np.float32)
    if not zero and active_size:
        active_shape = (1, active_size, *shape[2:])
        values = np.arange(np.prod(active_shape), dtype=np.float32).reshape(
            active_shape
        )
        host[:, :active_size] = values * np.float32(scale)
    device = jax.device_put(host)
    jax.block_until_ready(device)
    del host
    return device


def _make_replay(args: argparse.Namespace) -> ReplayState:
    capacity = args.capacity
    active_size = args.active_size
    if capacity <= 0 or not 0 <= active_size <= capacity:
        raise ValueError("active size must be between zero and capacity")
    return ReplayState(
        observations=_array(
            (1, capacity, args.observation_dim),
            active_size,
            0.01,
            args.zero_data,
        ),
        actions=_array(
            (1, capacity, args.action_dim),
            active_size,
            0.02,
            args.zero_data,
        ),
        rewards=_array(
            (1, capacity), active_size, 0.03, args.zero_data
        ),
        next_observations=_array(
            (1, capacity, args.observation_dim),
            active_size,
            0.04,
            args.zero_data,
        ),
        dones=_array(
            (1, capacity), active_size, 0.0, True
        ),
        position=jax.device_put(
            np.asarray([active_size % capacity], dtype=np.int32)
        ),
        size=jax.device_put(np.asarray([active_size], dtype=np.int32)),
    )


def _active_checksums(replay: ReplayState, size: int) -> dict[str, float]:
    fields = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
    )
    return {
        name: float(
            np.asarray(jax.device_get(getattr(replay, name)))[:, :size].sum(
                dtype=np.float64
            )
        )
        for name in fields
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    replay = _make_replay(args)
    initialized_seconds = time.perf_counter() - started
    expected_checksums = _active_checksums(replay, args.active_size)

    pack_started = time.perf_counter()
    compact = _compact_replay_to_host(replay)
    pack_seconds = time.perf_counter() - pack_started
    checks = {
        "format": compact["format"] == "active_prefix_v1",
        "capacity": compact["capacity"] == args.capacity,
        "size": compact["size"] == args.active_size,
        "position": compact["position"] == args.active_size % args.capacity,
        "stored_rows": all(
            value.shape[0] == args.active_size
            for value in compact["arrays"].values()
        ),
    }

    payload = {
        "schema_version": FULL_CHECKPOINT_SCHEMA_VERSION,
        "purpose": "compact_replay_checkpoint_smoke",
        "replay": compact,
    }
    save_started = time.perf_counter()
    digest = _save_full_checkpoint_atomic(args.output, payload)
    save_seconds = time.perf_counter() - save_started
    loaded = _load_full_checkpoint(args.output)
    checks["checkpoint_schema"] = (
        loaded["schema_version"] == FULL_CHECKPOINT_SCHEMA_VERSION
    )

    del replay
    gc.collect()
    jax.clear_caches()

    restore_started = time.perf_counter()
    restored = _compact_replay_to_sharded_device(loaded["replay"])
    restore_seconds = time.perf_counter() - restore_started
    actual_checksums = _active_checksums(restored, args.active_size)
    checks.update(
        {
            "restored_size": int(np.asarray(restored.size)[0])
            == args.active_size,
            "restored_position": int(np.asarray(restored.position)[0])
            == args.active_size % args.capacity,
            "restored_observation_shape": restored.observations.shape
            == (1, args.capacity, args.observation_dim),
            "restored_action_shape": restored.actions.shape
            == (1, args.capacity, args.action_dim),
            "active_checksums": actual_checksums == expected_checksums,
        }
    )
    if args.active_size < args.capacity:
        tail_index = args.active_size
        checks["zero_tail"] = all(
            float(
                np.asarray(
                    jax.device_get(getattr(restored, name))
                )[0, tail_index].sum()
            )
            == 0.0
            for name in (
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "dones",
            )
        )

    report = {
        "schema_version": 1,
        "purpose": "compact_replay_checkpoint_roundtrip",
        "status": "pass" if all(checks.values()) else "fail",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "capacity": args.capacity,
        "active_size": args.active_size,
        "observation_dim": args.observation_dim,
        "action_dim": args.action_dim,
        "full_allocation_bytes": compact["full_allocation_bytes"],
        "stored_replay_bytes": compact["stored_bytes"],
        "checkpoint_bytes": args.output.stat().st_size,
        "checkpoint_sha256": digest,
        "timing_seconds": {
            "initialize": initialized_seconds,
            "pack": pack_seconds,
            "save_and_hash": save_seconds,
            "restore": restore_seconds,
        },
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
