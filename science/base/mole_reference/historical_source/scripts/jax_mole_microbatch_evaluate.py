#!/usr/bin/env python3
"""Memory-efficient paired evaluation of Mole actors on online 224D LES."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import make_mole_training_case
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    PUBLISHED_STATIC_BO_YAW_DEG,
    initialize_continuous_training,
    initialize_evaluation_from_training,
    load_actor_checkpoint,
    load_initial_precursor,
    make_evaluation_step,
    protocol_metadata,
    slice_initial_precursor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precursor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-batch", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--burnin-interactions", type=int, default=150)
    parser.add_argument("--eval-interactions", type=int, default=500)
    parser.add_argument(
        "--inlet-mode",
        choices=("online", "replay_volume"),
        default="online",
    )
    parser.add_argument("--sac-action-scale", type=float, default=1.0)
    parser.add_argument("--mirror-sac-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def make_scan(step, length: int):
    def run(initial):
        def body(state, _):
            return step(state)

        return jax.lax.scan(body, initial, xs=None, length=length)

    return jax.jit(run, donate_argnums=())


def reset_controller_input(state):
    return state._replace(
        observation_frames=jnp.zeros_like(state.observation_frames),
        observation=jnp.zeros_like(state.observation),
        previous_yaw=jnp.zeros_like(state.previous_yaw),
        interaction_index=jnp.zeros_like(state.interaction_index),
    )


def mean_metrics(metrics: dict[str, jax.Array]) -> dict[str, Any]:
    arrays = {key: np.asarray(value) for key, value in metrics.items()}
    return {
        "mean_power_mw_per_environment": np.mean(
            arrays["farm_power_mw"], axis=0
        ).tolist(),
        "mean_turbine_power_mw_per_environment": np.mean(
            arrays["turbine_power_mw"], axis=0
        ).tolist(),
        "mean_reward_per_environment": np.mean(
            arrays["reward"], axis=0
        ).tolist(),
        "mean_yaw_deg": np.mean(arrays["yaw_deg"], axis=(0, 1)).tolist(),
        "max_abs_yaw_deg": float(np.max(np.abs(arrays["yaw_deg"]))),
        "mean_inlet_delta_rms": float(np.mean(arrays["inlet_delta_rms"])),
    }


def main() -> int:
    args = parse_args()
    if args.source_batch % args.microbatch:
        raise ValueError("--microbatch must divide --source-batch")
    source_case = make_mole_training_case(args.source_batch)
    source_precursor, source_metadata = load_initial_precursor(
        args.precursor, source_case
    )
    case = make_mole_training_case(args.microbatch)
    config = ContinuousControlConfig(inlet_mode=args.inlet_mode)
    actor, actor_metadata = load_actor_checkpoint(args.checkpoint)
    _, _, precursor_functions, farm_functions = initialize_continuous_training(
        case,
        config,
        slice_initial_precursor(source_precursor, 0, args.microbatch),
        args.seed,
        256,
        3.0e-6,
        args.microbatch * 16,
    )

    steps = {}
    for name in ("zero", "static_bo", "sac"):
        steps[name] = make_evaluation_step(
            case,
            config,
            precursor_functions,
            farm_functions,
            name,
            actor=actor if name == "sac" else None,
            sac_action_scale=args.sac_action_scale,
            mirror_sac_policy=args.mirror_sac_policy,
        )
    burn_scan = make_scan(steps["zero"], args.burnin_interactions)
    eval_scans = {
        name: make_scan(step, args.eval_interactions)
        for name, step in steps.items()
    }

    controllers: dict[str, dict[str, list[Any]]] = {
        name: {
            "power": [],
            "turbine_power": [],
            "reward": [],
            "yaw": [],
            "max_yaw": [],
            "inlet_delta_rms": [],
            "wallclock_seconds": [],
        }
        for name in steps
    }
    compile_seconds = {name: 0.0 for name in ("burnin", *steps)}
    shard_count = args.source_batch // args.microbatch
    for shard in range(shard_count):
        start = shard * args.microbatch
        stop = start + args.microbatch
        training, _, _, _ = initialize_continuous_training(
            case,
            config,
            slice_initial_precursor(source_precursor, start, stop),
            args.seed + shard,
            256,
            3.0e-6,
            args.microbatch * 16,
        )
        initial = initialize_evaluation_from_training(training)
        started = time.perf_counter()
        common, burn_metrics = burn_scan(initial)
        jax.block_until_ready(burn_metrics["farm_power_mw"])
        if shard == 0:
            compile_seconds["burnin"] = time.perf_counter() - started
        common = reset_controller_input(common)

        for name, run_scan in eval_scans.items():
            started = time.perf_counter()
            _, metrics = run_scan(common)
            jax.block_until_ready(metrics["farm_power_mw"])
            elapsed = time.perf_counter() - started
            if shard == 0:
                compile_seconds[name] = elapsed
            summary = mean_metrics(metrics)
            target = controllers[name]
            target["power"].extend(summary["mean_power_mw_per_environment"])
            target["turbine_power"].extend(
                summary["mean_turbine_power_mw_per_environment"]
            )
            target["reward"].extend(summary["mean_reward_per_environment"])
            target["yaw"].append(summary["mean_yaw_deg"])
            target["max_yaw"].append(summary["max_abs_yaw_deg"])
            target["inlet_delta_rms"].append(
                summary["mean_inlet_delta_rms"]
            )
            target["wallclock_seconds"].append(elapsed)

    zero = np.asarray(controllers["zero"]["power"])
    for name, values in controllers.items():
        power = np.asarray(values["power"])
        values["mean_power_mw_per_environment"] = values.pop("power")
        values["mean_turbine_power_mw_per_environment"] = values.pop(
            "turbine_power"
        )
        values["mean_reward_per_environment"] = values.pop("reward")
        values["mean_yaw_deg_per_microbatch"] = values.pop("yaw")
        values["max_abs_yaw_deg"] = float(max(values.pop("max_yaw")))
        values["mean_inlet_delta_rms"] = float(
            np.mean(values.pop("inlet_delta_rms"))
        )
        values["total_wallclock_seconds"] = float(
            sum(values.pop("wallclock_seconds"))
        )
        if name != "zero":
            paired = 100.0 * (power / zero - 1.0)
            values["paired_gain_percent_per_environment"] = paired.tolist()
            values["local_mean_gain_percent"] = float(paired.mean())

    checks = {
        "gpu_backend": jax.default_backend() == "gpu",
        "source_batch_preserved": len(zero) == args.source_batch,
        "time_resolved_inlet_changes": all(
            values["mean_inlet_delta_rms"] > 1.0e-6
            for values in controllers.values()
        ),
        "yaw_bounds_respected": all(
            values["max_abs_yaw_deg"] <= config.max_yaw_angle_deg + 1.0e-6
            for values in controllers.values()
        ),
        "zero_controller_is_zero": controllers["zero"]["max_abs_yaw_deg"]
        == 0.0,
    }
    result = {
        "schema_version": 1,
        "purpose": "mole_online_microbatch_paired_evaluation",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "configuration": {
            "source_batch": args.source_batch,
            "microbatch": args.microbatch,
            "burnin_interactions": args.burnin_interactions,
            "eval_interactions": args.eval_interactions,
            "inlet_mode": args.inlet_mode,
            "sac_action_scale": args.sac_action_scale,
            "mirror_sac_policy": args.mirror_sac_policy,
            "static_bo_target_deg": list(PUBLISHED_STATIC_BO_YAW_DEG),
        },
        "compile_inclusive_first_shard_seconds": compile_seconds,
        "controllers": controllers,
        "checkpoint_metadata": actor_metadata,
        "source_precursor_sha256": source_metadata["npz_sha256"],
        "protocol": protocol_metadata(case, config),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    result["payload_sha256_without_hash_field"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
