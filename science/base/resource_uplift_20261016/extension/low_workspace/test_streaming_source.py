#!/usr/bin/env python3
"""CPU-only frozen-source checks, including batch/padding/first-step semantics."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("this test explicitly requires JAX_PLATFORMS=cpu")
    sys.path.insert(0, str(args.source_dir))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jax_mole_compact_sac_benchmark as core
    from streaming_adm import build_streaming_flow_functions
    from benchmark_full_flow import compare
    case = replace(core.CompactMoleCase(batch=2), nx=65, ny=9)
    dense = core.build_flow_functions(case, False, True)
    rng = np.random.default_rng(81731)
    velocity = jnp.asarray(rng.normal(0., .2, (2, 3, case.nx, case.ny, case.nz)).astype(np.float32))
    velocity = velocity.at[:, 0].add(7.5)
    old = jnp.asarray(rng.uniform(4., 7., (2, 3)).astype(np.float32))
    records = []
    for group in (1, 2, 3):
        stream = build_streaming_flow_functions(core, case, group_size=group)
        source = jax.jit(stream["disk_source"])
        for yaws in ([[0., 0., 0.], [0., 0., 0.]],
                     [[-40., 17., 40.], [31., -23., -39.]]):
            yaw = jnp.asarray(yaws, dtype=jnp.float32)
            for first in (True, False):
                reference = dense["disk_source"](velocity, yaw, dense["disk_weights"](yaw), old, jnp.asarray(first))
                result = source(velocity, yaw, stream["disk_weights"](yaw), old, jnp.asarray(first))
                comparison = {key: compare(a, b, np) for key, a, b in zip(
                    ["acceleration", "filtered_speed", "power", "thrust"], result, reference, strict=True)}
                if not all(v["finite"] and v["allclose_rtol1e-4_atol1e-5"] for v in comparison.values()):
                    raise AssertionError(comparison)
                # A physically meaningful invariant checks force scale and direction.
                acceleration, filtered, power, thrust = jax.tree.map(np.asarray, result)
                yaw_rad = np.deg2rad(np.asarray(yaw))
                normals = np.stack((np.cos(yaw_rad), np.zeros_like(yaw_rad), -np.sin(yaw_rad)), axis=2)
                expected = -np.sum(thrust[:, :, None] * normals, axis=1)
                actual = np.sum(acceleration.astype(np.float64), axis=(-3, -2, -1)) * case.density * case.dx * case.dy * case.dz
                force_error = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
                if force_error > 2e-6:
                    raise AssertionError(("momentum-source invariant", force_error))
                records.append({"group": group, "first_step": first, "yaw": yaws,
                                "comparison": comparison, "momentum_relative_error": force_error})
    payload = {"status": "pass", "case": "2 batches, 3 turbines, full-support 65x9x72 grid",
               "checks": len(records), "includes_non_divisor_padded_group": True,
               "records": records, "jax_version": jax.__version__, "devices": [str(x) for x in jax.devices()],
               "source_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
               "scope": "CPU source-equation and integrated momentum checks; not GPU or full-grid certification"}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "pass", "checks": len(records)}), flush=True)


if __name__ == "__main__":
    main()
