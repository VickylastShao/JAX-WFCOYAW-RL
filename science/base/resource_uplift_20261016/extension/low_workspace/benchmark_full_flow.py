#!/usr/bin/env python3
"""Bounded full LES branch runner. Run one branch per process for GPU memory.

This is an execution/equivalence diagnostic, not policy training, long-window
energy validation, cross-grid convergence, or a new wind-farm environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def compare(a, b, np):
    a, b = np.asarray(a), np.asarray(b)
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(np.asarray(b, dtype=np.float64).reshape(-1)))
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / max(norm, 1e-30)),
            "bitwise_equal": bool(a.dtype == b.dtype and np.array_equal(a, b)),
            "allclose_rtol1e-4_atol1e-5": bool(np.allclose(a, b, rtol=1e-4, atol=1e-5))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["dense", "stream"], required=True)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--grid", choices=["mini", "n9", "n81"], default="mini")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--common-state", type=Path)
    p.add_argument("--tile-common-state", action="store_true",
                   help="replicate one frozen history across batch for execution only")
    p.add_argument("--inlet-prefix", type=Path)
    p.add_argument("--yaw-npy", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--source-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.substeps <= 400 or not 1 <= args.repeat <= 5:
        raise ValueError("substeps in [1,400] and repeats in [1,5] required")
    if args.batch < 1 or args.batch > 4:
        raise ValueError("batch in [1,4] required")
    if args.output.exists():
        raise FileExistsError("refuse to overwrite an existing branch")
    args.output.mkdir(parents=True)
    save(args.output / "STATUS.json", {"phase": "import", "time": time.time()})
    sys.path.insert(0, str(args.source_dir.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jax_mole_compact_sac_benchmark as core
    from jax_mole_layouts import rectangular_mole_layout
    from streaming_adm import build_streaming_flow_functions
    layout = rectangular_mole_layout(3, 1) if args.grid == "mini" else rectangular_mole_layout(3, 3) if args.grid == "n9" else rectangular_mole_layout(9, 9)
    case = core.make_scaled_mole_training_case(layout, args.batch)
    if args.grid == "mini":
        case = replace(case, nx=65, ny=9, nz=72, initialize_with_log_profile=True,
                       precursor_initial_ti=0.01)
    if args.common_state is not None and args.batch != 1 and not args.tile_common_state:
        raise ValueError("batch > 1 from one common history requires explicit --tile-common-state")
    started_setup = time.perf_counter()
    functions = (core.build_flow_functions(case, False, True) if args.mode == "dense" else
                 build_streaming_flow_functions(core, case, group_size=args.group))
    provenance = functions.get("streaming_adm_provenance", {})
    if args.common_state:
        opener = gzip.open if args.common_state.suffix == ".gz" else open
        with opener(args.common_state, "rb") as stream:
            common = pickle.load(stream)
        initial = common.farm_flow if hasattr(common, "farm_flow") else common
        if not isinstance(initial, core.FlowState):
            raise TypeError("expected frozen FlowState or common.farm_flow")
        initial = jax.tree.map(jnp.asarray, initial)
        if args.tile_common_state:
            initial = jax.tree.map(lambda value: jnp.repeat(value, args.batch, axis=0)
                                   if value.ndim > 0 else value, initial)
    else:
        initial = core.make_initial_flow(case, functions, 81731)
    if initial.velocity.shape != (args.batch, 3, case.nx, case.ny, case.nz):
        raise ValueError("input state/grid mismatch")
    if args.inlet_prefix:
        prefix = np.load(args.inlet_prefix, allow_pickle=False)
        if prefix.ndim != 4 or prefix.shape[1:] != (3, case.ny, case.nz) or len(prefix) < args.substeps:
            raise ValueError("inlet prefix must contain [steps,3,ny,nz] at original cadence")
        inlet = jnp.asarray(np.broadcast_to(prefix[:args.substeps, None], (args.substeps, args.batch, 3, case.ny, case.nz)))
    else:
        inlet = jnp.broadcast_to(initial.velocity[:, :, 0], (args.substeps, args.batch, 3, case.ny, case.nz))
    yaw = (np.load(args.yaw_npy, allow_pickle=False) if args.yaw_npy else
           np.broadcast_to(np.linspace(-20., 20., case.num_turbines, dtype=np.float32), (args.batch, case.num_turbines)).copy())
    yaw = jnp.asarray(yaw, dtype=initial.velocity.dtype)
    if yaw.shape != (args.batch, case.num_turbines):
        raise ValueError("yaw shape must match [batch,turbines]")
    weight_started = time.perf_counter()
    save(args.output / "STATUS.json", {"phase": "compile", "operation": "weights_setup", "time": time.time()})
    weights_compile_started = time.perf_counter()
    weights_executable = jax.jit(functions["disk_weights"]).lower(yaw).compile()
    weights_setup_compile_seconds = time.perf_counter() - weights_compile_started
    weights_analysis = weights_executable.memory_analysis()
    weights_setup_memory = ({key: int(getattr(weights_analysis, key)) for key in dir(weights_analysis)
                             if key.endswith("_in_bytes") and isinstance(getattr(weights_analysis, key), int)}
                            if weights_analysis is not None else None)
    save(args.output / "STATUS.json", {"phase": "weights_setup_run", "time": time.time(),
                                       "weights_setup_compile_seconds": weights_setup_compile_seconds})
    weights_execution_started = time.perf_counter()
    weights = weights_executable(yaw)
    jax.block_until_ready(weights)
    weights_setup_execution_seconds = time.perf_counter() - weights_execution_started
    weights_setup_seconds = time.perf_counter() - weight_started
    if args.source_only:
        def operation(state, yaw, weights, inlet):
            return functions["disk_source"](state.velocity, yaw, weights,
                                             state.filtered_disk_speed, state.step_index == 0)
    else:
        def operation(state, yaw, weights, inlet):
            def body(flow, plane):
                advanced, power = functions["advance"](flow, yaw, weights, plane)
                return advanced, power
            return jax.lax.scan(body, state, inlet)
    jax.block_until_ready((initial, inlet, yaw))
    setup_seconds = time.perf_counter() - started_setup
    memory_stats_before_compile = jax.devices()[0].memory_stats()
    save(args.output / "STATUS.json", {"phase": "compile", "time": time.time()})
    started = time.perf_counter()
    compiled = jax.jit(operation).lower(initial, yaw, weights, inlet).compile()
    compile_seconds = time.perf_counter() - started
    analysis = compiled.memory_analysis()
    memory = ({key: int(getattr(analysis, key)) for key in dir(analysis)
               if key.endswith("_in_bytes") and isinstance(getattr(analysis, key), int)}
              if analysis is not None else None)
    save(args.output / "STATUS.json", {"phase": "run", "time": time.time(),
                                       "compile_seconds": compile_seconds})
    times = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        result = compiled(initial, yaw, weights, inlet)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - started)
    memory_stats_after_execution = jax.devices()[0].memory_stats()
    if args.source_only:
        values = {key: np.asarray(value) for key, value in zip(
            ["acceleration", "filtered_disk_speed", "power", "thrust"], result, strict=True)}
    else:
        final, power = result
        values = {"state_" + key: np.asarray(value) for key, value in zip(final._fields, final, strict=True)}
        values["power_trace"] = np.asarray(power)
        values["final_probes"] = np.asarray(functions["probes"](final.velocity))
        values["final_divergence"] = np.asarray(functions["diagnostic_divergence"](functions["divergence"](final.velocity)))
    finite = {key: bool(np.isfinite(value).all()) for key, value in values.items()}
    if not all(finite.values()):
        raise FloatingPointError("nonfinite output; no resource claim accepted")
    np.savez(args.output / "OUTPUT.npz", **values)
    comparison = None
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            if set(reference.files) != set(values):
                raise ValueError("reference keys mismatch")
            comparison = {key: compare(values[key], reference[key], np) for key in values}
    identity = lambda path: None if path is None else {"path": str(path), "sha256": sha(path)}
    summary = {
        "schema": "paper1_full_support_streaming_full_flow_v1", "status": "completed",
        "mode": args.mode, "group": args.group if args.mode == "stream" else case.num_turbines,
        "grid_label": args.grid, "case": asdict(case), "substeps": args.substeps,
        "source_only": args.source_only, "donation": False,
        "warmup_policy": "first execution recorded separately, repeats begin at the identical frozen state",
        "setup_seconds": setup_seconds, "weights_setup_seconds": weights_setup_seconds,
        "weights_setup_compile_seconds": weights_setup_compile_seconds,
        "weights_setup_execution_seconds": weights_setup_execution_seconds,
        "weights_setup_compiled_memory_analysis": weights_setup_memory,
        "weights_setup_implementation": "JIT fusion of the unmodified original disk_weights closure; same preparation policy in dense and streaming branches",
        "compile_seconds": compile_seconds, "run_seconds": times,
        "steady_median_seconds": float(np.median(times[1:] or times)),
        "compiled_memory_analysis": memory,
        "device_memory_stats_before_compile": memory_stats_before_compile,
        "device_memory_stats_after_execution": memory_stats_after_execution,
        "device_memory_scope": "backend allocator stats, may be None; not whole-process nvidia-smi peak",
        "persistent_input_bytes": {"flow_state": sum(int(x.nbytes) for x in jax.tree.leaves(initial)),
                                   "weights_or_yaw_token": int(weights.nbytes), "yaw": int(yaw.nbytes),
                                   "inlet_prefix": int(inlet.nbytes)},
        "finite_outputs": finite, "comparison": comparison, "provenance": provenance,
        "inputs": {"common_state": identity(args.common_state), "inlet_prefix": identity(args.inlet_prefix),
                   "yaw": identity(args.yaw_npy)},
        "source_dir": str(args.source_dir.resolve()), "core_sha256": sha(core.__file__),
        "runner_sha256": sha(__file__), "device": [str(x) for x in jax.devices()],
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "history_replication": ("one physical history repeated across batch; execution-only, not independent scientific replication"
                                if args.tile_common_state else None),
        "evidence_scope": "complete frozen LES advance with one prescribed yaw held over a bounded prefix; not trained-policy validation or time-to-quality",
    }
    save(args.output / "RESULT.json", summary)
    save(args.output / "STATUS.json", {"phase": "complete", "time": time.time()})
    print(json.dumps({"output": str(args.output), "compile_seconds": compile_seconds,
                      "run_seconds": times, "all_outputs_finite": True}), flush=True)


if __name__ == "__main__":
    main()
