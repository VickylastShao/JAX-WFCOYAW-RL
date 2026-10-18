#!/usr/bin/env python3
"""Contract test for formal N=9 Actor and full-state milestone paths."""

from pathlib import Path

from jax_mole_microbatch_distributed import _checkpoint_milestone_paths


def main() -> None:
    actor = Path("results/run/actor.pkl")
    full = Path("results/run/full.pkl")
    process0 = _checkpoint_milestone_paths(actor, full, 0, 0)
    assert process0 == {
        "actor_process0_only": Path("results/run/actor_step0.pkl"),
        "full_state_process_local": Path(
            "results/run/full_step0.process0.pkl"
        ),
    }
    process1 = _checkpoint_milestone_paths(actor, full, 250_368, 1)
    assert process1 == {
        "full_state_process_local": Path(
            "results/run/full_step250368.process1.pkl"
        )
    }
    print("mole N=9 milestone checkpoint contract: PASS")


if __name__ == "__main__":
    main()
