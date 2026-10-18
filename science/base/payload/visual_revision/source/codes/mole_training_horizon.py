"""Whole-run inlet accounting, including the trainer's real compile collection.

This module is stdlib-only and can reject an exhausted archive before JAX
initializes devices or allocates large LES/replay states.
"""
from __future__ import annotations

from collections.abc import Sequence


def plane_horizon(*, cursors: Sequence[int], completed_planes: int,
                  iterations: int, collection_steps: int,
                  global_environments: int, burnin_interactions: int,
                  episode_interactions: int, reset_interactions: int,
                  les_steps_per_interaction: int = 50,
                  resumed_global_interactions: int | None = None,
                  warmup_cursors: Sequence[int] | None = None,
                  planned_global_interactions: int | None = None) -> dict:
    """Return the exact continuous-input budget or fail before collection.

    Fresh runs retain one compilation collector block. Full-state resume
    discards that temporary block and then restores the saved cursors/state.
    The discarded compile read still needs to be inside the archive.
    """
    for name, value in (('completed_planes', completed_planes),
                        ('collection_steps', collection_steps),
                        ('global_environments', global_environments),
                        ('episode_interactions', episode_interactions),
                        ('reset_interactions', reset_interactions),
                        ('les_steps_per_interaction', les_steps_per_interaction)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if iterations < 0 or burnin_interactions < 0:
        raise ValueError('iterations and burnin must be non-negative')
    if not cursors or any(int(v) != v or v < 0 for v in cursors):
        raise ValueError('one non-negative integer cursor per local environment required')
    resumed = resumed_global_interactions is not None
    start_global = int(resumed_global_interactions or 0)
    if start_global < 0 or start_global % global_environments:
        raise ValueError('resume interactions do not match global environment cadence')
    start_control = start_global // global_environments
    retained_compile_blocks = 0 if resumed else 1
    controls = (iterations + retained_compile_blocks) * collection_steps
    resets = ((start_control + controls) // episode_interactions
              - start_control // episode_interactions)
    burnin = 0 if resumed else burnin_interactions
    consumed = (burnin + controls + resets * reset_interactions) * les_steps_per_interaction
    end_global = start_global + controls * global_environments
    if planned_global_interactions is not None and end_global != planned_global_interactions:
        raise ValueError(f'global budget mismatch: actual {end_global}, declared {planned_global_interactions}; '
                         'fresh compilation collects one retained block')
    ends = [int(cursor) + consumed for cursor in cursors]
    compile_ends = []
    if resumed:
        if warmup_cursors is None or len(warmup_cursors) != len(cursors):
            raise ValueError('full-state resume needs temporary compile cursors')
        compile_ends = [int(cursor) + collection_steps * les_steps_per_interaction
                        for cursor in warmup_cursors]
        if any(cursor < 0 for cursor in warmup_cursors):
            raise ValueError('negative compile cursor')
    required = max(ends + compile_ends)
    if required > completed_planes:
        raise ValueError(f'whole-run inlet exhaustion: need exclusive end {required}, '
                         f'archive has {completed_planes}; resets and compile reads included')
    return {'schema': 'mole_continuous_input_horizon_v1',
            'resumed': resumed, 'global_interactions_start': start_global,
            'global_interactions_end': end_global,
            'retained_compile_blocks': retained_compile_blocks,
            'controls_per_environment': controls,
            'initial_burnin_interactions': burnin,
            'resets_per_environment': resets,
            'planes_consumed_per_environment': consumed,
            'initial_cursors': list(map(int, cursors)),
            'final_cursors_exclusive': ends,
            'discarded_compile_read_ends': compile_ends,
            'required_archive_end_exclusive': required,
            'available_planes': completed_planes,
            'headroom_planes': completed_planes - required}
