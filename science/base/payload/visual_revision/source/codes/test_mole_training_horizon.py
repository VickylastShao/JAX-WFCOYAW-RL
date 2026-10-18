"""Regression cases for whole-horizon and resume exhaustion failures."""
from mole_training_horizon import plane_horizon


def main():
    common = dict(completed_planes=2700000, iterations=3907, collection_steps=8,
                  global_environments=32, burnin_interactions=150,
                  episode_interactions=500, reset_interactions=150)
    cursors = list(range(20000, 640001, 20000))
    result = plane_horizon(cursors=cursors, planned_global_interactions=1000448, **common)
    assert result['global_interactions_end'] == 1000448
    assert result['resets_per_environment'] == 62
    assert result['required_archive_end_exclusive'] == 2675700
    assert result['planes_consumed_per_environment'] == 2035700
    for override in ({'iterations': 3908}, {'completed_planes': 2675699}):
        try:
            plane_horizon(cursors=cursors, planned_global_interactions=1000448, **{**common, **override})
        except ValueError:
            pass
        else:
            raise AssertionError('incorrect budget or one-plane short archive accepted')
    # Reset exactly at an episode endpoint is counted, also across resume.
    edge = plane_horizon(cursors=[100], completed_planes=100000, iterations=1,
                         collection_steps=8, global_environments=32,
                         burnin_interactions=150, episode_interactions=500,
                         reset_interactions=150, resumed_global_interactions=492*32,
                         warmup_cursors=[100])
    assert edge['resets_per_environment'] == 1
    assert edge['initial_burnin_interactions'] == 0
    assert edge['planes_consumed_per_environment'] == (8+150)*50
    # Historical pattern: a single block fits but the whole continuation does not.
    try:
        plane_horizon(cursors=[2348200], completed_planes=2355700, iterations=1954,
                      collection_steps=16, global_environments=32,
                      burnin_interactions=0, episode_interactions=500,
                      reset_interactions=150, resumed_global_interactions=1000448,
                      warmup_cursors=[20000])
    except ValueError:
        pass
    else:
        raise AssertionError('historical exhausted continuation accepted')
    # A discarded compilation read matters even when the retained trajectory fits.
    try:
        plane_horizon(cursors=[100], completed_planes=10000, iterations=1,
                      collection_steps=8, global_environments=32,
                      burnin_interactions=0, episode_interactions=500,
                      reset_interactions=150, resumed_global_interactions=100*32,
                      warmup_cursors=[9900])
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range discarded compile read accepted')
    print('mole whole-input-horizon regression: PASS (6 cases)')


if __name__ == '__main__':
    main()
