"""Trace-level generation and actuation accounting; no independence claims."""
import numpy as np


def score_trace(metrics, boundary_yaw, *, control_dt=10., max_angle=40., max_rate=1.):
    power = np.asarray(metrics['farm_power_mw'], dtype=np.float64)
    turbines = np.asarray(metrics['turbine_power_mw'], dtype=np.float64)
    yaw = np.asarray(metrics['yaw_deg'], dtype=np.float64)
    boundary = np.asarray(boundary_yaw, dtype=np.float64)
    if power.ndim != 2 or power.shape[0] < 1 or yaw.shape[:2] != power.shape or turbines.shape != yaw.shape:
        raise ValueError('expected [T,B] farm power and [T,B,N] turbine power/yaw')
    if boundary.shape != yaw.shape[1:] or min(control_dt, max_angle, max_rate) <= 0:
        raise ValueError('invalid boundary or actuation limits')
    if not all(np.isfinite(x).all() for x in (power, turbines, yaw, boundary)):
        raise ValueError('nonfinite evaluation trace')
    if not np.allclose(power, turbines.sum(axis=-1), rtol=2e-6, atol=1e-5):
        raise ValueError('farm and individual-turbine power disagree beyond float32 summation')
    differences = np.diff(np.concatenate((boundary[None], yaw), axis=0), axis=0)
    peak_rate = float(np.max(np.abs(differences)) / control_dt)
    angle = float(np.max(np.abs(yaw)))
    return {'controls': int(power.shape[0]), 'duration_seconds': power.shape[0] * control_dt,
            'generation_energy_mwh': float(power.sum() * control_dt / 3600),
            'mean_farm_power_mw': float(power.mean()),
            'turbine_generation_energy_mwh': (turbines.sum(axis=(0, 1)) * control_dt / 3600).tolist(),
            'yaw_travel_deg_per_turbine': np.abs(differences).sum(axis=(0, 1)).tolist(),
            'peak_yaw_rate_deg_s': peak_rate, 'maximum_abs_yaw_deg': angle,
            'rate_limit_satisfied': peak_rate <= max_rate + 1e-6,
            'angle_limit_satisfied': angle <= max_angle + 1e-5,
            'energy_definition': 'gross generated electricity; actuator energy not modeled'}


def paired_gain(rl, zero):
    if rl['controls'] != zero['controls'] or rl['duration_seconds'] != zero['duration_seconds']:
        raise ValueError('paired scored durations differ')
    baseline = float(zero['generation_energy_mwh'])
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('baseline generation must be finite and positive')
    delta = float(rl['generation_energy_mwh']) - baseline
    return {'absolute_generation_gain_mwh': delta,
            'generation_gain_percent': 100 * delta / baseline,
            'baseline_generation_mwh': baseline}
