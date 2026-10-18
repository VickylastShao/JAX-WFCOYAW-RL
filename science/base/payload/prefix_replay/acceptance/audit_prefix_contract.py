#!/usr/bin/env python3
"""Independent CPU-only audit of immutable prefix reference and driver scope."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'paper1_mainline_revision_20261015'
PACK = Path('/mnt/d/research-backups/JAX-WFCOYAW-RL/github_release_ready_20261014/repository/n81_20261014_visual_revision')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()

def main():
    out = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope': 'Independent CPU reference and source-contract audit; no network, GPU, or manuscript changes',
           'branch_references': {}, 'source_identities': {}}
    original_result_path = PACK / 'data/BC/B_w1/RESULT.json'
    original_result = json.loads(original_result_path.read_text())
    assert original_result['context_id'] == 'n81_B_w1'
    assert original_result['common_state_sha256'] == '1db258cbe44cfe3eedcca2ad82c7965baba35b7f280be520dc74128d838a58be'
    assert original_result['common_burnin']['final_farm_step_index'] == 7500
    assert original_result['common_burnin']['final_plane_cursor_exclusive'] == 27500
    assert original_result['evaluation_numerics'] == {'compact_solver': 'dense_inverse', 'rhs_assembly': 'tensor'}
    out['original_evaluation_result_sha256'] = sha(original_result_path)
    energies = {}
    for branch in ('zero', 'rl_r1'):
        original = PACK / f'data/BC/B_w1/{branch}.npz'
        reference = OLD / f'REFERENCE_B_W1_PREFIX8_{branch}.npz'
        with np.load(original, allow_pickle=False) as original_npz, np.load(reference, allow_pickle=False) as ref:
            fields = {}
            assert set(original_npz.files) == set(ref.files)
            for field in ref.files:
                a, b = original_npz[field][:8], ref[field]
                exact = a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')
                assert exact and np.isfinite(b).all(), field
                fields[field] = {'shape': list(b.shape), 'dtype': str(b.dtype), 'first8_original_bytes_equal': exact}
            energy = float(np.sum(ref['farm_power_mw'], dtype=np.float64) * 10 / 3600)
            energies[branch] = energy
        original_hash = sha(original)
        assert original_hash == original_result['controllers'][branch]['trace_sha256']
        assert original_result['controllers'][branch]['common_state_sha256'] == original_result['common_state_sha256']
        assert original_result['controllers'][branch]['scoring_cursor'] == 27500
        out['branch_references'][branch] = {'reference_sha256': sha(reference),
            'original_full_trace_sha256': original_hash, 'fields': fields, 'energy_MWh': energy}
    out['paired_prefix_gain_percent'] = (energies['rl_r1'] / energies['zero'] - 1) * 100
    out['field_arrays_total'] = sum(len(v['fields']) for v in out['branch_references'].values())
    for name in ('mole_large_scale_evaluation.py', 'eval_mole_large_scale_paired.py', 'jax_mole_continuous_sac.py'):
        source = PACK / 'source/codes' / name
        out['source_identities'][name] = sha(source)
    out['contract_findings'] = [
        'New worker uses the original canonical 81-turbine layout and identical dense_inverse/tensor case.',
        'ContinuousControlConfig(time_planes) and original make_step/make_scan agree with the frozen B_w1 evaluator.',
        'Exact saved observation/history/previous-yaw cache is consumed, with no new reset or observation reconstruction.',
        'Local farm-step origin 7500 maps to inlet prefix global 27500; after 8 controls each branch ends at farm step 7900, interaction index 8, global inlet endpoint 27900.',
        'Output acceptance must compare all 15 arrays (7 zero plus 8 RL) byte-for-byte, including signed zero; negative transient gain is retained.',
        'Only output-prefix bitwise reproduction can be claimed; all seven final flow-history arrays lack an independently archived 8-control final-state reference.',
        'Driver CPU preflight occurs before its own worker timing; outer GPU container lifecycle must independently enforce and record the 900-second maximum including startup and cleanup.',
    ]
    out['prelaunch_scientific_blockers_identified'] = []
    out['unclosed_before_gpu_result'] = ['Actual authorized original-image GPU replay and exit/output receipts',
        'Full lifecycle allocated GPU-time accounting no greater than 0.25 GPUh',
        'Native runtime source/image and driver identity verification on the execution host']
    path = HERE / 'PRELAUNCH_ACCEPTANCE_AUDIT.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': 'CPU_reference_audit_pass', 'field_arrays_total': out['field_arrays_total'],
        'paired_prefix_gain_percent': out['paired_prefix_gain_percent'], 'report': str(path)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
