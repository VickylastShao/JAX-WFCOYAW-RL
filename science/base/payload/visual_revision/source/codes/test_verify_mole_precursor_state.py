"""Small corruption cases for the full-field precursor verifier."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from verify_mole_precursor_state import digest, verify


class SavedStateTest(unittest.TestCase):
    def fixture(self, directory, nan_rhs=False):
        path = Path(directory) / 'state.npz'
        case = {'batch': 1, 'nx': 4, 'ny': 3, 'nz': 3,
                'domain_ly': 2., 'hub_height': 1.}
        velocity = np.ones((1, 3, 4, 3, 3), dtype=np.float32)
        rhs = np.zeros_like(velocity)
        if nan_rhs:
            rhs[0, 0, 0, 0, 0] = np.nan
        np.savez(path, velocity=velocity, rhs_previous=rhs,
                 rhs_previous_2=np.zeros_like(velocity),
                 filtered_disk_speed=np.zeros((1, 3), dtype=np.float32),
                 bottom_pressure_gradient=np.zeros((1, 2, 4, 3), dtype=np.float32),
                 open_x_pressure_gradient=np.zeros((1, 2, 2, 3, 3), dtype=np.float32),
                 step_index=np.array(10, dtype=np.int32))
        meta = {'seed': 7, 'case': case, 'completed_steps': 10,
                'npz_sha256': digest(path), 'provenance': {'layout': 'fixture'}}
        path.with_suffix('.meta.json').write_text(json.dumps(meta))
        spec = {'source_id': 'fixture', 'seed': 7, 'case': case,
                'saved_steps': 10, 'expected_sha256': meta['npz_sha256'],
                'provenance': meta['provenance']}
        return path, spec

    def test_valid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.fixture(directory)
            self.assertEqual(verify(path, spec)['status'], 'pass')

    def test_nonfinite_rhs_is_not_hidden_by_finite_velocity(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.fixture(directory, nan_rhs=True)
            result = verify(path, spec)
            self.assertEqual(result['status'], 'fail')
            self.assertFalse(result['checks']['finite_rhs_previous'])

    def test_expected_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.fixture(directory)
            spec['expected_sha256'] = '0' * 64
            self.assertEqual(verify(path, spec)['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
