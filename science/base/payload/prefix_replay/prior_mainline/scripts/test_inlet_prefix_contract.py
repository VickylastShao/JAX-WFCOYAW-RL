"""Small CPU negative controls; fixtures are not scientific inlet data."""
import unittest
import tempfile
import json
from pathlib import Path
from inlet_prefix_contract import select_chunks, prefix_plan, extract_verified_chunk, sha256


class PrefixContractTests(unittest.TestCase):
    def setUp(self):
        self.m = {'dtype':'float32', 'completed_planes':40,
                  'plane_shape':[3,2,2], 'bytes_per_plane':48,
                  'chunks':[{'start':0, 'stop':20, 'sha256':'a'*64},
                            {'start':20, 'stop':40, 'sha256':'b'*64}]}

    def test_cross_chunk_and_half_open_endpoint(self):
        self.assertEqual(len(select_chunks(self.m, 18, 22)), 2)
        self.assertEqual(len(select_chunks(self.m, 20, 40)), 1)

    def test_rejects_gap(self):
        self.m['chunks'][1]['start']=21
        with self.assertRaises(ValueError): select_chunks(self.m, 18, 22)

    def test_rejects_missing_hash(self):
        self.m['chunks'][1]['sha256']='unknown'
        with self.assertRaises(ValueError): select_chunks(self.m, 20, 30)

    def test_rejects_overlap(self):
        self.m['chunks'][1]['start']=19
        with self.assertRaises(ValueError): select_chunks(self.m, 18, 22)

    def test_rejects_inconsistent_shape(self):
        self.m['bytes_per_plane']=24
        with self.assertRaises(ValueError): select_chunks(self.m, 20, 30)

    def test_rejects_out_of_range(self):
        for lo, hi in [(-1, 8), (20,20), (0,41)]:
            with self.assertRaises(ValueError): select_chunks(self.m, lo, hi)


class IdentityExtractionTests(unittest.TestCase):
    def test_full_chunk_identity_and_extract_negative_control(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk = root / 'synthetic_fixture.npy'
            np.save(chunk, np.zeros((50,3,41,483),dtype=np.float32))
            source = {'completed_steps':200000,'seed':1,'sha256':'a'*64}
            m = {'dtype':'float32','completed_planes':50,'plane_shape':[3,41,483],
                 'bytes_per_plane':237636,'source_state':source,
                 'chunks':[{'start':0,'stop':50,'sha256':sha256(chunk),'bytes':chunk.stat().st_size}]}
            mp, rp = root/'manifest.json', root/'RESULT.json'
            mp.write_text(json.dumps(m))
            r = {'archive_manifest_sha256':sha256(mp),'source_state':source,
                 'context_id':'synthetic_non_LES_fixture','common_state_sha256':'b'*64,
                 'common_burnin':{'controls':0,'final_plane_cursor_exclusive':0,'final_farm_step_index':0},
                 'controllers':{'zero':{'scoring_cursor':0,'common_state_sha256':'b'*64},
                                'rl':{'scoring_cursor':0,'common_state_sha256':'b'*64}}}
            rp.write_text(json.dumps(r))
            output = root/'prefix.npy'
            result = extract_verified_chunk(mp,rp,1,chunk,output)
            self.assertEqual(result['payload_bytes'],11881800)
            self.assertEqual(np.load(output,allow_pickle=False).shape,(50,3,41,483))
            with chunk.open('r+b') as f:
                f.seek(-1,2); f.write(b'x')
            with self.assertRaises(ValueError):
                extract_verified_chunk(mp,rp,1,chunk,root/'must_not_exist.npy')
            self.assertFalse((root/'must_not_exist.npy').exists())
            r['controllers']['rl']['scoring_cursor']=1
            rp.write_text(json.dumps(r))
            with self.assertRaises(ValueError): prefix_plan(mp,rp,1)


if __name__ == '__main__': unittest.main()
