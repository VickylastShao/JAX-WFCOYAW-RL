#!/usr/bin/env python3
"""Deliberate copies/tampering of the small export; original data stays read-only."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest

from validate_resource_export import validate

HERE=Path(__file__).resolve().parent


class ExportValidationTests(unittest.TestCase):
    def setUp(self):
        original=HERE.parent/'remote_export/SMALL_EXPORT'
        if not original.exists():self.skipTest('Returned small export is not available beside this script')
        self.tmp=tempfile.TemporaryDirectory(prefix='deliberate_export_tamper_')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'SMALL_EXPORT'
        shutil.copytree(original,self.root)

    def rebind_modified_stats(self):
        path=self.root/'INDEPENDENT_ARRAY_CHECK.json'
        manifest=json.loads((self.root/'SMALL_EXPORT_MANIFEST.json').read_text())
        for row in manifest:
            if row['path']=='INDEPENDENT_ARRAY_CHECK.json':
                row.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (self.root/'SMALL_EXPORT_MANIFEST.json').write_text(json.dumps(manifest))

    def test_complete_small_export_passes(self):
        result=validate(self.root)
        self.assertEqual(result['status'],'pass')
        self.assertTrue(result['locally_recomputed_full_returned_power_and_probe_differences'])
        self.assertFalse(result['locally_recomputed_complete_field_differences'])

    def test_wrong_norm_is_rejected_even_if_file_manifest_is_rebound(self):
        path=self.root/'INDEPENDENT_ARRAY_CHECK.json'
        value=json.loads(path.read_text());row=next(r for r in value['arrays'] if r['branch']=='stream_g3_b1' and r['field']=='state_velocity')
        row['sum_squared_difference']*=4
        path.write_text(json.dumps(value));self.rebind_modified_stats()
        result=validate(self.root)
        self.assertTrue(result['checks']['export_manifest_all_file_SHA_and_sizes'])
        self.assertFalse(result['checks']['all_norm_arithmetic_matches'])
        self.assertEqual(result['status'],'fail')

    def test_missing_field_is_rejected_even_if_file_manifest_is_rebound(self):
        path=self.root/'INDEPENDENT_ARRAY_CHECK.json'
        value=json.loads(path.read_text())
        value['arrays']=[r for r in value['arrays'] if not(r['branch']=='stream_g3_b1' and r['field']=='state_velocity')]
        path.write_text(json.dumps(value));self.rebind_modified_stats()
        # Missing reference-row metadata must fail, never turn into a complete label.
        try:result=validate(self.root)
        except KeyError:return
        self.assertFalse(result['checks']['expected_branch_ten_field_coverage'])
        self.assertEqual(result['status'],'fail')


if __name__=='__main__':unittest.main()
