"""Small CPU output-identity negative controls; no physical transfer executed."""
import unittest
import numpy as np
from run_saved_state_prefix import compare_trace


class OriginalOutputIdentityTests(unittest.TestCase):
    def test_exact_values_pass(self):
        a={'yaw_deg':np.zeros((8,1,81),dtype=np.float32)}
        self.assertTrue(compare_trace(a,a)['all_fields_bitwise_equal'])

    def test_one_float32_ulp_mismatch_is_not_hidden(self):
        a={'farm_power_mw':np.ones((8,1),dtype=np.float32)}
        b={'farm_power_mw':a['farm_power_mw'].copy()}
        b['farm_power_mw'][0,0]=np.nextafter(np.float32(1),np.float32(2))
        c=compare_trace(a,b)
        self.assertFalse(c['all_fields_bitwise_equal'])
        self.assertGreater(c['fields']['farm_power_mw']['max_abs_difference'],0)

    def test_signed_zero_is_distinguished_for_bitwise_claim(self):
        a={'yaw_deg':np.zeros((8,1,81),dtype=np.float32)}
        b={'yaw_deg':-a['yaw_deg']}
        c=compare_trace(a,b)
        self.assertTrue(c['fields']['yaw_deg']['numerically_equal'])
        self.assertFalse(c['all_fields_bitwise_equal'])

    def test_missing_fields_fail(self):
        with self.assertRaises(ValueError):compare_trace({}, {'yaw_deg':np.zeros((8,1,81),dtype=np.float32)})

    def test_wrong_shape_dtype_and_nonfinite_fail(self):
        ref={'farm_power_mw':np.ones((8,1),dtype=np.float32)}
        for bad in [np.ones((7,1),dtype=np.float32),np.ones((8,1),dtype=np.float64),np.full((8,1),np.nan,dtype=np.float32)]:
            with self.assertRaises(ValueError):compare_trace({'farm_power_mw':bad},ref)


if __name__=='__main__':unittest.main()
