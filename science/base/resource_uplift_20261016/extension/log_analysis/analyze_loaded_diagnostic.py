#!/usr/bin/env python3
"""Separate loaded-prefix numerical audit; never a resource-model calibration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from validate_resource_export import validate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root,out):
    campaign=root/'gpu_loaded_check'
    proto=json.loads((campaign/'PROTOCOL.json').read_text())
    terminal=json.loads((campaign/'EXIT.json').read_text())
    gen=json.loads((root/'inputs_loaded/GENERATION_RESULT.json').read_text())
    dense=json.loads((campaign/'dense_b1/RESULT.json').read_text())
    stream=json.loads((campaign/'stream_g9_b1/RESULT.json').read_text())
    fields=[]
    for name,value in stream['comparison'].items():
        fields.append(dict(field=name,relative_L2=value['relative_l2'],maximum_absolute_difference=value['max_abs'],
            allclose_rtol1e4_atol1e5=value['allclose_rtol1e-4_atol1e-5'],bitwise_equal=value['bitwise_equal'],
            finite=value['finite'],shape=value['shape']))
    gates={
      'terminal_completed':terminal['status']=='completed',
      'generation_receipt_SHA_bound':sha(root/'inputs_loaded/GENERATION_RESULT.json')==proto['input_generation_receipt_sha256'],
      'recorded_generator_script_SHA':sha(root/'gen_loaded_inputs.py')==proto['generator_sha256'],
      'both_runner_core_SHA_match':all(r['core_sha256']==proto['source_core_sha256'] and r['runner_sha256']==proto['runner_sha256'] for r in [dense,stream]),
      'common_state_hashes_match':all(r['inputs']['common_state']['sha256']==proto['state_sha256'] for r in [dense,stream]),
      'held_inlet_hashes_match':all(r['inputs']['inlet_prefix']['sha256']==proto['inlet_sha256'] for r in [dense,stream]),
      'all_ten_comparison_fields_present':len(fields)==10,
      'all_comparison_fields_finite':all(r['finite'] for r in fields),
      'velocity_norm_within_frozen_stop':stream['comparison']['state_velocity']['relative_l2']<proto['numerical_stop']['velocity_relative_l2'],
      'power_norm_within_frozen_stop':stream['comparison']['power_trace']['relative_l2']<proto['numerical_stop']['power_relative_l2'],
    }
    export=validate(root,expected_branch_count=2)
    result=dict(schema='loaded-prefix-diagnostic/1',status='pass_scoped_norm_checks' if all(gates.values()) and export['status']=='pass' else 'failed_scoped_diagnostic',
       checks=gates,nominal_configured_hub_speed_mps=gen['nominal_configured_hub_speed_mps'],
       resolved_hub_y_m=gen['resolved_hub_y_m'],resolved_hub_streamwise_statistics=gen['resolved_hub_streamwise_statistics'],
       initializer_override=proto['initializer_override'],all_ten_fields=fields,
       pointwise_allclose_for_every_field=all(r['allclose_rtol1e4_atol1e5'] for r in fields),
       bitwise_equal_every_field=all(r['bitwise_equal'] for r in fields),
       formal_full_state_equivalence_established=False,
       additional_gpu_seconds=terminal['gpu_hours']*3600,additional_gpu_hours=terminal['gpu_hours'],
       timing_not_used_in_resource_model=True,physical_case_not_mature_turbulence=True,
       scope=proto['scope'],returned_export_validation=export,
       source_binding={str(p.relative_to(root)):sha(p) for p in [campaign/'PROTOCOL.json',campaign/'EXIT.json',root/'inputs_loaded/GENERATION_RESULT.json']},
       warning='Velocity pointwise and RHS/pressure field checks need not pass despite velocity/power aggregate norm gates. All per-field failures remain explicit. This is a 10s implementation diagnostic, not predictive-physics validation or learned-policy equivalence.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'LOADED_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'LOADED_ALL_TEN_FIELDS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(fields[0]));writer.writeheader();writer.writerows(fields)
    (out/'LOADED_RETURNED_EXPORT_VALIDATION.json').write_text(json.dumps(export,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=analyze(args.root,args.output)
    print(json.dumps(dict(status=r['status'],pointwise_allclose_for_every_field=r['pointwise_allclose_for_every_field'],
       bitwise_equal_every_field=r['bitwise_equal_every_field'],additional_gpu_hours=r['additional_gpu_hours'],
       actual_hub_mean=r['resolved_hub_streamwise_statistics']['mean'],checks=r['checks'])))


if __name__=='__main__':main()
