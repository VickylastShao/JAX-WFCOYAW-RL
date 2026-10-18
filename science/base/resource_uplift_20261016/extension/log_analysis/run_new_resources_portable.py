#!/usr/bin/env python3
"""Replay actual main-campaign arithmetic and returned small-data checks.

Only standard-library Python is used; no GPU, NumPy, network or server access.
The loaded-flow follow-up has a separate cost/validation layer and is never
pooled into the original fixed-fixture timing/model fit.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=here.parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.is_relative_to(root):raise ValueError('Keep outputs outside the frozen extension')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh/empty output directory required')
    out.mkdir(parents=True,exist_ok=True)
    analysis=module(root/'log_analysis/analyze_new_campaign.py','resource_campaign_analysis')
    exporter=module(root/'log_analysis/validate_resource_export.py','resource_export_validation')
    sys.path.insert(0,str(root/'log_analysis'))
    pilots=[root/name for name in ['gpu_pilot','gpu_pilot_retry1','gpu_pilot_retry2','gpu_remote_pilot']]
    if not all((path/'EXIT.json').exists() for path in pilots):raise FileNotFoundError('All four prior pilot receipts must be included')
    cost_only=[root/'gpu_loaded_check'] if (root/'gpu_loaded_check/EXIT.json').exists() else []
    result=analysis.analyze(root/'gpu_main',pilots,out,root/'source',root/'inputs/GENERATION_RESULT.json',cost_only)
    validation=exporter.validate(root/'remote_export/SMALL_EXPORT')
    (out/'RETURNED_EXPORT_VALIDATION.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    loaded=None
    if cost_only:
        loaded_module=module(root/'log_analysis/analyze_loaded_diagnostic.py','loaded_resource_diagnostic')
        loaded=loaded_module.analyze(root/'loaded_export/LOADED_SMALL_EXPORT',out)
    print(json.dumps(dict(main_campaign_status=result['status'],returned_export_status=validation['status'],
        total_including_pilots_and_loaded_followup_GPUh=result['budget']['accounted_gpu_hours'],
        loaded_diagnostic_status=loaded['status'] if loaded else None,
        loaded_followup_timings_not_pooled=True,no_GPU_or_network=True)))
    if validation['status']!='pass':raise SystemExit(1)
    if loaded and loaded['status']!='pass_scoped_norm_checks':raise SystemExit(1)


if __name__=='__main__':main()
