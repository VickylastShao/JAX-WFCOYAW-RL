#!/usr/bin/env python3
"""Bounded tamper tests on temporary copies; archived inputs remain untouched."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path);args=ap.parse_args()
    outcomes=[]
    with tempfile.TemporaryDirectory(prefix='n9-visual-negative-') as temp:
        for case in ['changed_derived_cell','changed_raw_power_with_rebound_file_SHA','incomplete_window_matrix_with_rebound_file_SHA']:
            root=Path(temp)/case;root.mkdir();shutil.copytree(ROOT/'data',root/'data')
            shutil.copy2(ROOT/'SOURCE_IDENTITIES.json',root/'SOURCE_IDENTITIES.json')
            if case=='changed_derived_cell':
                p=root/'data/N9_VISUAL_DATA.json';d=json.loads(p.read_text());d['controllers'][4]['direction_mean_gain_percent'][0]+=.2
                p.write_text(json.dumps(d))
            else:
                p=root/'data/wp7_combined_summary.json';d=json.loads(p.read_text())
                if case.startswith('changed_raw_power'):
                    next(r for r in d['window_rows'] if r['controller_id']=='selected_realtime_policy')['mean_farm_power_mw']+=.02
                else:d['window_rows'].pop()
                p.write_text(json.dumps(d));ip=root/'SOURCE_IDENTITIES.json';identities=json.loads(ip.read_text())
                source=next(x for x in identities['sources'] if x['relative_path']=='data/wp7_combined_summary.json')
                source['bytes']=p.stat().st_size;source['source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
                ip.write_text(json.dumps(identities))
            result=subprocess.run([sys.executable,str(ROOT/'validate_n9_visual_data.py'),'--root',str(root)],capture_output=True,text=True)
            if result.returncode==0:raise AssertionError('tamper unexpectedly accepted: '+case)
            outcomes.append({'case':case,'expected':'FAIL','observed_exit_code':result.returncode,
                             'assertion':result.stderr.splitlines()[-1]})
    report={'status':'PASS','expected_failures':len(outcomes),'cases':outcomes,
            'original_inputs_modified':False,'new_GPUh':0}
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','expected_failures':len(outcomes),'new_GPUh':0}))
if __name__=='__main__':main()
