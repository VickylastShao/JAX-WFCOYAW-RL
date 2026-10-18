#!/usr/bin/env python3
"""Validate the disclosed executed roster; no LES or model loading."""
from pathlib import Path
import argparse, json, hashlib
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads((ROOT/'ROSTER_CHANGE.json').read_text())
pr=json.loads((ROOT/'raw/evaluation/wp6_controller_preregistration.final.json').read_text())
assert len(pr['controller_roster'])==d['original_wp6_preregistered_roles']==10
formal={x['controller_id'] for x in pr['controller_roster'] if not x['evidence_role'].startswith('descriptive')}
historical={x['controller_id'] for x in pr['controller_roster'] if x['evidence_role'].startswith('descriptive')}
assert len(formal)==d['executed_wp6_wp7_controllers']==7
assert historical==set(d['historical_controllers'])
for row in d['raw_historical_sources']:
 q=ROOT/row['included_path'];assert q.is_file() and q.stat().st_size==row['bytes']
 assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
for row in d['historical_controllers'].values():
 assert row['wp6_wp7_omission_reason']=='unknown'
 assert row['independently_verified_omission_time']=='unknown'
out={'status':'PASS_RECORD_DISCLOSURE','preregistered_roles':10,'executed_controllers':7,'historical_development_inputs_SHA_verified':4,'omission_reason_and_chronology':'unresolved; not validated by this check','new_GPUh':0}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
