#!/usr/bin/env python3
"""Verify the added scientific visual inputs without LES or a GPU."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve();v=r/'visual_revision_20261015';rows=json.loads((v/'VISUAL_INPUT_INDEX.json').read_text())['files'];checked=[]
for row in rows:
 q=v/row['path'];assert q.is_file() and q.stat().st_size==row['bytes']
 assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'];checked.append(row['path'])
d=v/'figure_sources/data_figures';pro=json.loads((d/'data/PROVENANCE.json').read_text());data_count=0
for row in pro['unchanged_files']:
 q=d/row['package_path'];assert q.is_file() and q.stat().st_size==row['bytes'];assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'];data_count+=1
f=v/'figure_sources/flow_snapshot';pr=json.loads((f/'FLOW_SNAPSHOT_PROVENANCE.json').read_text());q=f/pr['array_export']['path'];assert q.stat().st_size==pr['array_export']['bytes'];assert hashlib.sha256(q.read_bytes()).hexdigest()==pr['array_export']['sha256']
with np.load(q,allow_pickle=False) as z:
 u=z['u_xz'];assert u.dtype==np.float32 and list(u.shape)==pr['velocity_plane_profile']['shape'];assert np.isfinite(u).all();assert float(z['actual_plane_y_m'])==87.5;assert np.array_equal(z['turbine_xyz_m'],np.asarray(json.loads((d/'data/SPATIAL_DATA.json').read_text())['geometry']['turbine_positions_m']))
 for k,actual in [('min',float(u.min())),('max',float(u.max())),('mean',float(u.mean()))]:assert abs(actual-pr['velocity_plane_profile'][k])<1e-10
result={'status':'pass','new_visual_files_SHA_checked':len(checked),'parent_bound_data_files_checked':data_count,'instantaneous_flow_export_SHA_and_array_contract':'pass','new_GPUh':0,'physical_replay_executed':False,'source_state_large_SHA_rechecked_now':False,'source_state_SHA_receipt_retained':True};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
