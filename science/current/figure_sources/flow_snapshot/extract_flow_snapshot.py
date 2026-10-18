"""Extract an actual B_w1 zero-yaw scoring-start LES plane on CPU only."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np

START=time.monotonic()
ROOT=Path(sys.argv[1])
BASE=Path(sys.argv[2])
OUT=ROOT/'figure_sources/flow_snapshot'
OUT.mkdir(parents=True,exist_ok=True)
tools=BASE/'repro-pack/n81_20261015_minor_revision_repro/payload/prefix_replay/runtime/tools'
sys.path.insert(0,str(tools))
from saved_prefix_cpu_contract import trusted_structural_load


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            if time.monotonic()-START>180:raise TimeoutError('180-second flow extraction bound')
            h.update(b)
    return h.hexdigest()


def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


assert shutil.disk_usage(ROOT).free>40*2**30
write(OUT/'JOB_HANDLE.json',{'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
    'command':sys.argv,'timeout_seconds':180,'CPU_only':True,'new_GPUh':0,
    'completion_criteria':'compressed/raw state SHA and array contract; exact grid-plane export with provenance'})
index=json.loads((BASE/'repro-pack/n81_20261015_minor_revision_repro/provenance/EXTERNAL_INPUT_INDEX.json').read_text())
row=next(x for x in index['files'] if x['name']=='evaluation/n81_B_w1/common_state.pkl.gz')
# Root is the WSL-mapped D: tree; adapt only the recorded local D: path.
local=row['local_external_path']
assert local.startswith('/mnt/d/')
compressed=Path('D:/'+local[len('/mnt/d/'):])
assert compressed.stat().st_size==row['expected_compressed_bytes']
assert sha(compressed)==row['expected_compressed_sha256']
raw=OUT/'.decode_B_w1_common_state.pkl'
assert not raw.exists()
with gzip.open(compressed,'rb') as src,raw.open('xb') as dst:
    shutil.copyfileobj(src,dst,8<<20)
assert raw.stat().st_size==row['original_bytes'] and sha(raw)==row['original_sha256']
code=BASE/'repro-pack/n81_20261015_minor_revision_repro/payload/visual_revision/source/codes'
state=trusted_structural_load(raw,row['original_sha256'],code)
velocity=np.asarray(state.farm_flow.velocity)
assert velocity.shape==(1,3,496,41,483) and velocity.dtype==np.float32
assert np.isfinite(velocity).all() and int(state.farm_flow.step_index)==7500
assert np.all(np.asarray(state.previous_yaw)==0)
case=json.loads((BASE/'repro-pack/n81_20261015_minor_revision_repro/payload/visual_revision/configs/executed_physical_contract.json').read_text())['formal_farm_case']
x=np.arange(case['nx'],dtype=np.float64)*case['dx']
y=np.arange(case['ny'],dtype=np.float64)*case['dy']
z=np.arange(case['nz'],dtype=np.float64)*case['dz']
j=int(round(case['hub_height']/case['dy']))
assert y[j]==87.5
dest=OUT/'B_w1_zero_yaw_scoring_start.npz'
assert not dest.exists()
np.savez_compressed(dest,x_m=x,y_m=y,z_m=z,u_xz=velocity[0,0,:,j,:],w_xz=velocity[0,2,:,j,:],
                    u_xy_center=velocity[0,0,:,:,case['nz']//2],
                    turbine_xyz_m=np.asarray(case['configured_turbine_positions_m']),
                    actual_plane_y_m=np.array(y[j]),diameter_m=np.array(case['diameter']))
with np.load(dest,allow_pickle=False) as check:
    assert np.array_equal(check['u_xz'],velocity[0,0,:,j,:])
    assert np.array_equal(check['w_xz'],velocity[0,2,:,j,:])
    plane=check['u_xz']
    profile={'shape':list(plane.shape),'dtype':str(plane.dtype),'finite':bool(np.isfinite(plane).all()),
             'min':float(plane.min()),'max':float(plane.max()),'mean':float(plane.mean())}
write(OUT/'FLOW_SNAPSHOT_PROVENANCE.json',{'source':'evaluation/n81_B_w1/common_state.pkl.gz',
    'selection':'First predeclared B window, zero-yaw shared scoring start; no selection on visual appearance or gain.',
    'physical_phase':'After 150 zero-yaw controls, before policy/zero scoring branches.',
    'farm_step_index':7500,'farm_time_s':1500,'external_inlet_cursor':27500,
    'snapshot_kind':'Instantaneous actual LES velocity; not a mean or paired RL-controlled field.',
    'compressed_bytes':row['expected_compressed_bytes'],'compressed_sha256':row['expected_compressed_sha256'],
    'raw_bytes':row['original_bytes'],'raw_sha256':row['original_sha256'],
    'compressed_and_raw_SHA_rechecked_this_revision':True,
    'structural_loader_sha256':sha(tools/'saved_prefix_cpu_contract.py'),
    'grid_case_sha256':sha(BASE/'repro-pack/n81_20261015_minor_revision_repro/payload/visual_revision/configs/executed_physical_contract.json'),
    'grid_axes':'x streamwise, y vertical, z lateral; arange(n)*recorded spacing',
    'grid_plane_index':j,'actual_y_m':float(y[j]),'nominal_hub_y_m':90,
    'interpolation_or_new_simulation':False,'velocity_plane_profile':profile,
    'array_export':{'path':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest)},
    'source_state_modified':False,'new_GPUh':0})
raw.unlink()  # Only this job's redundant decoded temporary; original gzip remains untouched.
write(OUT/'JOB_EXIT.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-START,
    'output':dest.name,'temporary_decoded_copy_removed':True,'original_preserved':True,'new_GPUh':0})
print(json.dumps({'status':'verified_original_state_plane_exported','profile':profile,'new_GPUh':0}),flush=True)
