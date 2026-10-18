#!/usr/bin/env python3
"""Render included figures to a new external directory; no experiment or simulation."""
from pathlib import Path
import argparse,os,subprocess,sys,json,time
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);p.add_argument('--export',action='store_true');a=p.parse_args();r=a.root.resolve();o=a.output.resolve();assert not o.exists() and not o.is_relative_to(r);o.mkdir(parents=True);v=r/'font_harmony_20261015/figure_sources';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu');commands=[]
for name,args in [('data',[v/'data_figures/plot_revision.py','--out',o/'data_figures','--audit',o/'data_audit','--profile','--export' if a.export else '--preview']),('method',[v/'method_overview/draw_method_overview.py','--output-dir',o/'method_figures','--audit-dir',o/'method_audit'])]:
 cmd=[sys.executable,'-B',*map(str,args)];t=time.monotonic()
 with (o/(name+'.stdout')).open('w') as so,(o/(name+'.stderr')).open('w') as se:z=subprocess.run(cmd,env=env,cwd=o,stdout=so,stderr=se,timeout=300)
 commands.append({'name':name,'command':cmd,'exit_code':z.returncode,'seconds':time.monotonic()-t});assert z.returncode==0,name+' failed; see retainedlogs'
(o/'RENDER_EXECUTION.json').write_text(json.dumps({'status':'pass','commands':commands,'new_GPUh':0,'no_LES':True,'export':a.export,'numerical_scope':'same saved observations; rendering version/metadata can change output file SHA across platforms'},indent=2)+'\n');print(json.dumps({'status':'pass','commands':len(commands),'GPUh':0}))
