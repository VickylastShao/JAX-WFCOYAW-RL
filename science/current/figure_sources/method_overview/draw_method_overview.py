#!/usr/bin/env python3
"""Source-linked, editable vector method schematic; no empirical fields are drawn.
Run with Python+matplotlib+numpy+Pillow already present in the bundled runtime.
"""
from pathlib import Path
import argparse,hashlib,json,platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Circle,FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image
W,H=178,130
ROOT=Path(__file__).resolve().parent
# Register existing fonts for this process only; no installation or global change.
FONT_DIRS=[ROOT.parent/'fonts'/'STIXTwoText']
FONT_FILES=[]
for font_dir in FONT_DIRS:
 if all((font_dir/('STIXTwoText-'+style+'.otf')).is_file() for style in ['Regular','Bold','Italic']):
  for style in ['Regular','Bold','Italic']:
   font_path=font_dir/('STIXTwoText-'+style+'.otf');font_manager.fontManager.addfont(font_path);FONT_FILES.append({'name':font_path.name,'sha256':hashlib.sha256(font_path.read_bytes()).hexdigest()})
  break
else:
 raise RuntimeError('The bundled STIX Two Text Regular/Bold/Italic fonts are required.')
plt.rcParams.update({'font.family':['STIX Two Text','STIXGeneral'],'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'stix','font.size':8})
TEXT=[];ART=[];EDGES=[]
def scene(gray=False,hide_titles=False,hide_l3=False):
 global TEXT,ART,EDGES
 TEXT=[];ART=[];EDGES=[]
 fig=plt.figure(figsize=(W/25.4,H/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.set_axis_off()
 ink='#24343D';muted='#61737D';blue='#1E6B86';orange='#9C521E';pale='#ECF3F6';warm='#F8F1EB';light='#F5F7F8'
 def c(x):
  if not gray:return x
  r,g,b=to_rgb(x);y=.2126*r+.7152*g+.0722*b;return (y,y,y)
 def text(i,x,y,s,sz=8.1,bold=False,col=ink,ha='center',level='L2',bg=False):
  if hide_titles and level=='title':return
  if hide_l3 and level=='L3':return
  # A compact type hierarchy matches the template's 10 pt STIX Two Text body.
  if level=='title':
   panel=ax.text(x,y,s[0],fontsize=9,fontweight='bold',color=c(col),ha='left',va='center',zorder=6);panel.set_gid(i+'_panel')
   TEXT.append({'id':i+'_panel','text':s[0],'font_pt':9,'level':'panel_label','x':x,'y':y,'artist':panel})
   s=s[3:];x+=3;sz=9;bold=False
  elif bold:sz=8.5
  else:sz=8 if sz>=7.7 else 7.5
  t=ax.text(x,y,s,fontsize=sz,fontweight='bold' if bold else 'normal',color=c(col),ha=ha,va='center',linespacing=1.25,zorder=6,bbox={'facecolor':'white','edgecolor':'none','pad':1.0} if bg else None);t.set_gid(i)
  TEXT.append({'id':i,'text':s,'font_pt':sz,'level':level,'x':x,'y':y,'artist':t})
 def box(i,x,y,w,h,fill='white',edge=muted,lw=.7,z=1):
  p=Rectangle((x,y),w,h,facecolor=c(fill),edgecolor=c(edge),lw=lw,zorder=z);p.set_gid(i);ax.add_patch(p);ART.append({'id':i,'kind':'rect','bounds':[x,y,w,h]})
 def line(i,p,col=muted,lw=.8,ls='-',z=2):
  a,=ax.plot(*zip(*p),color=c(col),lw=lw,ls=ls,zorder=z);a.set_gid(i);ART.append({'id':i,'kind':'line','points':p})
 def dot(i,x,y,r=.6,col=blue):
  p=Circle((x,y),r,facecolor=c(col),edgecolor='none',zorder=4);p.set_gid(i);ax.add_patch(p);ART.append({'id':i,'kind':'circle','center':[x,y],'r':r})
 def arrow(i,p,col=ink,lw=1.1,dash=False,semantics='data',prominence='primary'):
  a=FancyArrowPatch(path=MPath(p,[MPath.MOVETO]+[MPath.LINETO]*(len(p)-1)),arrowstyle='-|>',mutation_scale=7.5,lw=lw,color=c(col),linestyle=(0,(3,2)) if dash else '-',zorder=3);a.set_gid(i);ax.add_patch(a);EDGES.append({'id':i,'points':p,'semantics':semantics,'prominence':prominence})
 text('title_a',4,6,'a  Physical feedback at each 10 s control interval',9.2,True,ha='left',level='title')
 text('observations',20,17,'Observations',8.6,True,level='L1')
 # Sensor-vector glyph: values are deliberately not empirical data.
 for j in range(3):
  box('sensor_vector_'+str(j),7,24+j*2.8,26,1.8,fill=light,edge='#9AAAB2',lw=.45)
  for k in range(8):line(f'sensor_divider_{j}_{k}',[(9+k*3,24+j*2.8),(9+k*3,25.8+j*2.8)],col='#B4BFC5',lw=.35)
 text('sensor_mean',20,40,'77 probes + yaw\nper turbine',7.8,level='L2')
 text('policy',65,17,'Shared graph actor',8.6,True,level='L1')
 # Small schematic graph, explicitly scoped in caption; not a new architecture claim.
 for j in range(3):
  for k in range(3):
   if j<2:arrow(f'graph_edge_{j}_{k}',[(54+j*9.5,25+k*4),(62+j*9.5,25+k*4)],col='#9EB6C3',lw=.55,semantics='message',prominence='secondary')
   dot(f'graph_node_{j}_{k}',53+j*9.5,25+k*4,.9)
 text('policy_messages',65,38.5,'Upstream turbine messages',7.5,level='L2')
 text('policy_scope',65,43.3,'Shared turbine weights',7.5,level='L2')
 text('command',110,17,'Yaw command',8.6,True,level='L1')
 line('yaw_axis',[(97,36),(124,36)],col='#B4BFC5',lw=.6)
 line('held_command',[(98,32),(105,32),(105,27),(121,27)],col=blue,lw=1.7)
 text('yaw_update',110,42,'|Δγ| ≤ 10°\nthen hold orientation',7.8,level='L2')
 text('les',155,17,'Actuator-disc LES',8.6,True,level='L1')
 box('les_domain',137,23,36,14,fill=pale,edge=blue,lw=.75)
 for j in range(9):
  for k in range(9):line(f'turbine_{j}_{k}',[(139+j*3.9,24.7+k*1.38),(140+j*3.9,24.7+k*1.38)],col=blue,lw=.6)
 text('les_substeps',155,41.5,'50 × 0.2 s substeps\nPower − yaw penalty\n→ reward',7.5,level='L2')
 arrow('obs_to_policy',[(36,30),(48,30)])
 arrow('policy_to_yaw',[(82,30),(94,30)])
 arrow('yaw_to_les',[(125,30),(135,30)])
 arrow('physical_return',[(155,46.8),(155,54),(20,54),(20,46.8)],col=blue,lw=1.15,semantics='physical_feedback')
 text('return_label',87,54,'Next interval-averaged probes',8,col=blue,bg=True,level='L1')
 line('panel_separator',[(4,61),(174,61)],col='#CCD6DC',lw=.65)
 text('title_b',4,66,'b  Training execution separates three resource lifetimes',9.2,True,ha='left',level='title')
 text('training_only_scope',5,72,'Training only · 4 GPU ranks',7.5,col=muted,ha='left',level='L1')
 # Training-only region has distinct state, operator and learner shapes.
 box('stored_scope',5,76,36,34,fill=light,edge='#98A8B0',lw=.7)
 text('stored_title',23,80.7,'Retain 8 contexts',8.3,True,level='L1')
 for j in range(8):
  x=8+(j%4)*7.6;y=86+(j//4)*6.4
  box(f'context_{j}',x,y,5.7,4.7,fill=pale if j==0 else 'white',edge=blue if j==0 else '#98A8B0',lw=1.1 if j==0 else .55)
  line(f'context_hist_{j}',[(x+.7,y+1.3),(x+4.9,y+1.3)],col='#8AA3B1',lw=.5)
  line(f'context_cursor_{j}',[(x+.7,y+3.2),(x+3.2,y+3.2)],col='#8AA3B1',lw=.5)
 text('stored_details',23,104.6,'Flow + history + cursor',7.5,level='L3')
 box('workspace',51,76,38,34,fill=pale,edge=blue,lw=1.2)
 text('workspace_title',70,80.7,'Advance 1 at a time',8.3,True,col=blue,level='L1')
 text('physical_cycle_link',70,84.3,'Physical cycle (a) ×8',7.5,col=blue,level='L2')
 # Repeated visitation cue exposes the microbatch rather than showing memory as erased.
 box('active_slot',62,87.4,16,7,fill='white',edge=blue,lw=.9)
 for k in range(3):line('active_tensor_'+str(k),[(64,89+k*1.5),(75,89+k*1.5)],col='#86A8B8',lw=.55)
 text('workspace_details',70,100.2,'Reuse temporary arrays',7.5,col=blue,level='L2')
 text('workspace_context',70,105,'64 transitions / rank',7.5,level='L2')
 arrow('read_context',[(41,87),(51,87)],col=blue,lw=.8,semantics='state_read',prominence='secondary')
 arrow('retain_context',[(51,100),(41,100)],col=blue,lw=.8,semantics='state_write',prominence='secondary')
 # Replay is a data-store, not another flow operator.
 for j in [2,1,0]:box('replay_layer_'+str(j),100+j,87-j*1.2,24,12,fill='white',edge=orange,lw=.65)
 text('replay',113,91.2,'Replay',8.2,True,col=orange,level='L1')
 text('gather_contract',112,80.7,'All-gather',8.3,True,col=orange,level='L1')
 text('gather_details',112,104,'4 × 64 = 256\ntransitions / block',7.5,level='L2')
 arrow('transitions',[(89,91),(99,91)],col=orange,lw=1.0,semantics='experience')
 box('learner',138,76,35,34,fill=warm,edge=orange,lw=.8)
 text('learner_title',155.5,80.7,'Synchronized SAC',8.1,True,col=orange,level='L1')
 for j in range(4):dot('gpu_rank_'+str(j),143+j*8,89,.9,col=orange)
 line('gradient_average',[(143,91.7),(143,94),(167,94),(167,91.7)],col=orange,lw=.75)
 text('learner_details',155.5,101.2,'64 samples / rank\n256 updates / block',7.5,level='L2')
 arrow('replay_to_learn',[(126,91),(137,91)],col=orange,lw=1,semantics='sample')
 arrow('actor_update',[(155.5,110),(155.5,115),(70,115),(70,110)],col=orange,lw=.8,dash=True,semantics='parameter_update',prominence='secondary')
 text('actor_update_label',111,115,'Updated actor → next collection block',7.6,col=orange,bg=True,level='L2')
 text('training_scope',4,123,'Training: 4 ranks × 8 continuing contexts. Evaluation: fixed deterministic actor; physical loop only.',7.7,ha='left',level='L1')
 fig.canvas.draw()
 # Measure text extents against the actual publication-size canvas.
 rr=fig.canvas.get_renderer();bad=[]
 for t in TEXT:
  b=t['artist'].get_window_extent(rr); x0,y0=ax.transData.inverted().transform((b.x0,b.y0));x1,y1=ax.transData.inverted().transform((b.x1,b.y1))
  t['bounds_mm']=[min(x0,x1),min(y0,y1),abs(x1-x0),abs(y1-y0)]
  if min(x0,x1)<.2 or max(x0,x1)>W-.2 or min(y0,y1)<.2 or max(y0,y1)>H-.2:bad.append(t['id'])
 return fig,bad

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--audit-dir',type=Path,required=True);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);args.audit_dir.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for suffix,gray,ht,hl in [('',False,False,False),('_grayscale',True,False,False),('_review_hidden',True,True,True)]:
  f,bad=scene(gray,ht,hl)
  if bad:raise RuntimeError('clipped text:'+repr(bad))
  for ext in ['svg','pdf','png']:
   p=args.output_dir/f'method_overview{suffix}.{ext}';f.savefig(p,dpi=300,facecolor='white',metadata={'Creator':'source-linked method schematic'} if ext=='pdf' else None);manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  obj={'texts':[{k:v for k,v in x.items() if k!='artist'} for x in TEXT],'objects':ART,'edges':EDGES,'clipped_texts':bad,'minimum_effective_text_pt':min(x['font_pt'] for x in TEXT),'width_mm':W,'height_mm':H,'renderer':'matplotlib '+matplotlib.__version__}
  (args.audit_dir/f'OBJECTS{suffix}.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8');plt.close(f)
 for p in args.output_dir.glob('method_overview*.png'):
  with Image.open(p) as im: im.verify()
 (args.audit_dir/'RENDER_RECEIPT.json').write_text(json.dumps({'python':platform.python_version(),'renderer':'matplotlib '+matplotlib.__version__,'GPUh':0,'empirical_fields':False,'font_family':'STIX Two Text; STIXGeneral symbol fallback','font_files':FONT_FILES,'type_hierarchy_pt':[7.5,8,8.5,9],'outputs':manifest,'target_rendering':'SVG/PDF native vector exports; not PowerPoint or draw.io','native_sivia_gate':'not claimed; Sivia source-to-scene and visual grammar guidance used with project-local vector renderer'},indent=2)+'\n',encoding='utf8')
 print(json.dumps({'outputs':len(manifest),'status':'RENDERED','GPUh':0}))
if __name__=='__main__':main()
