"""Shared publication-size styling and validated export; CPU only."""
from pathlib import Path
import sys,json,hashlib,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import offset_copy
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT
OUT=ROOT/'manuscript/figures'
VALID=ROOT/'validation'
PREVIEW=ROOT/'previews'
WIDTH=178/25.4
BLUE='#1E6B86'; ORANGE='#AF5A1D'; DARK='#26353D'; GRAY='#75828B'; GRID='#E4E9ED'; TEAL='#377E78'
SRC={'B':BLUE,'C':ORANGE}
sys.path.insert(0,str(ROOT/'figure_sources/data_figures/vendor'))
from visual_qa import audit_layout

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
def style():
 for v in ['Regular','Bold','Italic','BoldItalic']:
  font_manager.fontManager.addfont(str(ROOT/f'figure_sources/fonts/STIXTwoText/STIXTwoText-{v}.otf'))
 plt.rcParams.update({'font.family':['STIX Two Text','STIXGeneral'],'mathtext.fontset':'stix','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'axes.titleweight':'normal','xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#829099','xtick.color':DARK,'ytick.color':DARK,'text.color':DARK,'axes.labelcolor':DARK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','svg.hashsalt':'wind_energy_figure_synthesis_20261017','path.simplify':False})
def panel(ax,letter,text):
 for s,x,sz,weight in [(letter,0,10,'bold'),(text,13,9,'normal')]:
  ax.text(0,1,s,transform=offset_copy(ax.transAxes,fig=ax.figure,x=x,y=9,units='points'),ha='left',va='baseline',fontsize=sz,fontweight=weight,color=DARK,clip_on=False)
def grid(ax,axis='y'):
 ax.grid(axis=axis,color=GRID,lw=.5);ax.set_axisbelow(True);ax.tick_params(length=2.8,width=.6,pad=3)
def finish(fig,name,payload,changes):
 fig.canvas.draw()
 issues=[list(i) for i in audit_layout(fig)]
 if any(i[0]=='FAIL' for i in issues):raise RuntimeError((name,issues))
 texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
 sizes=sorted(set(float(t.get_fontsize()) for t in texts))
 if min(sizes)<7.5:raise RuntimeError((name,'font size',sizes))
 rr=fig.canvas.get_renderer();bounds=fig.bbox
 clipped=[]
 for t in texts:
  b=t.get_window_extent(rr)
  if b.x0<bounds.x0-1 or b.x1>bounds.x1+1 or b.y0<bounds.y0-1 or b.y1>bounds.y1+1:clipped.append(t.get_text())
 if clipped:raise RuntimeError((name,'clipped',clipped))
 def numeric(values):
  arr=np.ma.asarray(values,dtype=np.float64).filled(np.nan)
  flat=arr[np.isfinite(arr)]
  return {'shape':list(arr.shape),'sha256_float64_le':hashlib.sha256(arr.astype('<f8').tobytes()).hexdigest(),'min':float(flat.min()) if flat.size else None,'max':float(flat.max()) if flat.size else None,'values':arr.tolist() if arr.size<=500 else None}
 axes=list(fig.axes)
 for ax in list(axes):axes.extend(ax.child_axes)
 artists=[]
 for index,ax in enumerate(axes):
  rec={'axis_index':index,'bounds_figure_fraction':list(ax.get_position().bounds),'limits':[list(ax.get_xlim()),list(ax.get_ylim())],'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'lines':[],'collections':[],'images':[],'patches':[]}
  for p in ax.patches:
   entry={'type':type(p).__name__,'gid':p.get_gid(),'label':p.get_label()}
   if isinstance(p,matplotlib.patches.Rectangle):entry.update({'x':float(p.get_x()),'y':float(p.get_y()),'width':float(p.get_width()),'height':float(p.get_height()),'hatch':p.get_hatch()})
   rec['patches'].append(entry)
  for line in ax.lines:rec['lines'].append({'label':line.get_label(),'x':numeric(line.get_xdata()),'y':numeric(line.get_ydata()),'linestyle':line.get_linestyle(),'color':line.get_color(),'linewidth':float(line.get_linewidth()),'marker':line.get_marker()})
  for c in ax.collections:
   entry={'type':type(c).__name__,'label':c.get_label(),'offsets':numeric(c.get_offsets()),'paths':[numeric(p.vertices) for p in c.get_paths()],'facecolors':numeric(c.get_facecolors())}
   if c.get_array() is not None:entry['color_values']=numeric(c.get_array());entry['clim']=list(c.get_clim());entry['cmap']=c.get_cmap().name
   if hasattr(c,'get_segments'):entry['segments']=numeric(c.get_segments())
   rec['collections'].append(entry)
  for im in ax.images:rec['images'].append({'data':numeric(im.get_array()),'extent':list(im.get_extent()),'clim':list(im.get_clim()),'cmap':im.get_cmap().name,'interpolation':im.get_interpolation()})
  artists.append(rec)
 dump(VALID/(name+'_artists.json'),{'figure':name,'actual_matplotlib_artists':artists,'figure_level_lines':[{'gid':l.get_gid(),'label':l.get_label(),'x':numeric(l.get_xdata()),'y':numeric(l.get_ydata()),'color':l.get_color()} for l in fig.lines],'visible_texts':[{'text':t.get_text(),'font_pt':float(t.get_fontsize()),'gid':t.get_gid()} for t in texts]})
 fig.savefig(PREVIEW/(name+'.png'),dpi=180)
 with Image.open(PREVIEW/(name+'.png')) as im:ImageOps.grayscale(im).save(PREVIEW/(name+'_grayscale.png'))
 exports=[]
 for ext in (['pdf','svg','png'] if os.environ.get('WIND_FIG_FINAL')=='1' else []):
  p=OUT/(name+'.'+ext)
  kwargs={'metadata':{'Creator':'Source-verified Wind Energy figure refinement','CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
  fig.savefig(p,dpi=600,**kwargs)
  exports.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 if exports:
  with Image.open(OUT/(name+'.png')) as im:im.verify()
 dump(VALID/(name+'_refinement.json'),{'status':'FINAL_EXPORT_AFTER_PREVIEW_REVIEW' if exports else 'PREVIEW_ONLY_FOR_VISUAL_REVIEW','figure':name,'size_mm':(fig.get_size_inches()*25.4).tolist(),'font_sizes_pt':sizes,'layout_issues':issues,'out_of_canvas_texts':clipped,'source_checked_payload':payload,'changes':changes,'outputs':exports,'new_GPUh':0})
 print(name+(' exported' if exports else ' previewed'),flush=True)
 plt.close(fig)
style()
