#!/usr/bin/env python3
"""Source-checked CPU redesign of the paper's quantitative figures.

No LES execution; power and yaw use all recorded control-boundary samples.
Run --profile before --preview, visually review previews, then --export.
"""
from pathlib import Path
import argparse, hashlib, json, sys, shutil, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
sys.path.insert(0,str(ROOT/'vendor'))
from profile_data import profile_data
from visual_qa import audit_layout
from check_figure import check_figure
WIDTH=178/25.4
BLUE='#0072B2'; ORANGE='#D55E00'; GRAY='#6B7280'; DARK='#202A35'; TEAL='#009E73'
SRC={'B':BLUE,'C':ORANGE}
CMAP=LinearSegmentedColormap.from_list('signed_gain',['#2166AC','#F7F7F7','#B35806'])
REPORT={}; MAPPING={}; INPUTS={}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(name):
 p=DATA/name;INPUTS[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def checkequal(a,b,label,atol=1e-8):
 if not np.allclose(a,b,rtol=1e-11,atol=atol):raise AssertionError(label)
def trace(sub):
 p=DATA/sub;INPUTS[str(p.relative_to(ROOT))]=sha(p)
 with np.load(p,allow_pickle=False) as z:r={k:z[k].astype(np.float64) for k in z.files}
 for k,v in r.items():
  if not np.isfinite(v).all():raise AssertionError(str(p)+k+' nonfinite')
 return r

def load():
 s=readj('reference_mechanism_summary.json');sp=readj('SPATIAL_DATA.json');e=readj('E1_COMPARISON.json');p2=readj('reference_P2_summary.json')
 traces={};records=[];allrates=[]
 for source in ['B','C']:
  ezs=[];ers=[]
  for w in range(1,5):
   ctx=f'{source}_w{w}';result=readj(f'BC/{ctx}/RESULT.json')
   z=trace(f'BC/{ctx}/zero.npz');r=trace(f'BC/{ctx}/rl_r1.npz')
   for mode,t in [('zero',z),('rl_r1',r)]:
    if INPUTS[f'data/BC/{ctx}/{mode}.npz']!=result['controllers'][mode]['trace_sha256']:raise AssertionError(ctx+mode+' SHA')
    assert t['farm_power_mw'].shape==(2000,1) and t['yaw_deg'].shape==(2000,1,81)
   assert np.array_equal(z['inlet_mean_u'],r['inlet_mean_u']) and not np.any(z['yaw_deg'])
   ez=z['farm_power_mw'].sum()*10/3600;er=r['farm_power_mw'].sum()*10/3600
   ref=next(a for a in s['windows'] if a['source_id']==source and a['window_number']==w)
   checkequal(100*(er/ez-1),ref['gain_percent'],ctx+' gain')
   yaw=r['yaw_deg'][:,0];rates=np.abs(np.diff(np.vstack([np.zeros((1,81)),yaw]),axis=0))/10;allrates.append(rates)
   ezs.append(ez);ers.append(er)
   traces[ctx]=(z,r)
   records.append({'source':source,'window':str(w),'energy_gain_percent':100*(er/ez-1),'zero_energy_MWh':ez,'RL_energy_MWh':er,'negative_power_difference_controls':int(np.sum(r['farm_power_mw']<z['farm_power_mw'])),'minimum_instantaneous_power_difference_MW':float(np.min(r['farm_power_mw']-z['farm_power_mw'])),'yaw_min_deg':float(yaw.min()),'yaw_max_deg':float(yaw.max()),'command_travel_deg_per_turbine_hour':ref['yaw']['mean_command_travel_deg_per_turbine_hour']})
  so=next(a for a in s['sources'] if a['source_id']==source)
  checkequal(100*(sum(ers)/sum(ezs)-1),so['gain_percent'],source+' aggregate')
  pyaw=np.concatenate([traces[f'{source}_w{w}'][1]['yaw_deg'][:,0] for w in range(1,5)]).mean(0)
  zt=sum(traces[f'{source}_w{w}'][0]['turbine_power_mw'].sum(axis=(0,1))*10/3600 for w in range(1,5))
  rt=sum(traces[f'{source}_w{w}'][1]['turbine_power_mw'].sum(axis=(0,1))*10/3600 for w in range(1,5))
  checkequal(pyaw,sp['sources'][source]['time_mean_yaw_deg'],'spatial yaw')
  checkequal(100*(rt/zt-1),sp['sources'][source]['turbine_gain_percent'],'spatial gain')
  checkequal((rt-zt).reshape(9,9).sum(1),so['row_gain_mwh'],'row energy')
 return s,sp,e,p2,traces,records,np.concatenate(allrates).ravel()

def style():
 fonts=ROOT.parent/'fonts/STIXTwoText'
 for variant in ['Regular','Bold','Italic','BoldItalic']:
  p=fonts/f'STIXTwoText-{variant}.otf'
  if not p.is_file():raise FileNotFoundError(p)
  font_manager.fontManager.addfont(str(p))
 plt.rcParams.update({'font.family':['STIX Two Text','STIXGeneral'],'font.serif':['STIX Two Text'],'mathtext.fontset':'stix','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'axes.titleweight':'normal','xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7B8188','xtick.color':DARK,'ytick.color':DARK,'text.color':DARK,'axes.labelcolor':DARK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','svg.hashsalt':'n81_scientific_redesign_20261015'})
def basegrid(ax,axis='y'):
 ax.grid(axis=axis,color='#E5E7EB',linewidth=.55);ax.set_axisbelow(True);ax.tick_params(length=3,width=.6)
def title(ax,label,text):
 # Independent panel letters and normal-weight headings share one baseline.
 # Point offsets retain their physical sizes at the fixed 178 mm figure width.
 for content,x,size,weight in [(label,0,9,'bold'),(text,12,8.5,'normal')]:
  tr=offset_copy(ax.transAxes,fig=ax.figure,x=x,y=10,units='points')
  ax.text(0,1,content,transform=tr,fontsize=size,fontweight=weight,
          ha='left',va='baseline',clip_on=False)

def legend(ax,**kw):return ax.legend(frameon=False,**kw)
def points(ax,x,y,color,marker='o',labels=True):
 ax.scatter(x,y,color=color,marker=marker,s=33,zorder=3,edgecolors='white',linewidths=.4)
 if labels:
  for a,b in zip(x,y):ax.annotate(f'{b:.2f}',(a,b),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)

def finish(fig,name,axes,description,args):
 fig.canvas.draw();issues=audit_layout(fig)
 texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
 sizes=[t.get_fontsize() for t in texts];assert min(sizes)>=7
 issues=[list(a) for a in issues]
 if any(a[0]=='FAIL' for a in issues):raise AssertionError((name,issues))
 out=args.audit;out.mkdir(parents=True,exist_ok=True)
 preview=out/(name+'_preview.png');fig.savefig(preview,dpi=180)
 with Image.open(preview) as im:ImageOps.grayscale(im).save(out/(name+'_gray_preview.png'),dpi=(180,180))
 inf={'size_inches':list(fig.get_size_inches()),'width_mm':float(fig.get_size_inches()[0]*25.4),'min_font_pt':min(sizes),'max_font_pt':max(sizes),'scipilot_layout':issues,'description':description,'axes_bounds':[list(ax.get_position().bounds) for ax in axes],'typography':{'family':['STIX Two Text','STIXGeneral'],'mathtext':'stix','label_pt':8.5,'tick_pt':8,'default_legend_pt':8,'panel_heading_pt':8.5,'panel_letter_pt':9,'minimum_detail_pt':7,'resolved_font':font_manager.findfont('STIX Two Text'),'actual_font_sizes_pt':sorted(set(float(t.get_fontsize()) for t in texts))}}
 if args.export:
  exported=[]
  for ext in ['pdf','svg','png']:
   path=args.out/(name+'.'+ext)
   kwargs={'metadata':{'Creator':'Source-checked N81 scientific data figures','CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
   fig.savefig(path,dpi=600,**kwargs)
   exported.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
  with Image.open(args.out/(name+'.png')) as im:ImageOps.grayscale(im).save(args.out/(name+'_grayscale.png'),dpi=(600,600))
  inf['exports']=exported
  checks={}
  for ext in ['pdf','svg','png']:
   result,info=check_figure(str(args.out/(name+'.'+ext)),min_dpi=300,target_inches=tuple(fig.get_size_inches()))
   checks[ext]={'issues':[list(x) for x in result],'info':info}
   if any(x[0]=='FAIL' for x in result):raise AssertionError((name,ext,result))
  inf['formal_export_checks']=checks
 REPORT[name]=inf;MAPPING[name]=description;plt.close(fig)

def microbatch(e,args):
 fig,ax=plt.subplots(1,2,figsize=(WIDTH,2.8));fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.84,wspace=.33)
 xpos=np.arange(3);ms=[1,2,4];reports=[e['reports'][str(m)] for m in ms]
 peak=np.array([r['device_memory']['peak_bytes_in_use']/1e9 for r in reports]);work=np.array([r['compiled_kernel_memory']['temp_size_in_bytes']/1e9 for r in reports]);persistent=reports[0]['persistent_state_bytes']/1e9
 for x,v,c,mark in [(xpos-.08,peak,BLUE,'o'),(xpos+.08,work,ORANGE,'s')]:points(ax[0],x,v,c,mark,False)
 ax[0].axhline(persistent,color=GRAY,ls=':',lw=1)
 for x,v in zip(xpos,peak):ax[0].annotate(f'{v:.2f}',(x-.08,v),xytext=(-3,8),textcoords='offset points',fontsize=8,ha='center')
 ax[0].set_ylabel('Memory (GB)');ax[0].set_ylim(0,38);title(ax[0],'a','Active flow working memory')
 ax[0].legend([Line2D([],[],marker='o',color=BLUE,lw=0),Line2D([],[],marker='s',color=ORANGE,lw=0),Line2D([],[],ls=':',color=GRAY)],['Measured active peak','Compiler workspace','Persistent states'],frameon=True,facecolor='white',edgecolor='none',framealpha=1,loc='lower right',fontsize=7.5)
 for i,r in enumerate(reports):
  t=np.array(r['retained_pass_seconds']);ax[1].scatter(i+np.linspace(-.10,.10,len(t)),t,c=BLUE,s=24,edgecolors='white',linewidths=.4)
  med=np.median(t);ax[1].plot([i-.17,i+.17],[med,med],color=DARK,lw=1.4);ax[1].annotate(f'{med:.2f} s',(i,med),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
 ax[1].set_ylim(0,16);ax[1].set_ylabel('Eight-context pass time (s)');title(ax[1],'b','Retained pass timings')
 for a in ax:a.set_xticks(xpos,['1','2','4']);a.set_xlabel('Active flow microbatch');a.set_xlim(-.45,2.45);basegrid(a)
 finish(fig,'microbatch_effect',ax,{'source':'data/E1_COMPARISON.json','memory':'reports[m].device_memory.peak_bytes_in_use and compiled_kernel_memory.temp_size_in_bytes, /1e9','timing':'all three retained_pass_seconds per m','unit':'one measured peak and three sequential timings for each active flow microbatch; fixed eight resident contexts; no CI'},args)

def scaling(s,args):
 top=s['efficiency']['topologies'];ps=[1,2,4];x=np.arange(3)
 fig,axs=plt.subplots(2,2,figsize=(WIDTH,4.45));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.90,hspace=.57,wspace=.30)
 a,b,c,d=axs.ravel()
 collect=np.array([top[str(p)]['normal_collector_seconds_median'] for p in ps]);learn=np.array([top[str(p)]['normal_learner_including_communication_seconds_median'] for p in ps])
 total=np.array([top[str(p)]['total_including_cold_reset_learner_sync_and_checkpoint_seconds'] for p in ps]);cost=np.array([s['costs']['efficiency_allocated_gpu_hours_by_topology'][str(p)] for p in ps])
 points(a,x,collect,BLUE);points(b,x,learn,ORANGE);points(c,x,total/60,BLUE);points(d,x,cost,TEAL)
 for axis,values,label,ttl,letter in [(a,collect,'Time per block (s)','LES collection','a'),(b,learn,'Time per block (s)','Learning + communication','b'),(c,total/60,'Whole profile time (min)','Elapsed time','c'),(d,cost,'Allocated GPU hours','Resource cost','d')]:
  axis.set_xticks(x,['1','2','4']);axis.set_xlabel('GPUs');axis.set_ylabel(label);axis.set_ylim(0,max(values)*1.24);axis.set_xlim(-.4,2.4);title(axis,letter,ttl);basegrid(axis)
 finish(fig,'scaling_breakdown',list(axs.ravel()),{'source':'data/reference_mechanism_summary.json','upper_panels':'efficiency.topologies[p].normal_collector_seconds_median and normal_learner_including_communication_seconds_median','lower_panels':'total_including_cold_reset_learner_sync_and_checkpoint_seconds/60, costs.efficiency_allocated_gpu_hours_by_topology','unit':'one bounded 3072-interaction matched-work profile per topology; regular blocks use component medians; topologies discrete/unconnected'},args)

def process(traces,args):
 fig,axes=plt.subplots(3,2,figsize=(WIDTH,5.85));fig.subplots_adjust(left=.09,right=.88,bottom=.10,top=.90,wspace=.20,hspace=.30)
 t=(np.arange(2000)+1)*10/3600
 startup=[]
 ymax=max(float(np.max(z['farm_power_mw'])) for z,r in traces.values());ymax=max(ymax,max(float(np.max(r['farm_power_mw'])) for z,r in traces.values()))
 cumall=[]
 for ctx,(z,r) in traces.items():cumall.append(np.cumsum((r['farm_power_mw']-z['farm_power_mw']).ravel())*10/3600)
 low=min(0,min(float(v.min()) for v in cumall));high=max(float(v.max()) for v in cumall)
 for col,source in enumerate(['B','C']):
  z,r=traces[source+'_w1'];p0=z['farm_power_mw'].ravel();pr=r['farm_power_mw'].ravel();cum=np.cumsum(pr-p0)*10/3600
  a,b,c=axes[:,col]
  a.plot(t,p0,c=GRAY,lw=.75,ls='--',label='Zero yaw');a.plot(t,pr,c=BLUE,lw=.75,label='RL');a.set_ylim(0,ymax*1.03);title(a,'ab'[col],f'Source {source}, first scoring window');a.set_ylabel('Farm power (MW)' if col==0 else '');basegrid(a)
  if col==0:legend(a,loc='lower right',ncol=2)
  b.plot(t,cum,c=ORANGE,lw=1.35);b.axhline(0,c=GRAY,lw=.7);b.set_ylim(low-1,high*1.06);b.set_ylabel('Cumulative energy\ndifference (MWh)' if col==0 else '');title(b,'cd'[col],'RL − zero yaw');basegrid(b)
  # The full 2000-point parent axis remains intact; enlarge every saved
  # control-boundary sample in the first 720 s, without fixed y clipping.
  early=(t*60<=12);early_t=t[early]*60;early_cum=cum[early]
  min_idx=int(np.argmin(cum));return_idx=int(np.flatnonzero((np.arange(len(cum))>min_idx)&(cum>=0))[0])
  startup.append({'source':source,'main_samples':len(cum),'inset_samples':int(early.sum()),'inset_time_bounds_minutes':[0,12],'minimum_MWh':float(cum[min_idx]),'minimum_time_seconds':float(t[min_idx]*3600),'first_nonnegative_after_minimum_seconds':float(t[return_idx]*3600)})
  inset=b.inset_axes([.17,.70,.43,.27]);inset.set_facecolor('white')
  inset.plot(early_t,early_cum,c=ORANGE,lw=1.0);inset.axhline(0,c=GRAY,lw=.65)
  extent=float(np.ptp(early_cum));inset.set_ylim(float(early_cum.min())-.13*extent,float(early_cum.max())+.13*extent)
  inset.set_xlim(0,12);inset.set_xticks([0,5,10]);inset.set_yticks([-.5,0,.5] if early_cum.max()>.5 else [-.5,0])
  inset.set_xlabel('Time (min)',fontsize=7,labelpad=1);inset.set_ylabel('ΔE (MWh)',fontsize=7,labelpad=1)
  inset.set_title('First 12 min',fontsize=7,pad=3);inset.tick_params(labelsize=7,length=2,width=.55,pad=1.5)
  inset.spines['top'].set_visible(True);inset.spines['right'].set_visible(True)
  inset.scatter([t[min_idx]*60],[cum[min_idx]],s=10,c=ORANGE,zorder=4)
  inset.annotate(f'{cum[min_idx]:.3f}',xy=(t[min_idx]*60,cum[min_idx]),xytext=(.7,-.29),fontsize=7,ha='left',va='center',arrowprops={'arrowstyle':'-','lw':.5,'color':DARK})
  inset.scatter([t[return_idx]*60],[cum[return_idx]],s=12,c=DARK,marker='d',zorder=4)
  inset.annotate(f'{t[return_idx]*3600:.0f} s',xy=(t[return_idx]*60,cum[return_idx]),xytext=(5.7,.27 if source=='B' else .15),fontsize=7,ha='center',va='center',arrowprops={'arrowstyle':'-','lw':.5,'color':DARK})
  yaw=r['yaw_deg'][:,0].T
  im=c.imshow(yaw,origin='lower',aspect='auto',extent=[0,20000/3600,0,81],cmap='cividis',vmin=-40,vmax=40,interpolation='nearest',rasterized=False)
  for row in range(9,81,9):c.axhline(row,color='white',lw=.35,alpha=.7)
  c.set_yticks(np.arange(4.5,81,9),np.arange(1,10));c.set_ylabel('Streamwise row\n(9 turbines each)' if col==0 else '');title(c,'ef'[col],'All 81 yaw commands');c.set_xlabel('Scoring time (h)')
  for aa in [a,b,c]:aa.set_xlim(0,20000/3600);aa.set_xticks([0,1,2,3,4,5]);aa.tick_params(labelbottom=aa==c)
  if col==1:
   for aa in [a,b,c]:aa.tick_params(labelleft=False)
 cbax=fig.add_axes([.90,.10,.018,.236]);cb=fig.colorbar(im,cax=cbax);cb.set_label('Applied yaw (°)');cb.set_ticks([-40,-20,0,20,40])
 finish(fig,'control_process',list(axes.ravel()),{'source':[f'data/BC/{s}_w1/{c}.npz' for s in ['B','C'] for c in ['zero','rl_r1']],'selection':'First scoring window of each physical source; fixed before reading performance, full 2000 controls; all 8 windows covered in all_window_dynamics','x':'(control+1)*10/3600 h','power':'farm_power_mw, unsmoothed, no samples omitted','cumulative':'float64 cumsum(RL farm_power_mw - zero farm_power_mw)*10/3600','startup_insets':startup,'inset_encoding':'all 72 controls in the first 12 min; y limits from actual data extrema plus 13 percent range padding; zero line, minimum and first post-minimum nonnegative sample marked; full main axes and all 2000 controls retained','yaw':'yaw_deg[:,0,:].T with no spatial/time averaging; y9 turbine groups in actual row-major executed order'},args)

def all_dynamics(traces,args):
 fig,axes=plt.subplots(4,2,figsize=(WIDTH,6.5));fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.20,hspace=.50)
 t=(np.arange(2000)+1)*10/3600
 curves={ctx:np.cumsum((r['farm_power_mw']-z['farm_power_mw']).ravel())*10/3600 for ctx,(z,r) in traces.items()};lo=min(0,min(v.min() for v in curves.values()))-1;hi=max(v.max() for v in curves.values())*1.08
 for row in range(4):
  for col,source in enumerate(['B','C']):
   a=axes[row,col];v=curves[f'{source}_w{row+1}'];a.plot(t,v,color=SRC[source],lw=1.15);a.axhline(0,c=GRAY,lw=.6);a.set_ylim(lo,hi);a.set_xlim(0,t[-1]);a.set_xticks([0,1,2,3,4,5]);title(a,chr(97+2*row+col),f'Source {source}, window {row+1}');basegrid(a)
   if col==0:a.set_ylabel('')
   else:a.tick_params(labelleft=False)
   if row==3:a.set_xlabel('Scoring time (h)')
 fig.supylabel('Cumulative energy difference (MWh)',x=.025,fontsize=8.5)
 finish(fig,'all_window_dynamics',list(axes.ravel()),{'source':'all 16 data/BC/*/{zero,rl_r1}.npz','values':'float64 cumsum(RL farm_power_mw-zero farm_power_mw)*10/3600; every sample','unit':'all four dependent windows within each of two physical sources; identical axes'},args)

def spatial(s,sp,args):
 pos=np.asarray(sp['geometry']['turbine_positions_m'])/1000;rows=np.arange(1,10)
 fig=plt.figure(figsize=(WIDTH,4.8));gs=fig.add_gridspec(2,2,left=.11,right=.88,bottom=.12,top=.90,height_ratios=[1.35,1],hspace=.45,wspace=.20)
 a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:]);axes=[a,b,c]
 for axis,source,letter in [(a,'B','a'),(b,'C','b')]:
  gain=np.asarray(sp['sources'][source]['turbine_gain_percent']);positive=gain>=0
  for mask,marker in [(positive,'o'),(~positive,'v')]:
   im=axis.scatter(pos[mask,0],pos[mask,2],c=gain[mask],cmap=CMAP,norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50),s=53,marker=marker,edgecolors=DARK,linewidths=.35)
  axis.set_aspect('equal');axis.set_xlim(0,6.174);axis.set_ylim(0,5.922);axis.set_xticks([0,2,4,6]);axis.set_yticks([0,2,4]);axis.set_xlabel('Streamwise x (km)');axis.set_ylabel('Lateral z (km)' if source=='B' else '');title(axis,letter,'Source '+source);axis.spines['top'].set_visible(True);axis.spines['right'].set_visible(True)
  if source=='C':axis.tick_params(labelleft=False)
 cbax=fig.add_axes([.90,.48,.018,.35]);bar=fig.colorbar(im,cax=cbax);bar.set_label('Turbine energy gain (%)');bar.set_ticks([-50,-25,0,25,50])
 for source,offset,marker in [('B',-.10,'o'),('C',.10,'s')]:
  so=next(v for v in s['sources'] if v['source_id']==source);y=np.asarray(so['row_gain_mwh']);c.vlines(rows+offset,0,y,color=SRC[source],lw=1.1,alpha=.7);c.scatter(rows+offset,y,c=SRC[source],marker=marker,s=30,label='Source '+source,zorder=3)
 c.axhline(0,c=GRAY,lw=.8);c.set_xticks(rows);c.set_xlabel('Streamwise turbine row');c.set_ylabel('Energy difference\n(MWh)');c.yaxis.set_label_coords(-.06,.5);c.set_xlim(.55,9.45);title(c,'c','Upstream loss and downstream recovery');legend(c,loc='lower right',ncol=2);basegrid(c)
 finish(fig,'spatial_control_gain',axes,{'source':'data/SPATIAL_DATA.json and all 16 BC traces; SPATIAL_DATA validated by trace integration','upper':'100*(sum(RL turbine_power_mw)*10/3600 / sum(zero turbine_power_mw)*10/3600 - 1); actual turbine x,z coordinates; positive circle, loss triangle; shared signed color scale','lower':'(RL integrated turbine energy-zero integrated turbine energy) summed over 9 lateral turbines per streamwise row and four windows, separately B/C','no_interpolation':True,'unit':'spatially coupled turbines, source aggregates, no CI'},args)

def rows_plot(s,args):
 fig,ax=plt.subplots(figsize=(WIDTH,2.4));fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.82);rows=np.arange(1,10)
 for source,offset,marker in [('B',-.08,'o'),('C',.08,'s')]:
  v=next(x for x in s['sources'] if x['source_id']==source)
  ax.scatter(rows+offset,v['row_gain_percent'],c=SRC[source],marker=marker,s=30,label='Source '+source,zorder=3);ax.vlines(rows+offset,0,v['row_gain_percent'],color=SRC[source],lw=.7,alpha=.6)
 ax.axhline(0,c=GRAY,lw=.7);ax.set_xticks(rows);ax.set_xlabel('Streamwise turbine row');ax.set_ylabel('Row energy gain (%)');ax.set_title('Relative row energy response',loc='left',fontweight='normal',fontsize=8.5,pad=10);basegrid(ax)
 legend(ax,loc='lower right',ncol=2)
 finish(fig,'row_energy',[ax],{'source':'data/reference_mechanism_summary.json sources[source].row_gain_percent, independently validated from turbine trace integration','unit':'all four windows per physical source; source-level relative row energy percentages, no inferential error bars','absolute_energy':'main Figure 6c retains absolute row energy differences in MWh'},args)

def blocks(s,args):
 fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.4),sharex=True,sharey=True);fig.subplots_adjust(left=.09,right=.86,bottom=.26,top=.85,wspace=.15)
 values=np.array([[a['gain_percent'] for a in w['time_blocks_500_controls']] for w in s['windows']]);norm=plt.Normalize(0,22)
 for col,source in enumerate(['B','C']):
  ax=axes[col];arr=values[col*4:(col+1)*4];im=ax.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,arr,cmap='cividis',norm=norm,shading='flat');ax.set_ylim(3.5,-.5);ax.set_xticks(range(4),['0–500','500–1000','1000–1500','1500–2000'],rotation=25,ha='right');ax.set_yticks(range(4),['1','2','3','4']);ax.set_xlabel('Scored control interval');ax.set_ylabel('Window within source' if col==0 else '');title(ax,'ab'[col],'Source '+source)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black' if arr[i,j]>13 else 'white')
 cbax=fig.add_axes([.885,.26,.014,.59]);bar=fig.colorbar(im,cax=cbax);bar.set_label('Block energy gain (%)')
 finish(fig,'time_block_gain',axes,{'source':'reference_mechanism_summary windows time_blocks_500_controls; independently recomputed','unit':'all 32 predeclared disjoint time blocks within 8 correlated windows; not 32 physical replicates','zero_based_control_bounds':'[0,500),[500,1000),[1000,1500),[1500,2000)'},args)

def yaw(s,rates,args):
 fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.8));fig.subplots_adjust(left=.11,right=.98,bottom=.13,top=.91,hspace=.58,wspace=.32);a,b,c,d=axes.ravel();rows=np.arange(1,10)
 for source in ['B','C']:
  for w in s['windows']:
   if w['source_id']==source:a.plot(rows,w['row_mean_yaw_deg'],color=SRC[source],alpha=.35,lw=.8,ls='-' if source=='B' else '--')
 a.plot(rows,s['average_pattern']['row_mean_yaw_deg'],c=DARK,lw=1.3,marker='o',ms=3,label='Eight-window mean');a.axhline(0,c=GRAY,lw=.6);a.set_xticks(rows);a.set_xlabel('Streamwise row');a.set_ylabel('Time-mean yaw (°)');title(a,'a','Spatial command pattern')
 a.legend([Line2D([],[],c=SRC['B'],alpha=.35,lw=.8,ls='-'),Line2D([],[],c=SRC['C'],alpha=.35,lw=.8,ls='--'),Line2D([],[],c=DARK,lw=1.3,marker='o',ms=3)],['Source B windows','Source C windows','Eight-window mean'],frameon=False,loc='upper right',fontsize=7)
 bins=np.linspace(0,1,31);b.hist(rates,bins=bins,weights=np.ones(len(rates))*100/len(rates),color=BLUE,edgecolor='white',lw=.3);b.set_xlabel('Absolute command-equivalent rate (°/s)');b.set_ylabel('Samples (%)');b.set_xlim(0,1);b.set_ylim(0,8);title(b,'b','Command-rate distribution')
 x=np.arange(8);windows=s['windows'];travel=np.array([w['yaw']['mean_command_travel_deg_per_turbine_hour'] for w in windows]);sat=100*np.array([w['yaw']['strict_angle_saturation_fraction'] for w in windows]);near=100*np.array([w['yaw']['near_angle_limit_fraction_abs_at_least_38_deg'] for w in windows])
 for i,w in enumerate(windows):c.scatter(i,travel[i],c=SRC[w['source_id']],marker='o' if w['source_id']=='B' else 's',s=30)
 d.scatter(x-.1,sat,c=BLUE,marker='o',s=25,label='At 40°');d.scatter(x+.1,near,c=ORANGE,marker='s',s=25,label=r'|yaw| $\geq$ 38°');legend(d,loc='lower right',fontsize=7)
 for ax in [c,d]:ax.set_xticks(x,[w['source_id']+str(w['window_number']) for w in windows]);ax.set_xlim(-.5,7.5);ax.set_xlabel('Window within physical source');ax.axvline(3.5,c='#BCC1C7',lw=.6)
 c.set_ylabel('Command travel\n(°/turbine/h)');c.yaxis.set_label_coords(-.14,.5);c.set_ylim(0,travel.max()*1.2);d.set_ylabel('Samples (%)');d.set_ylim(0,max(near.max()*1.3,1));title(c,'c','Command travel');title(d,'d','Angle-limit occupancy')
 for ax in axes.ravel():basegrid(ax)
 finish(fig,'yaw_motion',list(axes.ravel()),{'source':'all 8 RL yaw_deg trace arrays, reference_mechanism_summary yaw fields checked','rate':'abs(diff([initial_zero;yaw_deg]))/10; all 1,296,000 turbine-control samples','occupancy':'strict40 within4FP32ULP and near38; plotted values per window','unit':'command equivalents; dependent time samples/windows; no measured drive load/fatigue or iid inference'},args)

def learning(p2,args):
 fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.9));fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.84,wspace=.30)
 x=np.array([n['actual_training_interactions'] for n in p2['nodes']])/1e6;gain=np.array([n['modeled_gross_ADM_energy_gain_percent'] for n in p2['nodes']]);travel=np.array([n['command_equivalent_travel_deg_per_turbine_hour'] for n in p2['nodes']]);actual=[]
 for n in p2['nodes']:
  step=n['actual_training_interactions'];name='zero.npz' if not step else f'rl_step{step}.npz';tr=trace('P2/'+name)
  if INPUTS['data/P2/'+name]!=n['trace']['sha256']:raise AssertionError('P2 SHA')
  energy=tr['farm_power_mw'].sum()*10/3600;checkequal(energy,n['modeled_gross_ADM_energy_mwh'],'P2 energy');actual.append(energy)
 checkequal(100*(np.array(actual)/actual[0]-1),gain,'P2 gain')
 for ax,vals,col,mark,letter,ttl,ylab in zip(axes,[gain,travel],[BLUE,ORANGE],['o','s'],['a','b'],['Saved-policy energy response','Saved-policy command travel'],['Modeled gross energy gain (%)','Command travel (°/turbine/h)']):
  ax.plot(x,vals,c=col,lw=1,marker=mark,ms=4);ax.axhline(0,c=GRAY,lw=.7);ax.set_xlim(-.03,1.06);ax.set_ylim(0,max(vals)*1.18);ax.set_xlabel('Training interactions (millions)');ax.set_ylabel(ylab);ax.set_xticks([0,.131072,.524288,1.000448],['0','0.131','0.524','1.000']);title(ax,letter,ttl);basegrid(ax)
  ax.scatter([x[-1]],[vals[-1]],s=65,facecolors='none',edgecolors=DARK,lw=.8)
 finish(fig,'learning_curve_A',axes,{'source':'all four data/P2 traces and reference_P2_summary, each SHA checked and gain recomputed','x':'actual_training_interactions/1e6','unit':'four saved-policy nodes evaluated on one source-A development window; final budget actor highlighted; no selection of best checkpoint','lines':'ordered training interaction nodes, joined as reading guides, intermediate performance unobserved'},args)

def profile(s,sp,e,p2,traces,records,rates,args):
 p=ROOT/'window_data_profile.csv'
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 auto=profile_data(str(p),group_cols=['source'])
 result={'skill_profile':auto,'manual_type_overrides':{'source':'categorical physical independent source','window':'categorical nested dependent window, not ordinal dose','turbine':'spatially coupled unit, not independent replicate','time':'serially correlated controls'},'replication':{'independent_sources':2,'dependent_windows_per_source':4,'time_samples_per_window':2000,'turbines':81,'optimizer_seed':1},'missing_or_nonfinite':0,'all_16_branch_shapes':[2000,1,81],'energy_unit':'modeled gross actuator-disc energy, MWh; not net electricity','rate_samples':int(len(rates)),'command_rate_range':[float(rates.min()),float(rates.max())],'figure_selection_decision':'show individual windows and source labels; no box/violin/CI or significance; preserve negative instantaneous differences; first window from each source chosen for main time-process illustration; all windows retained in supplementary dynamics','input_hashes':INPUTS,'environment':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
 (args.audit/'DATA_PROFILE.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
 print(json.dumps({'status':'profile complete','window_records':records,'rates':result['command_rate_range']},indent=2))

def farm(sp,args):
    geom=sp['geometry'];pos=np.asarray(geom['turbine_positions_m']);d=geom['diameter_m']
    flowpath=ROOT.parent/'flow_snapshot/B_w1_zero_yaw_scoring_start.npz'
    provpath=flowpath.with_name('FLOW_SNAPSHOT_PROVENANCE.json')
    INPUTS[str(flowpath.relative_to(ROOT.parent))]=sha(flowpath);INPUTS[str(provpath.relative_to(ROOT.parent))]=sha(provpath)
    provenance=json.loads(provpath.read_text());assert sha(flowpath)==provenance['array_export']['sha256']
    with np.load(flowpath,allow_pickle=False) as data:
        f={k:np.asarray(data[k]) for k in data.files}
    assert f['u_xz'].shape==(len(f['x_m']),len(f['z_m']))
    checkequal(f['turbine_xyz_m'],pos,'flow turbine coordinates')
    assert float(f['actual_plane_y_m'])==87.5
    assert np.isfinite(f['u_xz']).all()
    checkequal(float(f['u_xz'].min()),provenance['velocity_plane_profile']['min'],'flow minimum')
    checkequal(float(f['u_xz'].max()),provenance['velocity_plane_profile']['max'],'flow maximum')
    fig=plt.figure(figsize=(WIDTH,4.2));a=fig.add_axes([.08,.27,.49,.63]);b=fig.add_axes([.69,.63,.27,.25]);c=fig.add_axes([.69,.19,.27,.23])
    im=a.pcolormesh(f['x_m']/1000,f['z_m']/1000,f['u_xz'].T,shading='auto',cmap='cividis',vmin=0,vmax=11,rasterized=True)
    segments=[[(x/1000,(z-d/2)/1000),(x/1000,(z+d/2)/1000)] for x,y,z in pos]
    a.add_collection(LineCollection(segments,colors='white',linewidths=2.2));a.add_collection(LineCollection(segments,colors='#26313B',linewidths=.85))
    a.set_aspect('equal');a.set_xlim(0,6.174);a.set_ylim(0,5.922);a.set_xticks([0,2,4,6]);a.set_yticks([0,2,4]);a.set_xlabel('Streamwise x (km)');a.set_ylabel('Lateral z (km)');title(a,'a','Resolved LES flow at the scoring start')
    a.annotate('Inflow',xy=(1.65,5.70),xytext=(.40,5.70),arrowprops={'arrowstyle':'->','lw':1.1,'color':'white'},color='white',fontsize=8,va='center')
    a.spines['top'].set_visible(True);a.spines['right'].set_visible(True)
    cbax=fig.add_axes([.08,.095,.49,.025]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal');cb.set_label('Instantaneous streamwise velocity, u (m/s)',labelpad=3);cb.set_ticks([0,2,4,6,8,10])
    rows=sp['probes']['local_probe_rows'];xp=[r['resolved_x_relative_D'] for r in rows];zp=[r['resolved_z_relative_D'] for r in rows]
    b.scatter(xp,zp,s=8,color=GRAY);b.plot([0,0],[-.5,.5],c=BLUE,lw=2);b.set_xlim(-2.2,3.2);b.set_ylim(-1.1,1.1);b.set_aspect('equal');b.set_xticks([-2,0,2]);b.set_yticks([-1,0,1]);b.set_xlabel('Relative streamwise x/D');b.set_ylabel('Relative z/D');title(b,'b','77 velocity probes')
    c.set_xlim(0,6.174);c.set_ylim(0,500);c.axhline(90,c=GRAY,ls=':',lw=.7)
    for x in np.unique(pos[:,0]):c.plot([x/1000]*2,[90-d/2,90+d/2],c=BLUE,lw=1.2)
    c.axhline(0,c=DARK,lw=1);c.axhline(500,c=GRAY,ls='--',lw=.8);c.set_xticks([0,3,6]);c.set_yticks([0,250,500]);c.set_xlabel('Streamwise x (km)');c.set_ylabel('Height y (m)');title(c,'c','Vertical domain')
    finish(fig,'farm_domain',[a,b,c],{'source':['data/SPATIAL_DATA.json','figure_sources/flow_snapshot/B_w1_zero_yaw_scoring_start.npz','figure_sources/flow_snapshot/FLOW_SNAPSHOT_PROVENANCE.json'],'image':'actual instantaneous u_xz, source B first preselected scoring window start after150 zero-yaw warmup controls','sampling':'y=87.5m resolved LES grid plane, nominalhub90m','geometry':'actual81turbine roster, D126m, 5D spacing;496x41x483grid; physical axes xstreamwise/yvertical/zlateral','snapshot_unit':'one preserved zero-yaw scoring-start snapshot, not RL controlled or paired wake comparison','vector_raster':'u field rasterized in PDF; turbine positions, labels, curves vector'},args)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--profile',action='store_true');ap.add_argument('--preview',action='store_true');ap.add_argument('--export',action='store_true');ap.add_argument('--only',nargs='*');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.audit.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(args.out).free<40*1024**3:raise RuntimeError('40 GiB reserve')
 s,sp,e,p2,traces,records,rates=load()
 if args.profile:profile(s,sp,e,p2,traces,records,rates,args)
 if args.preview or args.export:
  style();jobs={'farm_domain':lambda:farm(sp,args),'microbatch_effect':lambda:microbatch(e,args),'scaling_breakdown':lambda:scaling(s,args),'control_process':lambda:process(traces,args),'all_window_dynamics':lambda:all_dynamics(traces,args),'spatial_control_gain':lambda:spatial(s,sp,args),'row_energy':lambda:rows_plot(s,args),'time_block_gain':lambda:blocks(s,args),'yaw_motion':lambda:yaw(s,rates,args),'learning_curve_A':lambda:learning(p2,args)}
  for name,fn in jobs.items():
   if not args.only or name in args.only:fn();print(name+' rendered',flush=True)
  (args.audit/('EXPORT_AUDIT.json' if args.export else 'PREVIEW_AUDIT.json')).write_text(json.dumps({'figures':REPORT,'input_hashes':INPUTS,'mapping':MAPPING,'new_GPUh':0,'source_checks':'all BC and P2 hashes checked; all finite; source/window/row/spatial metrics checked','python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__},indent=2,default=str)+'\n')
if __name__=='__main__':main()
