#!/usr/bin/env python3
"""Portable CPU rendering of N9 archived-result figures.

Workflow: --profile, --preview, inspect the previews, then --export.
Dependencies: numpy, matplotlib, Pillow; pandas is used only by --profile.
Fonts and the SciPilot checks are local files, never installed globally.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from visual_qa import audit_layout
from check_figure import check_figure
from validate_n9_visual_data import analyze

BLUE='#0072B2'; ORANGE='#D55E00'; DARK='#202A35'; GRAY='#6B7280'
WIDTH=178/25.4

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonout(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def style():
    for variant in ['Regular','Bold','Italic','BoldItalic']:
        font_manager.fontManager.addfont(str(ROOT/f'fonts/STIXTwoText/STIXTwoText-{variant}.otf'))
    plt.rcParams.update({'font.family':['STIX Two Text','STIXGeneral'], 'font.serif':['STIX Two Text'],
        'mathtext.fontset':'stix', 'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,
        'axes.titleweight':'normal','xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'axes.linewidth':.65,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#7B8188','xtick.color':DARK,
        'ytick.color':DARK,'text.color':DARK,'axes.labelcolor':DARK,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','svg.hashsalt':'n9_visual_story_20261016'})

def title(ax,letter,heading):
    for s,x,size,weight in [(letter,0,9,'bold'),(heading,12,8.5,'normal')]:
        ax.text(0,1,s,transform=offset_copy(ax.transAxes,fig=ax.figure,x=x,y=10,units='points'),
                fontsize=size,fontweight=weight,ha='left',va='baseline',clip_on=False)

def grid(ax):
    ax.grid(axis='y',color='#E5E7EB',linewidth=.55);ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.6)

def save(fig,name,args,semantics):
    fig.canvas.draw()
    issues=[list(x) for x in audit_layout(fig)]
    if any(x[0]=='FAIL' for x in issues): raise AssertionError((name,issues))
    texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    sizes=sorted(set(t.get_fontsize() for t in texts))
    assert min(sizes)>=7
    renderer=fig.canvas.get_renderer()
    x0,y0,x1,y1=fig.bbox.extents
    outside=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0<x0-1 or b.x1>x1+1 or b.y0<y0-1 or b.y1>y1+1:
            outside.append({'text':t.get_text(),'bounds_px':list(b.extents)})
    if outside: raise AssertionError(('out-of-canvas text',outside))
    preview=args.audit/f'{name}_preview.png'
    fig.savefig(preview,dpi=180)
    with Image.open(preview) as im: ImageOps.grayscale(im).save(args.audit/f'{name}_grayscale_preview.png',dpi=(180,180))
    artists=[]
    for i,ax in enumerate(fig.axes):
        for line in ax.lines:
            artists.append({'axes':i,'kind':'line','label':line.get_label(),
                'x':np.asarray(line.get_xdata(),dtype=float).tolist(),
                'y':np.asarray(line.get_ydata(),dtype=float).tolist()})
        for col in ax.collections:
            arr=col.get_array()
            if arr is not None:
                artists.append({'axes':i,'kind':'collection_array','values':np.asarray(arr).tolist()})
            else:
                artists.append({'axes':i,'kind':'points','offsets':np.asarray(col.get_offsets()).tolist()})
        for image in ax.images:
            artists.append({'axes':i,'kind':'image','values':np.asarray(image.get_array()).tolist()})
    artist_json=json.dumps(artists,sort_keys=True,separators=(',',':'),allow_nan=False)
    report={'figure':name,'status':'MACHINE_CHECK_PASS_VISUAL_REVIEW_REQUIRED',
        'width_mm':float(fig.get_figwidth()*25.4),'height_mm':float(fig.get_figheight()*25.4),
        'font_sizes_pt':sizes,'font_family':'STIX Two Text; STIXGeneral symbol fallback',
        'resolved_font':str(Path(font_manager.findfont('STIX Two Text')).relative_to(ROOT)),
        'scipilot_layout_checks':issues,'out_of_canvas_text':outside,
        'semantics':semantics,'artist_sha256':hashlib.sha256(artist_json.encode()).hexdigest(),
        'new_GPUh':0}
    jsonout(args.audit/f'{name}_ARTISTS.json',artists)
    if args.export:
        exported=[]
        for ext in ['pdf','svg','png']:
            p=args.output/f'{name}.{ext}'
            meta={'Creator':'Source-checked N9 archived-result figures','CreationDate':None,'ModDate':None} if ext=='pdf' else {}
            fig.savefig(p,dpi=600,**({'metadata':meta} if meta else {}))
            checks,info=check_figure(str(p),min_dpi=300,target_inches=tuple(fig.get_size_inches()))
            if any(x[0]=='FAIL' for x in checks): raise AssertionError((name,ext,checks))
            exported.append({'filename':p.name,'sha256':sha(p),'bytes':p.stat().st_size,
                             'checks':[list(x) for x in checks],'info':info})
        with Image.open(args.output/f'{name}.png') as im:
            ImageOps.grayscale(im).save(args.audit/f'{name}_grayscale.png',dpi=(600,600))
        report['exports']=exported
    jsonout(args.audit/f'{name}_LAYOUT.json',report)
    plt.close(fig)
    return report

def direction_figure(data,args):
    x=np.array(data['directions_deg'])
    by={c['controller_id']:c for c in data['controllers']}
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.05),gridspec_kw={'width_ratios':[1.8,1]})
    fig.subplots_adjust(left=.077,right=.98,bottom=.185,top=.81,wspace=.38)
    a,b=axes
    a.axhline(0,color=GRAY,lw=.8,ls=':')
    for cid,color,marker,ls,label,dx in [
        ('selected_realtime_policy',BLUE,'o','-','Realtime policy',-.10),
        ('static_les_lookup',ORANGE,'s','--','Static LES lookup',.10)]:
        c=by[cid];pair=np.array(c['window_gain_percent_by_direction'])
        for j,jitter in enumerate([-.055,.055]):
            a.scatter(x+dx+jitter,pair[:,j],color=color,s=7,alpha=.65,zorder=3,edgecolors='none')
        a.plot(x,c['direction_mean_gain_percent'],color=color,ls=ls,lw=1.1,
               marker=marker,ms=3.2,mfc='white',mew=.8,label=label,zorder=4)
    a.set_xlim(-10.65,10.65);a.set_ylim(-.35,10.6)
    a.set_xticks(np.arange(-10,10.1,2.5));a.set_yticks(np.arange(0,11,2))
    a.set_xlabel('Fixed relative direction (°)');a.set_ylabel('Gain over paired zero yaw (%)')
    title(a,'a','Independent-source direction response');grid(a)
    handles,labels=a.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.077,.98),
               ncol=2,frameon=False,handlelength=2.,borderaxespad=0,fontsize=8)
    # One categorical rug; these are cases, not within-trajectory wind shifts.
    anchor=np.array([i for i in x if i/2.5==int(i/2.5)])
    midpoint=np.array([i for i in x if i not in anchor])
    a.scatter(anchor,np.full(len(anchor),10.25),s=13,marker='|',color=DARK,lw=1)
    a.scatter(midpoint,np.full(len(midpoint),10.25),s=9,marker='o',facecolors='none',edgecolors=GRAY,lw=.7)
    a.text(.98,.04,'Small dots: two windows',transform=a.transAxes,ha='right',fontsize=7.5)
    # Explicit key above the plot leaves the actual data panel unobscured.
    fig.text(.077,.885,'Direction key:  | training anchor    ○ midpoint',fontsize=7.5,ha='left')
    xx=np.array(data['lookup_interpolation']['directions_deg'])
    targets=np.array(data['lookup_interpolation']['group_targets_deg'])
    for group,(color,marker,ls) in enumerate([(BLUE,'o','-'),(ORANGE,'s','--'),(GRAY,'^',':')]):
        b.plot(xx,targets[:,group],color=color,marker=marker,ms=3.8,mfc='white',mew=.8,
               lw=1.,ls=ls,label=f'Group {group+1}')
    b.axhline(0,color=GRAY,lw=.6,alpha=.6,zorder=0)
    b.scatter([1.25],[0.],s=105,facecolors='none',edgecolors=DARK,lw=.8,zorder=5)
    b.annotate('All targets = 0°',xy=(1.25,0),xytext=(1.25,-18),ha='center',fontsize=8,
                arrowprops={'arrowstyle':'-','color':GRAY,'lw':.65,'shrinkA':1,'shrinkB':5})
    b.set_xlim(-.20,2.7);b.set_ylim(-24,24);b.set_xticks([0,1.25,2.5],['0','1.25','2.5'])
    b.set_yticks([-20,-10,0,10,20]);b.set_xlabel('Fixed relative direction (°)')
    b.set_ylabel('Lookup target yaw (°)');grid(b)
    title(b,'b','Frozen lookup interpolation')
    b.legend(loc='upper left',bbox_to_anchor=(-.01,1.0),frameon=False,
             fontsize=7.5,handlelength=1.8,labelspacing=.25,borderaxespad=.2)
    return save(fig,'n9_direction_response',args,{
        'panel_a':'All 17 fixed direction means; all 68 individual paired window gains for realtime and lookup.',
        'mean_lines':'Directions are ordered cases; lines guide the eye and are not within-trajectory time series.',
        'small_dots':'Two deterministic windows per direction; horizontal offsets only prevent occlusion; no uncertainty intervals.',
        'rug':'Nine training-anchor directions and eight intermediate directions.',
        'panel_b':'Frozen three-group LUT targets at 0 and 2.5 degrees and their componentwise linear interpolant at 1.25 degrees.',
        'boundary':'One independent precursor seed; no feedback, robustness, or cross-size superiority claim.'})

def matrix_figure(data,args):
    matrix=np.array([c['direction_mean_gain_percent'] for c in data['controllers']])
    fig=plt.figure(figsize=(WIDTH,3.65))
    ax=fig.add_axes([.235,.32,.60,.48])
    cax=fig.add_axes([.235,.12,.42,.033])
    cmap=LinearSegmentedColormap.from_list('n9_positive_gain',['#FFFFFF','#D9EAF1','#80B4CE','#337DA6','#00527D'])
    im=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=10,aspect='auto',interpolation='none')
    ax.set_xticks(range(17),[f'{d:g}' for d in data['directions_deg']],rotation=45,ha='right',rotation_mode='anchor')
    ax.set_yticks(range(7),[c['display_label'] for c in data['controllers']])
    ax.tick_params(length=0,pad=5,labelsize=8)
    ax.set_xlabel('Fixed relative direction (°)',labelpad=4)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xticks(np.arange(-.5,17,1),minor=True)
    ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='minor',length=0)
    for row,c in enumerate(data['controllers']):
        for col,value in enumerate(c['direction_mean_gain_percent']):
            ax.text(col,row,f'{value:.1f}',ha='center',va='center',fontsize=7.3,
                    color='white' if value>6.0 else DARK)
        for negative in c['negative_windows']:
            col=data['directions_deg'].index(negative['direction_deg'])
            ax.plot(col+.28,row-.27,marker='v',color=DARK,ms=3.4,clip_on=True)
    # Right-hand summaries have separate axes coordinates and retain every row.
    for y,c in enumerate(data['controllers']):
        ypos=1-(y+.5)/7
        ax.text(1.07,ypos,f"{c['equal_direction_mean_gain_percent']:.2f}",transform=ax.transAxes,
                ha='center',va='center',fontsize=8,clip_on=False)
        ax.text(1.20,ypos,str(c['negative_window_count']),transform=ax.transAxes,
                ha='center',va='center',fontsize=8,clip_on=False)
    ax.text(1.07,1.04,'Mean\n(%)',transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5)
    ax.text(1.20,1.04,'Negative\nwindows',transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5)
    ax.text(0,1.09,'a',transform=ax.transAxes,ha='left',va='bottom',fontsize=9,fontweight='bold')
    ax.text(.042,1.09,'All seven evaluated controllers',transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5)
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,2,4,6,8,10])
    cb.ax.tick_params(labelsize=8,length=2,width=.5,pad=2);cb.outline.set_visible(False)
    cb.set_label('Two-window direction-mean gain over paired zero yaw (%)',fontsize=8,labelpad=3)
    fig.text(.72,.12,'▼ One negative window\n    (start plane 0)',fontsize=7.5,ha='left',va='center')
    return save(fig,'n9_feedback_matrix',args,{
        'matrix':'Complete 7 controllers × 17 directions; each cell is the mean of two paired zero-yaw-normalized window gains.',
        'zero':'Zero-yaw and neutral-actor rows are retained; all 119 direction means are nonnegative.',
        'negative_windows':'Three negative individual windows are flagged at direction zero; start plane 0; counts include both windows at all 17 directions.',
        'right_mean':'Equal-weight arithmetic mean across all 17 direction means.',
        'color':'Sequential perceptually ordered gain scale, fixed 0–10%; numerals provide grayscale redundancy.',
        'boundary':'One precursor seed; no confidence intervals, p values, or causal feedback benefit inferred.'})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'rendered')
    ap.add_argument('--audit',type=Path,default=ROOT/'validation')
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--export',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.audit.mkdir(parents=True,exist_ok=True)
    data,validation=analyze()
    if json.loads((ROOT/'data/N9_VISUAL_DATA.json').read_text())!=data:
        raise AssertionError('derived JSON mismatch')
    jsonout(args.audit/'DATA_VALIDATION.json',validation)
    if args.profile:
        from profile_data import profile_data
        csvpath=args.audit/'N9_WINDOW_DATA.csv'
        with csvpath.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data['window_rows'][0]))
            writer.writeheader();writer.writerows(data['window_rows'])
        profile=profile_data(str(csvpath),group_cols=['controller_id','direction_deg'])
        # pandas uses NaN in non-estimable correlations; make those explicit nulls.
        def clean(v):
            if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [clean(x) for x in v]
            if isinstance(v,float) and not np.isfinite(v):return None
            return v
        jsonout(args.audit/'SCIPILOT_PROFILE.json',clean(profile))
        print(json.dumps({'status':'PROFILE_PASS','rows':238,'groups':119,'windows_per_group':2}))
    if args.preview or args.export:
        style()
        reports=[direction_figure(data,args),matrix_figure(data,args)]
        jsonout(args.audit/'FIGURE_VALIDATION.json',{'schema':'n9-visual-figure-validation/1.0',
            'status':'MACHINE_PASS_VISUAL_REVIEW_REQUIRED','reports':reports,'new_GPUh':0})
        print(json.dumps({'status':'RENDER_PASS','figures':2,'export':args.export,'new_GPUh':0}))

if __name__=='__main__':main()
