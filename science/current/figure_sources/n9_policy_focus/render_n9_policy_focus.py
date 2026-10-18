#!/usr/bin/env python3
"""Render the N9 policy response from unchanged archived results, CPU only.

Run --profile first, inspect --preview, then use --export. Fonts are registered
for this Python process only. No environment installation is performed.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
from visual_qa import audit_layout
from check_figure import check_figure
from validate_n9_visual_data import analyze

NAME = 'n9_policy_direction_response'
BLUE = '#0072B2'
DARK = '#202A35'
GRAY = '#6B7280'
WIDTH_MM, HEIGHT_MM = 178., 70.

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jout(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def style():
    for variant in ['Regular', 'Bold', 'Italic', 'BoldItalic']:
        font_manager.fontManager.addfont(str(ROOT/f'fonts/STIXTwoText/STIXTwoText-{variant}.otf'))
    plt.rcParams.update({
        'font.family': ['STIX Two Text', 'STIXGeneral'],
        'font.serif': ['STIX Two Text'], 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#7B8188', 'xtick.color': DARK,
        'ytick.color': DARK, 'text.color': DARK, 'axes.labelcolor': DARK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'svg.hashsalt': 'n9_policy_focus_20261016',
    })

def profile(data, audit):
    from profile_data import profile_data
    out = audit/'N9_WINDOW_DATA.csv'
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['window_rows'][0]))
        writer.writeheader(); writer.writerows(data['window_rows'])
    result = profile_data(str(out), group_cols=['controller_id', 'direction_deg'])
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    jout(audit/'SCIPILOT_PROFILE.json', clean(result))
    policy = next(c for c in data['controllers'] if c['controller_id']=='selected_realtime_policy')
    jout(audit/'POLICY_PROFILE.json', {
        'status': 'PROFILE_PASS', 'full_raw_window_rows': len(data['window_rows']),
        'displayed_controller': 'selected_realtime_policy',
        'directions': 17, 'displayed_window_values': 34,
        'training_anchor_directions': 9, 'intermediate_directions': 8,
        'independent_precursor_seeds': 1, 'windows_per_direction': 2,
        'window_gain_range_percent': [float(np.min(policy['window_gain_percent_by_direction'])), float(np.max(policy['window_gain_percent_by_direction']))],
        'all_policy_windows_positive': bool(np.all(np.array(policy['window_gain_percent_by_direction'])>0)),
        'no_missing_data_or_outlier_removal': True,
        'no_inferential_interval': True, 'new_GPUh': 0,
    })

def draw(data, args):
    x = np.array(data['directions_deg'])
    policy = next(c for c in data['controllers'] if c['controller_id']=='selected_realtime_policy')
    means = np.array(policy['direction_mean_gain_percent'])
    windows = np.array(policy['window_gain_percent_by_direction'])
    anchor = np.array([angle/2.5==int(angle/2.5) for angle in x])
    assert len(x)==17 and windows.shape==(17,2) and int(anchor.sum())==9
    fig, ax = plt.subplots(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
    fig.subplots_adjust(left=.085, right=.985, bottom=.205, top=.81)
    ax.axhline(0, color=GRAY, lw=.75, ls=(0,(3,2)), zorder=1)
    ax.plot(x, means, color=BLUE, lw=.9, zorder=2)
    for column, dx, marker in [(0,-.085,'x'),(1,.085,'+')]:
        ax.scatter(x+dx, windows[:,column], color=GRAY, marker=marker,
                   s=13, linewidths=.7, zorder=3)
    ax.scatter(x[anchor], means[anchor], color=BLUE, marker='o',
               s=20, edgecolors=BLUE, linewidths=.7, zorder=4)
    ax.scatter(x[~anchor], means[~anchor], facecolors='white', edgecolors=BLUE,
               marker='D', s=21, linewidths=.9, zorder=4)
    ax.set_xlim(-10.55,10.55); ax.set_ylim(-.40,8.65)
    ax.set_xticks(x, [f'{v:g}' for v in x]); ax.set_yticks([0,2,4,6,8])
    ax.set_xlabel('Fixed relative direction (°)', labelpad=6)
    ax.set_ylabel('Gain over paired zero yaw (%)', labelpad=6)
    ax.grid(axis='y', color='#E5E7EB', linewidth=.55); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)
    ax.text(.985,.065,'Zero-yaw baseline', transform=ax.transAxes,
            ha='right', va='center', fontsize=8, color=GRAY)
    handles = [
        Line2D([],[], marker='o', ls='none', color=BLUE, markersize=4,
               label='Mean: training direction'),
        Line2D([],[], marker='D', ls='none', color=BLUE, markerfacecolor='white',
               markersize=4, label='Mean: intermediate direction'),
        Line2D([],[], marker='x', ls='none', color=GRAY, markersize=4,
               label='Window: offset 0'),
        Line2D([],[], marker='+', ls='none', color=GRAY, markersize=4,
               label='Window: offset 40000'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.535,.975),
               ncol=4, frameon=False, handletextpad=.45, columnspacing=1.15,
               handlelength=1, borderaxespad=0, fontsize=8)
    fig.canvas.draw()
    issues = [list(issue) for issue in audit_layout(fig)]
    if any(issue[0]=='FAIL' for issue in issues): raise AssertionError(issues)
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    sizes = sorted(set(t.get_fontsize() for t in texts)); assert min(sizes)>=8
    renderer = fig.canvas.get_renderer(); x0,y0,x1,y1=fig.bbox.extents
    outside=[]
    for text in texts:
        bounds=text.get_window_extent(renderer)
        if bounds.x0<x0-1 or bounds.x1>x1+1 or bounds.y0<y0-1 or bounds.y1>y1+1:
            outside.append({'text':text.get_text(),'bounds_px':list(bounds.extents)})
    if outside: raise AssertionError(('out-of-canvas text',outside))
    artists=[]
    for line in ax.lines:
        artists.append({'kind':'line','x':np.asarray(line.get_xdata(),dtype=float).tolist(),
                        'y':np.asarray(line.get_ydata(),dtype=float).tolist()})
    for collection in ax.collections:
        artists.append({'kind':'points','offsets':np.asarray(collection.get_offsets()).tolist()})
    assert len(ax.collections)==4
    assert len(artists[2]['offsets'])==17 and len(artists[3]['offsets'])==17
    assert len(artists[4]['offsets'])==9 and len(artists[5]['offsets'])==8
    jout(args.audit/f'{NAME}_ARTISTS.json',artists)
    artist_json=json.dumps(artists,sort_keys=True,separators=(',',':'),allow_nan=False)
    preview=args.audit/f'{NAME}_preview.png'; fig.savefig(preview,dpi=180)
    with Image.open(preview) as image:
        ImageOps.grayscale(image).save(args.audit/f'{NAME}_grayscale_preview.png',dpi=(180,180))
    report={
        'schema':'n9-policy-focus-figure-validation/1.0',
        'status':'MACHINE_PASS_VISUAL_REVIEW_REQUIRED',
        'figure':NAME, 'width_mm':WIDTH_MM, 'height_mm':HEIGHT_MM,
        'font_sizes_pt':sizes,'font_family':'STIX Two Text; STIXGeneral symbol fallback',
        'resolved_font':str(Path(font_manager.findfont('STIX Two Text')).relative_to(ROOT)),
        'scipilot_layout_checks':issues,'out_of_canvas_text':outside,
        'artist_sha256':hashlib.sha256(artist_json.encode()).hexdigest(),
        'displayed_controller':'selected_realtime_policy',
        'direction_mean_count':17,'window_point_count':34,
        'training_mean_count':9,'intermediate_mean_count':8,
        'semantics':{
            'line':'Guide between ordered fixed-direction means; not a within-trajectory wind series.',
            'markers':'Filled circles: means at training directions. Open diamonds: means at interleaved directions.',
            'window_points':'All 34 archived windows, at inlet offsets 0 and 40000; horizontal offsets of -0.085 and +0.085 degrees are visual-only.',
            'baseline':'Zero percent is the paired zero-yaw reference, not a second evaluated curve.',
            'statistical_scope':'One independent precursor seed, two nested deterministic windows per fixed direction; no inferential intervals.',
            'other_controls':'All 238 rows remain in the source validation; the complete seven-controller comparison is supplementary.'},
        'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,
                   'matplotlib':matplotlib.__version__,'Pillow':Image.__version__},
        'new_GPUh':0,
    }
    if args.export:
        exports=[]
        for extension in ['pdf','svg','png']:
            path=args.output/f'{NAME}.{extension}'
            metadata={'Creator':'Source-checked N9 policy direction response','CreationDate':None,'ModDate':None} if extension=='pdf' else {}
            fig.savefig(path,dpi=600,**({'metadata':metadata} if metadata else {}))
            checks, info=check_figure(str(path),min_dpi=300,target_inches=tuple(fig.get_size_inches()))
            if any(check[0]=='FAIL' for check in checks): raise AssertionError((extension,checks))
            exports.append({'filename':path.name,'bytes':path.stat().st_size,
                            'sha256':sha(path),'checks':[list(check) for check in checks],'info':info})
        with Image.open(args.output/f'{NAME}.png') as image:
            ImageOps.grayscale(image).save(args.audit/f'{NAME}_grayscale.png',dpi=(600,600))
        report['exports']=exports
    jout(args.audit/f'{NAME}_LAYOUT.json',report)
    jout(args.audit/'FIGURE_VALIDATION.json',{'schema':'n9-policy-focus-render/1.0',
        'status':'MACHINE_PASS_VISUAL_REVIEW_REQUIRED','reports':[report],'new_GPUh':0})
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'rendered')
    parser.add_argument('--audit',type=Path,default=ROOT/'validation')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True);args.audit.mkdir(parents=True,exist_ok=True)
    data,validation=analyze()
    assert json.loads((ROOT/'data/N9_VISUAL_DATA.json').read_text())==data
    jout(args.audit/'DATA_VALIDATION.json',validation)
    if args.profile: profile(data,args.audit)
    if args.preview or args.export:
        style(); draw(data,args)
    print(json.dumps({'status':'PASS','profile':args.profile,'preview':args.preview,
                      'export':args.export,'new_GPUh':0}))

if __name__=='__main__':main()
