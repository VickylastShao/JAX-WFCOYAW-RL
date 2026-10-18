"""Schematic execution figure with persistent-state/workspace separation.

All positions are publication millimetres. This is a source-grounded diagram,
not a simulation trace or a measured timing chart. CPU only.
"""
from itertools import combinations
from matplotlib.patches import Rectangle, FancyArrowPatch
from story_common import *


def build():
    height = 140
    source = ROOT / 'manuscript/main.tex'
    if source.is_file():
        source_binding = {'path':'manuscript/main.tex','sha256':sha(source),
                          'sections':['Persistent states and microbatched flow advancement',
                                      'Reference and intermediate tasks, training contexts and policy endpoint'],
                          'source_available_for_this_rebuild':True}
    else:
        saved = json.loads((VALID/'METHOD_STORY_DESIGN.json').read_text())
        expected = {'resident_contexts_per_rank':8,'active_flow_microbatch':1,
                    'controls_per_visit':8,'seconds_per_control':10,'ranks':4,
                    'local_transitions':64,'global_transitions':256,
                    'global_updates_per_collection_block':256,
                    'samples_per_rank_per_update':64,'global_samples_per_update':256,
                    'identical_replay_buffer_replicas':4,'warmup_transitions':32000}
        assert saved['scientific_values']==expected,'Frozen design parameters differ'
        source_binding = dict(saved['source'])
        source_binding['source_available_for_this_rebuild'] = False
        source_binding['binding_scope'] = 'Original source SHA from the frozen design; manuscript excluded from science-only delivery.'
    contract = {
        'status': 'FROZEN_BEFORE_DRAWING',
        'source': source_binding,
        'figure_claim': 'Persistent LES histories share one active workspace while collection and synchronized learning retain their global workload.',
        'execution': {'deliverable': 'existing method_overview PDF/SVG/PNG',
                      'renderer': 'Matplotlib native vector artists',
                      'editable_source': 'redesign/method_story.py',
                      'new_GPUh': 0},
        'scientific_values': {'resident_contexts_per_rank': 8, 'active_flow_microbatch': 1,
                              'controls_per_visit': 8, 'seconds_per_control': 10,
                              'ranks': 4, 'local_transitions': 64, 'global_transitions': 256,
                              'global_updates_per_collection_block': 256,
                              'samples_per_rank_per_update': 64, 'global_samples_per_update': 256,
                              'identical_replay_buffer_replicas': 4, 'warmup_transitions': 32000},
        'narrative_map': {
            'first_glance': 'Physical feedback above; retained histories and one workspace at the center; synchronized learner below.',
            'working': 'Load one context, run eight consecutive controls, write back immediately, then visit the next context; only after collection gather and update.',
            'technical': 'Four ranks collect 4 x 64 global transitions, insert in identical replicas and execute 256 synchronized SAC updates.',
            'forbidden_readings': ['eight concurrent flow workspaces', 'one uniquely shared replay buffer',
                                   'evaluation updates the learner', '3-to-9-to-81 strategy transfer',
                                   'common physical protocol across the three cases']},
        'visual_grammar_receipt': {
            'primary_reading_spine': 'physical loop, state/workspace distinction, training loop, case-role context',
            'dominant_focal_zone': 'eight stored histories separated from one active workspace',
            'role_to_shape': {'history': 'small numbered tile in shared residency frame',
                              'workspace': 'one outlined execution region with eight control marks',
                              'replay': 'four overlapping vector sheets',
                              'process': 'arrow with verb', 'case_role': 'detached text columns'},
            'semantic_palette': {'flow': BLUE, 'learning': ORANGE, 'context': GRAY},
            'connector_prominence_levels': ['0.85 pt forward/feedback', '0.55 pt identity/detail'],
            'typography_pt': [7.5, 8, 8.5, 9, 10],
            'whitespace_functions': ['grouping', 'connector lanes', 'focal protection'],
            'evidence_role': 'schematic only; measured evidence stays in the linked figures and tables',
            'reference_firewall': 'informed redesign of current author figure; no external topology or result copied',
            'review_sizes_mm': [178, 140],
            'forbidden_motifs': ['equal card wall', 'performance arrows across case counts', 'decorative raster']},
        'abstraction_map': {
            'complete_physical_state': 'eight C1-C8 history tiles; state member expansion stays in manuscript',
            'sequential_context_visits': 'one horizontal ordered strip, one load edge, one write-back edge',
            'collector_distribution': 'four rank rows with explicit 64 counts',
            'replay_replication': 'four sheets with explicit identity label',
            'synchronized_SAC': '256 updates box with 64 samples per rank and gradient-average label'},
        'negative_path_checks': ['No parallel edges from eight histories into eight workspaces.',
                                 'No arrow joins the three case roles.',
                                 'Learning branch is explicitly training-only; evaluation points only to panel a.'],
        'permitted_omissions': ['Individual persistent field names stay in methods; tiles preserve context identities.',
                                'Expanded graph-network internals and loss equations stay in methods.'],
    }
    dump(VALID / 'METHOD_STORY_DESIGN.json', contract)
    fig = plt.figure(figsize=(WIDTH, height/25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0,178), ylim=(0,height), xticks=[], yticks=[])
    ax.axis('off')
    objects = []
    texts = []

    def txt(gid, x, y, s, size=8, color=DARK, ha='left', weight='normal', level='L2', **kwargs):
        if gid.startswith('c-'): y += 4
        t = ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight,
                    clip_on=False,linespacing=1.10,**kwargs)
        t.set_gid(gid); objects.append((gid,level,t)); texts.append(t)
        return t

    def rect(gid, x,y,w,h,face='white',edge=GRAY,lw=.65,level='L2',**kw):
        if gid.startswith('c-'): y += 4
        p=Rectangle((x,y),w,h,facecolor=face,edgecolor=edge,lw=lw,**kw)
        p.set_gid(gid); ax.add_patch(p); objects.append((gid,level,p));return p

    def line(gid, pts, color=GRAY, lw=.65, ls='-', level='L2', **kw):
        if gid.startswith('c-'): pts=[(x,y+4) for x,y in pts]
        p,=ax.plot(*zip(*pts),color=color,lw=lw,ls=ls,clip_on=False,**kw)
        p.set_gid(gid);objects.append((gid,level,p));return p

    def arrow(gid, start,end,color=DARK,lw=.85,scale=6,ls='-',**kw):
        if gid.startswith('c-'): start=(start[0],start[1]+4);end=(end[0],end[1]+4)
        p=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=scale,lw=lw,
                          color=color,linestyle=ls,shrinkA=0,shrinkB=0,**kw)
        p.set_gid(gid);ax.add_patch(p);objects.append((gid,'L2',p));return p

    def heading(letter,y,title):
        txt('panel-'+letter+'-id',7,y,letter,10,weight='bold',level='title')
        txt('panel-'+letter+'-title',12,y,title,9,level='title')

    # a. The physical control loop is shared by training and frozen evaluation.
    heading('a',135,'81-turbine physical feedback · 10 s control interval')
    xs=[24,66,108,151]
    labels=['Observations','Shared graph actor','Yaw command','Actuator-disc LES']
    for i,(x,label) in enumerate(zip(xs,labels)):
        txt('a-label-'+str(i),x,126,label,8.5,ha='center',weight='bold',level='L1')
    for row in range(3):
        for col in range(9):
            rect(f'a-observation-{row}-{col}',10+col*3.1,114+row*2.0,2.65,1.55,
                 face='#F1F5F7',edge='#B8C5CD',lw=.30,level='L3')
    for row in range(3):
        y=114.5+row*2.1
        for col in range(3):
            x=58+col*8
            pt,=ax.plot(x,y,'o',ms=2.4,color=BLUE)
            pt.set_gid(f'a-graph-node-{row}-{col}');objects.append((pt.get_gid(),'L2',pt))
            if col<2:arrow(f'a-graph-edge-{row}-{col}',(x+1.5,y),(x+6.2,y),BLUE,.55,4)
    line('a-yaw-step',[(98,114),(104,114),(104,119),(119,119)],BLUE,1.15)
    rect('a-LES-domain',137,112.5,28,8,face='#F1F7F9',edge=BLUE,lw=.7)
    for row in range(4):
        for col in range(8):
            x=139.5+col*3.1;y=114+row*1.5
            line(f'a-disc-{row}-{col}',[(x,y),(x+1.1,y)],BLUE,.40,level='L3')
    for i,(start,end) in enumerate([(40,52),(80,94),(122,134)]):
        arrow('a-forward-'+str(i),(start,116.5),(end,116.5),DARK,.85,5.8)
    for i,(x,s) in enumerate(zip(xs,['77 probes + yaw / turbine','Shared turbine weights',
                                    r'$|\Delta\gamma|\leq10^\circ;\ |\gamma|\leq40^\circ$',
                                    '50 substeps × 0.2 s'])):
        txt('a-detail-'+str(i),x,108,s,7.5,ha='center',level='L3')
    line('a-feedback-route',[(166,116.5),(169,116.5),(169,101),(7,101),(7,116.5)],BLUE,.75)
    arrow('a-feedback-return',(7,116.5),(9,116.5),BLUE,.75,5)
    txt('a-feedback-label',88,101,'Next interval-averaged probes; modeled power − yaw penalty → reward',7.5,
        color=BLUE,ha='center',bbox={'facecolor':'white','edgecolor':'none','pad':1.3})

    # b. Residency and the active flow workspace have visibly different shapes.
    heading('b',94,'Persistent histories and active workspace · one GPU rank')
    rect('b-history-residency',8,65.5,71,22.0,face='#F7F9FA',edge='#9EADB6',lw=.65)
    txt('b-history-heading',43.5,84,'8 persistent histories  (E = 8)',8.5,ha='center',weight='bold',level='L1')
    for i in range(8):
        x=11.5+(i%4)*16.5;y=74.4 if i<4 else 67.5
        rect('b-history-tile-'+str(i+1),x,y,13.5,5.1,face='white',edge='#AFBBC3',lw=.45)
        txt('b-history-id-'+str(i+1),x+6.75,y+2.55,'C'+str(i+1),8,ha='center',weight='bold')
    rect('b-active-workspace',108,68,60,19.5,face='#F2F8FA',edge=BLUE,lw=.95,level='L1')
    txt('b-active-heading',138,83,'One active workspace  (b = 1)',8.5,color=BLUE,ha='center',weight='bold',level='L1')
    txt('b-selected-context',138,77.5,'Selected context → 8 controls',8,ha='center')
    for k in range(8):
        rect('b-control-cell-'+str(k),114+k*5.8,70,5.2,3.1,face=BLUE,edge='white',lw=.3)
    arrow('b-load',(80.5,78.5),(106.5,78.5),BLUE,.90,6)
    txt('b-load-label',93.5,82.4,r'Load $C_i$',7.5,ha='center',color=BLUE)
    line('b-write-back-route',[(108,70),(104.5,70),(104.5,66)],BLUE,.75)
    arrow('b-write-back',(104.5,66),(79,66),BLUE,.75,5.5,
          connectionstyle='arc3,rad=0')
    txt('b-write-back-label',94,61.9,r'Write back $C_i$',7.5,ha='center',color=BLUE)
    txt('b-consecutive-interval',138,62.1,'8 consecutive × 10 s',7.5,ha='center')
    txt('b-identity-note',8,59.1,'Histories stay resident',7.5,color=GRAY,level='L3')
    txt('b-visit-order-label',8,53.5,'Visit order:',7.5)
    for i in range(8):
        x=35+i*17.2
        txt('b-visit-order-'+str(i+1),x,53.5,'C'+str(i+1),8,ha='center',color=BLUE)
        if i<7:arrow('b-visit-next-'+str(i+1),(x+4.2,53.5),(x+12.4,53.5),BLUE,.65,4.8)

    # c. Learner traffic is strictly downstream of the completed collection.
    heading('c',49,'Fixed global learning work · training only')
    txt('c-warmup',8,40,'After the 32,000-transition warmup',7.5,color=GRAY,level='L3')
    for i in range(4):
        y=34.7-i*3.2
        txt('c-rank-label-'+str(i),8,y,'Rank '+str(i),7.5)
        rect('c-rank-batch-'+str(i),23,y-1.25,17.5,2.5,face='#DCECF2',edge=BLUE,lw=.4)
        txt('c-rank-count-'+str(i),44,y,'64',7.5,ha='center',color=BLUE)
    txt('c-global-workload',8,20,'4 × 64 = 256 transitions',8,weight='bold',level='L1')
    arrow('c-all-gather',(49,30),(63.5,30),ORANGE,.9,6)
    txt('c-all-gather-label',56,35.5,'All-gather',7.5,ha='center',color=ORANGE)
    for i in range(3,-1,-1):
        rect('c-replay-replica-'+str(i),68+i*.95,25+i*.9,33.5,11,
             edge=ORANGE,face='white',lw=.6)
    txt('c-replay-title',84.75,30.8,'Replay buffer × 4',8,ha='center',color=ORANGE,weight='bold',level='L1')
    txt('c-replay-identity',85,21,'Identical replicas',7.5,ha='center')
    arrow('c-buffer-to-updates',(105.5,30),(116.8,30),ORANGE,.9,6)
    rect('c-updates',119,25,49,12,face='#FCF7F2',edge=ORANGE,lw=.65)
    txt('c-update-count',143.5,33.2,'256 synchronized SAC updates',8,ha='center',color=ORANGE,weight='bold',level='L1')
    txt('c-samples',143.5,28.3,'64 samples / rank / update',7.5,ha='center')
    txt('c-gradient-average',143.5,21,'Gradient average across ranks',7.5,ha='center',color=ORANGE)
    # A short update return names its destination rather than crossing panel b.
    line('c-updated-actor-route',[(168,30),(172,30),(172,17),(117,17)],ORANGE,.75,ls='--')
    arrow('c-updated-actor',(117,17),(106,17),ORANGE,.75,5.5,ls='--')
    txt('c-updated-actor-label',103,17,'Updated actor → next collection block',7.5,
        color=ORANGE,ha='right')
    txt('c-frozen-evaluation',8,12.7,'Frozen evaluation uses physical loop (a).',7.5,color=GRAY,level='L3')

    # Detached role navigation: deliberately no scaling/transfer connectors.
    line('d-case-divider',[(8,13.8),(168,13.8)],'#CBD3D8',.5,level='L3')
    for i,(x,title,body) in enumerate([
        (8,'3 turbines','Reference implementation / learner execution'),
        (63,'9 turbines','Graph-policy learning / independent-source directions'),
        (122,'81 turbines','Completed LES training / execution + B/C')]):
        txt('d-case-title-'+str(i),x,10.8,title,8,weight='bold',level='L2')
        # Two compact lines keep role text within the 53-58 mm column widths.
        body_lines=[['Reference implementation','Learner execution'],
                    ['Graph-policy learning','17 directions; independent inflow'],
                    ['Completed LES training','Execution + B/C']][i]
        txt('d-case-role-'+str(i),x,5.1,'\n'.join(body_lines),7.5,level='L3')

    # Machine checks are geometric, and do not stand in for visual review.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    overlaps=[]
    for a,b in combinations(texts,2):
        aa=a.get_window_extent(renderer);bb=b.get_window_extent(renderer)
        xover=min(aa.x1,bb.x1)-max(aa.x0,bb.x0)
        yover=min(aa.y1,bb.y1)-max(aa.y0,bb.y0)
        if xover>1 and yover>1:overlaps.append([a.get_gid(),b.get_gid(),xover,yover])
    dump(VALID/'METHOD_STORY_LAYOUT.json',{'text_text_overlap_pairs':overlaps,
          'renderer':'Matplotlib Agg','publication_size_mm':[178,height],
          'status':'PASS' if not overlaps else 'FAIL'})
    if overlaps:raise RuntimeError(('text overlap',overlaps))
    hidden=[]
    for gid,level,obj in objects:
        if level in {'title','L3'}:obj.set_visible(False);hidden.append(gid)
    hidden_path=PREVIEW/'METHOD_STORY_titles_L3_hidden.png'
    fig.savefig(hidden_path,dpi=180)
    with Image.open(hidden_path) as im:ImageOps.grayscale(im).save(PREVIEW/'METHOD_STORY_titles_L3_hidden_grayscale.png')
    for gid,level,obj in objects:obj.set_visible(True)
    payload={'schematic_not_empirical_data':True,'contract_sha256':sha(VALID/'METHOD_STORY_DESIGN.json'),
             **contract['scientific_values'],
             'per_visit_order':'load context, eight consecutive controls, immediate write-back to that context',
             'per_block_order':'all eight visits, all-gather, replica insertion, 256 synchronized updates',
             'invariant':'Each rank retains all eight histories, with only one active flow workspace.',
             'cases_separately_trained':True,'case_role_connections':0,
             'negative_path_checks':contract['negative_path_checks']}
    review_path=VALID/'METHOD_STORY_PREVIEW_REVIEW.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else None
    rebuilding = os.environ.get('WIND_FIG_REBUILD')=='1'
    if os.environ.get('WIND_FIG_FINAL')=='1' and not rebuilding:
        if not review or review['status']!='PASS_INFORMED_AUTHOR_VISUAL_REVIEW':
            raise RuntimeError('Final export requires recorded actual preview inspection.')
        if review['renderer_script_sha256']!=sha(Path(__file__)):
            raise RuntimeError('Preview review does not bind the current renderer script.')
        for item in review['inspected_previews']:
            if sha(ROOT/item['path'])!=item['sha256']:
                raise RuntimeError(('Inspected preview changed',item['path']))
    finish(fig,'method_overview',payload,[
        'Replace eight long diagonal context lanes with a distinct residency frame, one active workspace and a short ordered visit strip.',
        'Expose fixed E, b, local/global transition counts, replay replication and synchronized learner updates in separate visual groups.',
        'Keep physical feedback separate from training-only learning and frozen evaluation.',
        'Add detached case-role navigation without scaling or transfer arrows.',
        'Shorten height to 140 mm at the same 178 mm publication width, retaining at least 7.5 pt labels.'])
    dump(VALID/'METHOD_STORY_VALIDATION.json',{
        'status':('REBUILT_REQUIRES_LOCAL_VISUAL_REVIEW' if rebuilding else 'FINAL_EXPORT_AWAITING_ROOT_READER_REVIEW') if os.environ.get('WIND_FIG_FINAL')=='1' else 'PREVIEW_AWAITING_VISUAL_REVIEW',
        'artifact':'method_overview','scientific_reading':payload,
        'source':contract['source'],'original_scientific_data_modified':False,
        'no_added_performance_or_transfer_claim':True,'new_GPUh':0,
        'schematic_not_physical_data':True,'renderer':'Matplotlib Agg native vectors',
        'live_PowerPoint_or_drawio_acceptance_claimed':False,
        'author_figure_design_not_peer_review':True,
        'text_overlap_check':{'checked_visible_text_pairs':len(texts)*(len(texts)-1)//2,'overlaps':overlaps},
        'review_variant':str((PREVIEW/'METHOD_STORY_titles_L3_hidden_grayscale.png').relative_to(ROOT)),
        'review_hidden_objects':hidden,
        'visual_review':({'status':'NOT_REPEATED_FOR_THIS_REBUILD','original_review_receipt':'validation/METHOD_STORY_PREVIEW_REVIEW.json'} if rebuilding else review) or 'Pending actual preview inspection; no self-assigned publication certificate.',
        'root_reader_PDF_review':'Pending independent root-side PDF renderer inspection.'})


if __name__=='__main__':
    build()
