"""Safe structural decode and contracts; no JAX/LES dependency."""
import ast
import hashlib
import io
import pickle
from pathlib import Path
from typing import NamedTuple
import numpy as np

STATE_SHA='1db258cbe44cfe3eedcca2ad82c7965baba35b7f280be520dc74128d838a58be'
ACTOR_SHA='be68a370939eaecf3408af4bbbc2db84f0e9677fe46fe8f8189f933ff0a54ca5'
PREFIX_SHA='1780de4e3c66474825b4880a23f18c2bb941b2efee00d4adc5010bbb45cab7b0'
FLOW_FIELDS=('velocity','rhs_previous','rhs_previous_2','filtered_disk_speed',
             'bottom_pressure_gradient','open_x_pressure_gradient','step_index')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def ast_namedtuple(source,name):
    node=next(n for n in ast.parse(Path(source).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    ns={'NamedTuple':NamedTuple,'Array':np.ndarray,'FlowState':None,'__name__':'structural_pickle_audit'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    return ns[name]


def trusted_structural_load(path,expected_sha,code_root):
    if sha256(path)!=expected_sha:raise ValueError('trusted pickle SHA differs')
    flow=ast_namedtuple(code_root/'jax_mole_compact_sac_benchmark.py','FlowState')
    evaluation=ast_namedtuple(code_root/'jax_mole_continuous_sac.py','ContinuousEvaluationState')
    classes={('jax_mole_compact_sac_benchmark','FlowState'):flow,
             ('jax_mole_continuous_sac','ContinuousEvaluationState'):evaluation}
    class Restricted(pickle.Unpickler):
        def find_class(self,module,name):
            if (module,name) in classes:return classes[(module,name)]
            if module in {'numpy','numpy.core.numeric','numpy._core.numeric','numpy.core.multiarray','numpy._core.multiarray'} and name in {'dtype','ndarray','_frombuffer','_reconstruct','scalar'}:
                return super().find_class(module,name)
            raise pickle.UnpicklingError('unapproved scientific pickle global: '+module+'.'+name)
    with Path(path).open('rb') as f:return Restricted(f).load()


def describe(array):
    a=np.asarray(array)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),
            'all_zero':bool(np.all(a==0)),
            'sha256_c_order_bytes':hashlib.sha256(a.tobytes(order='C')).hexdigest(),
            'scalar':a.item() if a.ndim==0 else None}


def validate_saved_state(state):
    if state._fields!=('precursor_flow','farm_flow','observation_frames','observation','previous_yaw','interaction_index'):
        raise ValueError('complete saved evaluation state required')
    if state.farm_flow._fields!=FLOW_FIELDS or state.precursor_flow._fields!=FLOW_FIELDS:
        raise ValueError('complete physical history required')
    expected={
        'velocity':(1,3,496,41,483),'rhs_previous':(1,3,496,41,483),
        'rhs_previous_2':(1,3,496,41,483),'filtered_disk_speed':(1,81),
        'bottom_pressure_gradient':(1,2,496,483),'open_x_pressure_gradient':(1,2,2,41,483),
        'step_index':(),
    }
    farm={k:describe(getattr(state.farm_flow,k)) for k in FLOW_FIELDS}
    for k,shape in expected.items():
        if tuple(farm[k]['shape'])!=shape or not farm[k]['finite']:
            raise ValueError('saved physical history contract differs: '+k)
    if farm['step_index']['scalar']!=7500:raise ValueError('saved farm step differs')
    outer={k:describe(getattr(state,k)) for k in ('observation_frames','observation','previous_yaw','interaction_index')}
    shapes={'observation_frames':(1,1,6237),'observation':(1,6318),'previous_yaw':(1,81),'interaction_index':()}
    for k,shape in shapes.items():
        if tuple(outer[k]['shape'])!=shape or not outer[k]['finite']:
            raise ValueError('saved observation/controller cache differs: '+k)
        if not outer[k]['all_zero']:
            raise ValueError('actual cache differs from frozen scoring_boundary contract: '+k)
    return {'farm_history':farm,'controller_cache':outer,
            'precursor_marker':{k:describe(getattr(state.precursor_flow,k)) for k in FLOW_FIELDS},
            'first_actor_input_source':'saved state.observation bytes; neither instantaneous flow probes nor reconstructed zero cache',
            'saved_scoring_boundary_cache_is_actually_zero':True,'global_inlet_cursor':27500,'farm_step_index':7500}


def validate_actor_payload(payload):
    if payload.get('schema_version')!=1:raise ValueError('Actor schema differs')
    m=payload['metadata']
    for k,v in {'policy_architecture':'graph_sac_v2','actual_global_interactions':1000448,
                'target_turbines':81,'action_dim':81,'action_parameterization':'per_turbine',
                'learner_observation_dim':6318,'compact_solver':'dense_inverse'}.items():
        if m.get(k)!=v:raise ValueError('final Actor metadata differs: '+k)
    for k in ('reflection_equivariant_actor','direction_condition','steering_branch_condition'):
        if m.get(k,False):raise ValueError('unsupported conditioned Actor: '+k)
    leaves=[]
    def visit(x):
        if isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:leaves.append(np.asarray(x))
    visit(payload['actor'])
    if not leaves or not all(np.isfinite(x).all() for x in leaves):raise ValueError('nonfinite Actor')
    if np.asarray(payload['actor']['head']['b']).shape!=(2,):raise ValueError('shared Actor head differs')
    return {'metadata':m,'parameter_arrays':len(leaves),'all_finite':True,
            'frozen_runtime_load_actor_checkpoint_not_executed_on_local_CPU':True}
