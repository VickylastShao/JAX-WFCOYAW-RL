"""Finite, identity-checked source -> archive -> E32 -> N81-R1 workflow.

Only GPU0-3 are used. Archive blocks exit after atomic checkpoints, allowing
N81 priority without interrupting an LES step or changing archive contracts.
"""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import threading

from run_mole_large_scale_initial_stage import Stage, IMAGE, atomic, command


def replace_arg(argv, name, value):
    result = list(argv)
    if name in result:
        result[result.index(name) + 1] = str(value)
    else:
        result.extend([name, str(value)])
    return result


def engineering_argv(original, rank, resume=False):
    argv = list(original)
    values = {'--coordinator': '127.0.0.1:17481', '--seed': 28100181,
              '--iterations': 63 if resume else 1,
              '--planned-global-interactions': 16640 if resume else 512,
              '--minimum-replay-size': 256, '--random-action-transitions': 256,
              '--checkpoint-milestones': '16640' if resume else '512',
              '--full-checkpoint-interval': 2048,
              '--checkpoint': '/batch/engineering/n81/%s_actor.pkl' % ('resume' if resume else 'fresh'),
              '--full-checkpoint': '/batch/engineering/n81/%s_full_state.pkl' % ('resume' if resume else 'fresh'),
              '--stop-request': '/batch/engineering/n81/STOP',
              '--output': '/batch/engineering/n81/%s_rank%d.json' % ('resume' if resume else 'fresh', rank),
              '--stop-after-seconds': 13800}
    for key, value in values.items():
        argv = replace_arg(argv, key, value)
    if resume:
        argv.remove('--save-initial-checkpoint')
        argv += ['--resume-checkpoint', '/batch/engineering/n81/fresh_full_state.process%d.pkl' % rank]
    return argv


def assess_engineering(results):
    checks = {}
    for rank, (fresh, resumed) in enumerate(results):
        fc, rc = fresh['configuration'], resumed['configuration']
        hist = resumed['history']
        checks['rank%d_fresh_complete' % rank] = fresh['status'] == 'pass' and fc['total_global_transitions_including_compile_warmup'] == 512
        checks['rank%d_resume_complete' % rank] = resumed['status'] == 'pass' and rc['total_global_transitions_including_compile_warmup'] == 16640
        checks['rank%d_full_resume' % rank] = rc['resumed_global_interactions'] == 512 and rc['resumed_from_full_checkpoint'] is not None
        checks['rank%d_reset_exercised' % rank] = any(r.get('physical_reset_interactions_per_environment', 0) > 0 for r in hist)
        checks['rank%d_actor_updates' % rank] = rc['sac_actor_updates_applied_total'] > 0
        checks['rank%d_nonzero_actor_gradient' % rank] = any(r.get('sac_actor_gradient_norm', 0) > 0 for r in hist)
        checks['rank%d_actions_effective' % rank] = any(r.get('mean_abs_yaw_deg', 0) > 0 for r in hist)
        checks['rank%d_internal_checks' % rank] = all(resumed['checks'].values()) and all(fresh['checks'].values())
        checks['rank%d_same_capacity' % rank] = rc['replay_capacity_per_host'] == 262144 and rc['local_environments'] == 8
    return checks


def estimate_formal_budget(results):
    predictions = []
    for fresh, resumed in results:
        rows = resumed['history']
        regular = [r['wallclock_seconds'] for r in rows if not r.get('physical_reset_interactions_per_environment', 0)]
        if len(regular) < 10:
            raise ValueError('insufficient measured warm blocks for budget')
        regular.sort()
        p90 = regular[math.ceil(.9 * len(regular)) - 1]
        reset_extra = max([max(0., r['wallclock_seconds'] - p90) for r in rows if r.get('physical_reset_interactions_per_environment', 0)] or [0.])
        io = [r['checkpoint_io_seconds'] for r in resumed.get('rolling_checkpoint_records', [])]
        if not io:
            raise ValueError('missing actual full checkpoint timing')
        cold = fresh['timing']['total_process_wallclock_seconds']
        predicted = cold + 3908*p90 + 62*reset_extra + (math.ceil(1000448/2048) + 6)*max(io)
        predictions.append({'block_p90_seconds': p90, 'reset_extra_seconds': reset_extra,
                            'checkpoint_io_max_seconds': max(io), 'predicted_wall_seconds': predicted})
    seconds = max(x['predicted_wall_seconds'] for x in predictions)
    return {'rank_predictions': predictions, 'predicted_wall_seconds': seconds,
            'allocated_gpu_hour_limit': 1.25*seconds*4/3600,
            'wall_seconds_limit': 1.25*seconds,
            'scope': 'Measured real E32 reset/update/IO forecast; resource cap, no reward gate.'}


class Workflow(Stage):
    def __init__(self, batch, legacy):
        super().__init__(batch, legacy)
        self.n81_ready = threading.Event()
        self.phase = 'source_and_archive_preparation'
        self.cpu_jobs = {}
        self.formal_started = False

    def status(self, label):
        super().status(label)
        if time.monotonic() - self.started >= self.plan['wall_seconds_limit']:
            self.cancel.set()
        if getattr(self, 'phase', '') == 'real_n81_E32_engineering':
            consumed = self.gpu_seconds - getattr(self, 'phase_started_gpu_seconds', self.gpu_seconds)
            if consumed >= self.plan['engineering_gpu_hours_limit'] * 3600:
                self.cancel.set()
        p = self.batch / 'STATUS.json'
        d = json.loads(p.read_text())
        d.update(phase=getattr(self, 'phase', 'initializing'),
                 training_started=getattr(self, 'formal_started', False))
        atomic(p, d)

    def cpu(self, name, argv, timeout=3600):
        name_full = 'shao-large-' + self.task[:10] + '-' + name
        args = ['docker', 'run', '-d', '--name', name_full,
                '--label', 'research.task=' + self.task,
                '--user', '%d:%d' % (os.getuid(), os.getgid()), '--memory', '8g',
                '-e', 'JAX_PLATFORMS=cpu', '-e', 'PYTHONPATH=/project/codes',
                '-v', str(self.batch / 'stage') + ':/project:ro',
                '-v', str(self.batch) + ':/batch:rw', '-w', '/project', IMAGE]
        cid = command(args + argv)
        started = time.monotonic()
        with self.lock:
            self.cpu_jobs[name] = {'container_id': cid, 'started_unix': time.time()}
            atomic(self.batch / 'CPU_JOBS.json', self.cpu_jobs)
        while True:
            state = self.inspect(cid)['State']
            if not state['Running']:
                break
            if self.cancel.is_set() or time.monotonic()-started > timeout:
                self.inspect(cid)
                command(['docker', 'stop', '--time', '25', cid])
                raise RuntimeError('bounded CPU job stopped: ' + name)
            with self.lock:
                self.status('running')
            time.sleep(5)
        (self.batch / 'logs' / (name + '.log')).write_text(command(['docker', 'logs', cid]))
        with self.lock:
            self.cpu_jobs[name].update(exit_code=state['ExitCode'], elapsed_seconds=time.monotonic()-started)
            atomic(self.batch / 'CPU_JOBS.json', self.cpu_jobs)
        if state['ExitCode'] or state['OOMKilled']:
            raise RuntimeError('CPU job failed: ' + name)

    def preparation_hours(self):
        with self.lock:
            jobs = list(self.jobs.items())
        return sum(j.get('elapsed_seconds', time.time()-j['started_unix']) for name, j in jobs
                   if name.startswith(('source-', 'archive-'))) / 3600

    def archive(self, spec, gpu, priority=True):
        sid = spec['source_id']
        archive = self.batch / 'archives' / sid
        while True:
            manifest = archive / 'manifest.json'
            done = json.loads(manifest.read_text())['completed_planes'] if manifest.exists() else 0
            if done == spec['target_planes']:
                receipt = archive / 'VERIFIED.json'
                if not receipt.exists():
                    self.cpu('verify-' + sid, ['python', 'codes/verify_mole_large_scale_archive.py',
                        '--archive', '/batch/archives/' + sid, '--target-turbines', str(spec['turbines']),
                        '--output', '/batch/archives/' + sid + '/VERIFIED.json'])
                if sid == 'n81_train_A':
                    self.n81_ready.set()
                return True
            if priority and sid == 'n100_train_A' and self.n81_ready.is_set():
                atomic(self.batch / 'N100_ARCHIVE_YIELDED.json', {'completed_planes': done,
                    'source_id': sid, 'reason': 'N81 real E32 and R1 priority; resume same immutable archive afterwards'})
                return False
            if self.cancel.is_set() or self.preparation_hours() >= self.plan['preparation_gpu_hours_limit']:
                raise RuntimeError('preparation GPUh budget reached at archive block boundary')
            prefix_name = 'archive-' + sid + '-'
            with self.lock:
                spent = sum(j.get('elapsed_seconds', 0) for n,j in self.jobs.items() if n.startswith(prefix_name))/3600
            if spent >= spec['archive_gpu_hour_cap']:
                raise RuntimeError('archive source budget reached: ' + sid)
            if shutil.disk_usage(self.batch).free < 100*1024**3:
                raise RuntimeError('reserve disk exhausted; archive checkpoints retained')
            argv = ['python', '-u', 'codes/archive_mole_large_scale_block.py',
                    '--source-state', '/batch/qualified_sources/' + sid + '/state.npz',
                    '--archive-dir', '/batch/archives/' + sid, '--seed', str(spec['seed']),
                    '--target-turbines', str(spec['turbines']), '--profile', 'article_224d_inferred',
                    '--target-planes', str(spec['target_planes']), '--chunk-planes', '20000',
                    '--members', '8', '--first-offset-steps', '20000', '--offset-steps', '20000',
                    '--max-chunks', '1', '--minimum-free-gib-after', '100']
            self.run_job(prefix_name + str(done//20000), gpu, argv, 7200)

    def source_and_archive(self, spec, gpu):
        sid = spec['source_id']
        destination = self.batch / 'qualified_sources' / sid
        destination.mkdir(parents=True, exist_ok=True)
        for suffix in ('state.npz', 'state.meta.json'):
            original = Path(spec['original_state']).parent / suffix
            target = destination / suffix
            if target.exists():
                raise FileExistsError('new workflow source already exists: ' + str(target))
            shutil.copy2(original, target)
        if self.cancel.is_set():
            raise RuntimeError('cancelled after preserving source copy')
        if spec['saved_steps'] < 200000:
            self.run_job('source-' + sid, gpu, ['python', '-u', 'codes/generate_mole_precursor_resumable.py',
                '--state', '/batch/qualified_sources/' + sid + '/state.npz', '--output', '/batch/qualified_sources/' + sid + '/continuation.json',
                '--batch', '1', '--seed', str(spec['seed']), '--target-steps', '200000', '--chunk-steps', '20000',
                '--profile', 'article_224d_inferred', '--target-turbines', str(spec['turbines']),
                '--compact-solver', 'dense_inverse', '--compact-stdout'], 7200)
            meta = json.loads((destination / 'state.meta.json').read_text())
            updated = dict(spec, original_state='/batch/qualified_sources/' + sid + '/state.npz',
                           expected_sha256=meta['npz_sha256'], saved_steps=200000)
            audit_plan = self.batch / ('VERIFY_' + sid + '.json')
            atomic(audit_plan, {'sources': [updated]})
            self.cpu('verify-source-' + sid, ['python', 'codes/verify_mole_precursor_state.py',
                '--plan', '/batch/' + audit_plan.name, '--output', '/batch/qualified_sources/' + sid + '/VERIFIED.json'], 1800)
            result = json.loads((destination / 'VERIFIED.json').read_text())
            if result['status'] != 'pass':
                raise ValueError('resumed source failed full-field verification')
        atomic(destination / 'ADMISSION.json', {'source_id': sid, 'admitted_for_archive': True,
            'source_sha256': json.loads((destination / 'state.meta.json').read_text())['npz_sha256'],
            'statistical_stationarity_proven': False,
            'basis': 'Verified finite state, matching source physics and geometry; source physical assessment recorded, temporal statistics collected during archive generation.'})
        return self.archive(spec, gpu)

    def gpu_queue(self, gpu, ids):
        specs = {s['source_id']:s for s in self.plan['sources']}
        for sid in ids:
            if self.cancel.is_set():
                raise RuntimeError('workflow cancelled')
            self.source_and_archive(specs[sid], gpu)

    def group(self, label, argv_by_rank, timeout):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.run_job, label + '-rank' + str(rank), rank, argv, timeout)
                       for rank, argv in enumerate(argv_by_rank)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    self.cancel.set()
                    raise
        atomic(self.batch / (label + '_GROUP.json'), {'status':'all_ranks_exit_zero',
               'elapsed_seconds':time.monotonic()-started, 'gpu_ids':[0,1,2,3]})

    def execute(self):
        (self.batch / 'logs').mkdir(exist_ok=True)
        verification = json.loads((self.batch / 'SOURCE_VERIFICATION.json').read_text())
        if verification['status'] != 'pass' or len(verification['results']) != 6:
            raise ValueError('six source verification receipts required')
        admission = json.loads((self.batch / 'SOURCE_PHYSICAL_ASSESSMENT.json').read_text())
        if not admission['admit_for_archive_and_engineering']:
            raise ValueError('physical source admission missing')
        if shutil.disk_usage(self.batch).free < self.plan['minimum_free_bytes']:
            raise ValueError('entire planned archive payload plus reserve does not fit')
        queues = {0:['n81_train_A'],1:['n81_heldout_B','n81_heldout_C'],
                  2:['n100_train_A'],3:['n100_heldout_B','n100_heldout_C']}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.gpu_queue, gpu, ids) for gpu,ids in queues.items()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    self.cancel.set()
                    raise
        self.phase = 'real_n81_E32_engineering'
        self.phase_started_gpu_seconds = self.gpu_seconds
        self.status('running')
        recipe = json.loads((self.batch / 'COMMAND_RECIPES_APPROVED.json').read_text())['training_runs'][0]
        originals = [r['argv'] for r in recipe['ranks']]
        self.group('engineering-fresh', [engineering_argv(a,r) for r,a in enumerate(originals)], 14400)
        self.group('engineering-resume', [engineering_argv(a,r,True) for r,a in enumerate(originals)], 14400)
        results = [(json.loads((self.batch/'engineering/n81'/('fresh_rank%d.json'%r)).read_text()),
                    json.loads((self.batch/'engineering/n81'/('resume_rank%d.json'%r)).read_text())) for r in range(4)]
        checks = assess_engineering(results)
        engineering_gpu_hours = sum(j.get('elapsed_seconds',0) for n,j in self.jobs.items() if n.startswith('engineering-'))*1./3600
        checks['engineering_budget'] = engineering_gpu_hours <= self.plan['engineering_gpu_hours_limit']
        atomic(self.batch/'ENGINEERING_ACCEPTANCE.json', {'status':'pass' if all(checks.values()) else 'fail',
               'checks':checks,'allocated_gpu_hours':engineering_gpu_hours,
               'boundary':'Real E32 capacity/update/reset/full resume. Engineering warmup=256; not a scientific Actor or reward result.'})
        if not all(checks.values()):
            raise ValueError('engineering acceptance failed; formal training not started')
        budget = estimate_formal_budget(results)
        atomic(self.batch/'FORMAL_N81_BUDGET.json', budget)
        if budget['allocated_gpu_hour_limit'] > self.plan['formal_n81_gpu_hours_hard_limit']:
            raise ValueError('measured formal budget exceeds frozen outer cap; checkpoints preserved')
        formal = []
        for a in originals:
            a = replace_arg(a,'--coordinator','127.0.0.1:17491')
            a = replace_arg(a,'--stop-after-seconds',max(60.,budget['wall_seconds_limit']-900))
            formal.append(a)
        atomic(self.batch/'FORMAL_N81_LAUNCH_CONTRACT.json', {'run_id':'n81_train_r1','seed':27100181,
               'global_interactions':1000448,'argv_by_rank':formal,'budget':budget,
               'from_scratch':True,'engineering_checkpoint_not_used':True,'main_actor':'fixed_budget_final'})
        self.phase = 'formal_n81_R1'
        self.formal_started = True
        self.status('running')
        self.group('formal-n81-r1', formal, budget['wall_seconds_limit']+300)
        final = [json.loads((self.batch/'runs/n81_train_r1'/('rank%d.json'%r)).read_text()) for r in range(4)]
        if not all(x['status']=='pass' and x['declared_budget_completed'] for x in final):
            raise ValueError('formal fixed budget incomplete; do not declare convergence')
        atomic(self.batch/'N81_R1_COMPLETE.json', {'status':'fixed_budget_training_complete',
               'global_interactions':1000448,'seed':27100181,'positive_independent_gain_confirmed':False,
               'actor':'/batch/runs/n81_train_r1/actor.pkl'})
        self.phase = 'finish_n100_A_archive'
        spec = next(s for s in self.plan['sources'] if s['source_id']=='n100_train_A')
        self.archive(spec,2,priority=False)
        self.phase = 'complete'
        self.status('complete')
        atomic(self.batch/'WORKFLOW_RESULT.json', {'status':'complete','training_started':True,
               'n81_R1_fixed_budget_completed':True,'all_six_archives_complete_and_verified':True,
               'independent_control_gain_not_yet_evaluated':True,'allocated_gpu_hours':self.gpu_seconds/3600})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--legacy-root',type=Path,required=True)
    args=parser.parse_args()
    with (args.batch/'RUN_LOCK').open('x') as stream:
        stream.write(str(os.getpid()))
    workflow=None
    code=0
    try:
        workflow=Workflow(args.batch,args.legacy_root)
        workflow.execute()
    except BaseException as error:
        code=1
        if workflow is not None:
            workflow.cancel.set()
            workflow.phase='needs_attention'
            workflow.status('stopped_or_failed')
        atomic(args.batch/'WORKFLOW_RESULT.json', {'status':'stopped_or_failed',
               'error_type':type(error).__name__,'error':str(error),
               'scientific_failure_not_inferred':True,'checkpoints_preserved':True})
    (args.batch/'supervisor.exit').write_text(str(code)+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
