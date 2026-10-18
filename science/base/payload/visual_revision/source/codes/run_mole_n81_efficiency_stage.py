"""Wait for the owned evaluation to exit, then run finite matched 1/2/4 GPU profiles."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import time

from run_mole_large_scale_initial_stage import atomic
from run_mole_n81_final_evaluation import FinalEvaluation


WORK_FIELDS = {
    'E': 'global_environments', 'global_sample': 'global_sample_size',
    'replay': 'replay_capacity_per_rank', 'microbatch': 'microbatch',
    'episode': 'episode_interactions', 'physical_reset_controls': 'reset_interactions',
    'initial_burnin': 'initial_burnin_interactions',
    'synthetic_initial_episode_clock': 'initial_episode_clock_offset',
    'controls_per_environment': 'controls_per_environment',
    'real_LES_steps_per_environment': 'planes_consumed_per_environment',
    'real_SAC_updates': 'global_learner_updates',
}


def plan_settings(plan):
    efficiency = plan['efficiency']
    work = efficiency['compute_protocol']
    if work['topologies'] != [1, 2, 4]:
        raise ValueError('matched work must retain the declared 1/2/4 topologies')
    order = efficiency['execution_order']
    if len(order) != 3 or sorted(order) != [1, 2, 4] or any(type(p) is not int for p in order):
        raise ValueError('execution_order must be a permutation of 1/2/4')
    total = float(efficiency['gpu_hours_limit'])
    per_topology = float(work['topology_gpu_hour_cap'])
    if not all(math.isfinite(x) and x > 0 for x in (total, per_topology)):
        raise ValueError('profile budgets must be finite and positive')
    if total != float(plan['initial_stage_gpu_hours_limit']) or total != 3 * per_topology:
        raise ValueError('stage, profile total and three topology budgets differ')
    return total, per_topology, tuple(order)


def validate_protocol(plan, protocol):
    """Check prepared work and actual rank limits before any GPU is started."""
    total, per_topology, order = plan_settings(plan)
    work = plan['efficiency']['compute_protocol']
    if protocol['topologies'] != work['topologies']:
        raise ValueError('prepared matched topologies differ from the frozen plan')
    if (protocol['total_gpu_hour_cap'] != total
            or protocol['topology_gpu_hour_cap'] != per_topology):
        raise ValueError('prepared profile budgets differ from the frozen plan')
    if any(protocol[value] != work[key] for key, value in WORK_FIELDS.items()):
        raise ValueError('prepared profile computation differs from the frozen plan')
    source = next(s for s in plan['sources'] if s['source_id'] == 'A')
    if (protocol['source_manifest_sha256'] != source['manifest_sha256']
            or protocol['source_archive'] != '/inputs/archives/' + source['archive_name']):
        raise ValueError('prepared profile source differs from immutable training A')
    reserve = protocol['checkpoint_save_reserve_seconds']
    if not math.isfinite(reserve) or reserve <= 0:
        raise ValueError('positive synchronized checkpoint save reserve required')
    ranks = protocol['ranks']
    if len(ranks) != 7:
        raise ValueError('exactly seven rank specifications required for 1/2/4')
    identities = [(r['topology'], r['rank']) for r in ranks]
    expected = [(p, rank) for p in (1, 2, 4) for rank in range(p)]
    if sorted(identities) != expected:
        raise ValueError('prepared rank roster differs from 1/2/4')
    for rank in ranks:
        topology = rank['topology']
        hard_wall = per_topology * 3600 / topology
        argv = rank['trainer_argv']
        def flag(name):
            return argv[argv.index(name) + 1]
        if (rank['gpu'] != rank['rank'] or rank['hard_wall_seconds'] != hard_wall
                or float(flag('--stop-after-seconds')) != hard_wall - reserve
                or hard_wall <= reserve):
            raise ValueError('rank ownership or save/stop budget differs from the frozen profile')
        flags = {'--local-environments': work['E'] // topology,
                 '--sample-size': work['global_sample'] // topology,
                 '--num-processes': topology, '--process-id': rank['rank'],
                 '--microbatch': work['microbatch'], '--replay-capacity': work['replay'],
                 '--collection-steps': protocol['collection_steps'],
                 '--episode-interactions': work['episode'],
                 '--reset-interactions': work['physical_reset_controls'],
                 '--burnin-interactions': work['initial_burnin'],
                 '--planned-global-interactions': protocol['global_collected_interactions']}
        if any(int(flag(name)) != value for name, value in flags.items()):
            raise ValueError('rank arguments do not perform the prepared matched work')
    return order


def write_failure_status(batch, stage, exc):
    """Preserve evidence and make an exited supervisor's state terminal."""
    path = batch / 'STATUS.json'
    refresh_error = None
    if stage is not None:
        stage.cancel.set()
        stage.phase = 'efficiency_failed_or_budget_stopped'
        try:
            with stage.lock:
                stage.status('failed_or_budget_stopped')
        except BaseException as status_exc:
            refresh_error = type(status_exc).__name__
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(status='failed_or_budget_stopped', terminal=True, supervisor_exit_code=1,
                 phase='efficiency_failed_or_budget_stopped', updated_unix=time.time(),
                 failure={'type': type(exc).__name__, 'reason': str(exc)},
                 checkpoints_and_failed_artifacts_preserved=True,
                 scientific_policy_training=False, heldout_evaluation=False)
    if refresh_error:
        state['terminal_accounting_refresh_error'] = refresh_error
    atomic(path, state)
    return state


class Efficiency(FinalEvaluation):
    def __init__(self, batch, legacy):
        super().__init__(batch, legacy)
        self.total_gpu_hour_cap, self.topology_gpu_hour_cap, self.execution_order = plan_settings(self.plan)

    def status(self, label):
        super().status(label)
        if self.gpu_seconds >= self.total_gpu_hour_cap * 3600:
            self.budget_stop_reason = 'frozen total allocated GPU-hour budget exhausted'
        elif time.monotonic() - self.started >= self.plan['wall_seconds_limit']:
            self.budget_stop_reason = 'frozen supervisor wall-time budget exhausted'
        if getattr(self, 'phase', '') == 'matched_compute_profiles':
            if self.gpu_seconds - self.topology_started_gpu_seconds >= self.topology_gpu_hour_cap * 3600:
                self.cancel.set()
                self.budget_stop_reason = 'frozen topology allocated GPU-hour budget exhausted'
        d = json.loads((self.batch / 'STATUS.json').read_text())
        d.update(heldout_evaluation=False, scientific_policy_training=False,
                 profile_results_complete=(self.batch / 'efficiency/SCALING_SUMMARY.json').exists(),
                 topology=getattr(self, 'topology', None),
                 topology_allocated_gpu_hours=((self.gpu_seconds - self.topology_started_gpu_seconds) / 3600
                                               if hasattr(self, 'topology_started_gpu_seconds') else None),
                 budget_stop_reason=getattr(self, 'budget_stop_reason', None),
                 terminal=label in ('complete', 'failed_or_budget_stopped'))
        atomic(self.batch / 'STATUS.json', d)

    def execute(self):
        if shutil.disk_usage(self.batch).free < self.plan['minimum_free_gib'] * 1024**3:
            raise ValueError('insufficient disk reserve for matched profile checkpoints')
        self.phase = 'waiting_for_owned_evaluation_exit'
        prior = Path(self.plan['wait_for_evaluation'])
        identity_path = prior / 'SUPERVISOR.json'
        identity = json.loads(identity_path.read_text())
        if identity['task_id'] != self.plan['evaluation_task_id']:
            raise ValueError('evaluation identity differs from frozen dependency')
        deadline = time.monotonic() + self.plan['dependency_wait_seconds_limit']
        while not (prior / 'supervisor.exit').exists():
            if self.cancel.is_set() or time.monotonic() > deadline:
                raise RuntimeError('bounded evaluation dependency wait stopped')
            pid = identity['pid']
            stat_path = Path('/proc') / str(pid) / 'stat'
            if not stat_path.exists():
                raise RuntimeError('evaluation supervisor disappeared without an exit receipt')
            stat = stat_path.read_text().split(') ', 1)[1].split()
            if stat[19] != identity['proc_starttime']:
                raise ValueError('evaluation PID identity changed')
            self.status('waiting')
            time.sleep(20)
        atomic(self.batch / 'EVALUATION_DEPENDENCY_EXIT.json', {
            'exit_code': int((prior / 'supervisor.exit').read_text()),
            'evaluation_task_id': identity['task_id'],
            'boundary': 'Compute profile is independent of scientific gain; GPU availability rechecked for each rank.'})
        self.phase = 'efficiency_cpu_contracts_and_prepare'
        self.cpu('stage-contracts', ['python', 'codes/test_mole_n81_efficiency_stage.py'], 600)
        self.cpu('profile-contracts', ['python', 'codes/test_mole_n81_efficiency_profile.py'], 600)
        self.cpu('actual-yaw-contract', ['python', 'codes/test_mole_n81_yaw_observation_contract.py'], 600)
        self.cpu('profile-prepare', ['python', 'codes/profile_mole_n81_efficiency.py', 'prepare',
            '--source-archive', '/inputs/archives/n81_train_A',
            '--recipes', '/inputs/COMMAND_RECIPES_APPROVED.json',
            '--output-root', '/batch/efficiency', '--global-environments',
            str(self.plan['efficiency']['compute_protocol']['E'])], 600)
        protocol = json.loads((self.batch / 'efficiency/PROFILE.json').read_text())
        order = validate_protocol(self.plan, protocol)
        for topology in order:
            self.phase = 'matched_compute_profiles'
            self.topology = topology
            self.topology_started_gpu_seconds = self.gpu_seconds
            self.status('running')
            if self.cancel.is_set():
                raise RuntimeError('profile cancelled or frozen budget exhausted before topology launch')
            specs = [r for r in protocol['ranks'] if r['topology'] == topology]
            with concurrent.futures.ThreadPoolExecutor(max_workers=topology) as pool:
                futures = [pool.submit(self.run_job, 'p%d-rank%d' % (topology, r['rank']),
                                       r['gpu'], r['argv'], r['hard_wall_seconds']) for r in specs]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except BaseException:
                        self.cancel.set()
                        raise
            reports = [json.loads((self.batch / 'efficiency' / ('p%d' % topology) /
                                  ('rank%d.profile.json' % r['rank'])).read_text()) for r in specs]
            if any(r['status'] != 'pass' or not all(r['checks'].values()) for r in reports):
                raise ValueError('exact matched profile incomplete; preserve evidence, no automatic retry')
            atomic(self.batch / ('p%d_COMPLETE.json' % topology), {'status': 'complete',
                   'ranks': topology, 'allocated_gpu_hours':
                   (self.gpu_seconds - self.topology_started_gpu_seconds) / 3600})
        self.phase = 'efficiency_summary'
        self.cpu('profile-summary', ['python', 'codes/profile_mole_n81_efficiency.py', 'summarize',
                  '--protocol', '/batch/efficiency/PROFILE.json',
                  '--output', '/batch/efficiency/SCALING_SUMMARY.json'], 600)
        self.phase = 'efficiency_complete'
        self.status('complete')
        atomic(self.batch / 'WORKFLOW_RESULT.json', {'status': 'complete',
               'allocated_gpu_hours': self.gpu_seconds / 3600,
               'scientific_policy_training': False, 'end_to_end_training_speedup_proven': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--legacy-root', type=Path, required=True)
    a = p.parse_args()
    (a.batch / 'logs').mkdir(exist_ok=True)
    with (a.batch / 'RUN_LOCK').open('x') as f:
        f.write(str(os.getpid()))
    stage = None
    try:
        stage = Efficiency(a.batch, a.legacy_root)
        stage.execute()
    except BaseException as exc:
        failed = write_failure_status(a.batch, stage, exc)
        atomic(a.batch / 'WORKFLOW_RESULT.json', {'status': 'failed_or_budget_stopped',
               'type': type(exc).__name__, 'reason': str(exc), 'unix': time.time(),
               'allocated_gpu_hours': failed.get('allocated_gpu_hours'),
               'topology_allocated_gpu_hours': failed.get('topology_allocated_gpu_hours'),
               'budget_stop_reason': failed.get('budget_stop_reason'),
               'failed_artifacts_preserved': True})
        (a.batch / 'supervisor.exit').write_text('1\n')
        raise
    (a.batch / 'supervisor.exit').write_text('0\n')


if __name__ == '__main__':
    main()
