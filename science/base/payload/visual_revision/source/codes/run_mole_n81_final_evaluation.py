"""Finite, budgeted A diagnosis -> eight frozen B/C pairs; no Actor tuning."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import time

from run_mole_large_scale_initial_stage import Stage, IMAGE, atomic, command


class FinalEvaluation(Stage):
    def __init__(self, batch, legacy):
        super().__init__(batch, legacy)
        self.inputs = Path(self.plan['input_batch'])
        self.phase = 'input_integrity_and_cpu_contracts'
        self.cpu_jobs = {}
        if self.inputs == self.batch or not self.inputs.is_absolute():
            raise ValueError('distinct immutable source batch required')

    def docker_prefix(self, gpu):
        return super().docker_prefix(gpu) + ['-v', str(self.inputs) + ':/inputs:ro']

    def status(self, label):
        super().status(label)
        if time.monotonic() - self.started >= self.plan['wall_seconds_limit']:
            self.cancel.set()
        if getattr(self, 'phase', '') == 'A_interface_and_physical_diagnosis':
            if self.gpu_seconds >= self.plan['pilot_gpu_hours_limit'] * 3600:
                self.cancel.set()
        if getattr(self, 'phase', '') == 'frozen_BC_paired_evaluation':
            if self.gpu_seconds - self.phase_started_gpu_seconds >= self.plan['confirmation_gpu_hours_limit'] * 3600:
                self.cancel.set()
        d = json.loads((self.batch / 'STATUS.json').read_text())
        d.update(phase=getattr(self, 'phase', 'initializing'),
                 training_started=False, heldout_evaluation=True,
                 confirmation_results_complete=(self.batch / 'EVALUATION_SUMMARY.json').exists())
        atomic(self.batch / 'STATUS.json', d)

    def cpu(self, name, argv, timeout=3600):
        cid = command(['docker', 'run', '-d', '--name', 'shao-' + self.task + '-' + name,
                       '--label', 'research.task=' + self.task,
                       '--user', '%d:%d' % (os.getuid(), os.getgid()), '--memory', '8g',
                       '-e', 'JAX_PLATFORMS=cpu', '-e', 'PYTHONPATH=/project/codes',
                       '-v', str(self.batch / 'stage') + ':/project:ro',
                       '-v', str(self.batch) + ':/batch:rw',
                       '-v', str(self.inputs) + ':/inputs:ro', '-w', '/project', IMAGE] + argv)
        started = time.monotonic()
        with self.lock:
            self.cpu_jobs[name] = {'container_id': cid, 'argv': argv, 'started_unix': time.time()}
            atomic(self.batch / 'CPU_JOBS.json', self.cpu_jobs)
        try:
            while self.inspect(cid)['State']['Running']:
                with self.lock:
                    self.status('running')
                if self.cancel.is_set() or time.monotonic() - started > timeout:
                    self.inspect(cid)
                    command(['docker', 'stop', '--time', '25', cid])
                    raise RuntimeError('bounded CPU task stopped: ' + name)
                time.sleep(5)
            state = self.inspect(cid)['State']
            if state['ExitCode'] or state['OOMKilled']:
                raise RuntimeError('CPU validation failed: ' + name)
        finally:
            state = self.inspect(cid)['State']
            if state['Running']:
                command(['docker', 'stop', '--time', '25', cid])
                state = self.inspect(cid)['State']
            (self.batch / 'logs' / (name + '.log')).write_text(command(['docker', 'logs', cid]))
            with self.lock:
                self.cpu_jobs[name].update(exit_code=state['ExitCode'], oom_killed=state['OOMKilled'],
                                           elapsed_seconds=time.monotonic() - started)
                atomic(self.batch / 'CPU_JOBS.json', self.cpu_jobs)

    def argv(self, source, start, context, interactions, destination):
        e = self.plan['evaluation']
        return ['python', '-u', 'codes/eval_mole_large_scale_paired.py',
                '--target-turbines', '81', '--plane-archive', '/inputs/archives/' + source['archive_name'],
                '--verified-archive', '/inputs/archives/' + source['archive_name'] + '/VERIFIED.json',
                '--actor', '/inputs/' + self.plan['actor_final']['relative_path'],
                '--plane-start', str(start), '--context-id', context,
                '--output-dir', '/batch/' + destination,
                '--burnin', str(e['burnin']), '--interactions', str(interactions),
                '--block-interactions', str(e['block_interactions']), '--seed', str(e['seed']),
                '--required-actor-interactions', str(self.plan['actor_final']['actual_global_interactions'])]

    def execute(self):
        if shutil.disk_usage(self.batch).free < self.plan['minimum_free_gib'] * 1024**3:
            raise ValueError('insufficient disk reserve')
        self.status('running')
        self.cpu('contracts', ['python', 'codes/test_mole_large_scale_contracts.py'], 600)
        self.cpu('inputs', ['python', 'codes/verify_mole_n81_final_inputs.py',
                           '--batch', '/batch', '--inputs', '/inputs'], 1800)
        sources = {s['source_id']: s for s in self.plan['sources']}
        # Rehash B/C in CPU jobs while diagnosing A. No B/C policy outcome is read here.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            verification = [pool.submit(self.cpu, 'rehash-' + key,
                           ['python', 'codes/verify_mole_large_scale_archive.py',
                            '--archive', '/inputs/archives/' + sources[key]['archive_name'],
                            '--target-turbines', '81', '--output', '/batch/verified/' + key + '.json'], 3600)
                            for key in ('B', 'C')]
            self.phase = 'A_interface_and_physical_diagnosis'
            self.run_job('pilot-A', 0, self.argv(sources['A'], 20000, 'n81_A_diagnostic',
                         self.plan['evaluation']['pilot_scored_interactions'], 'pilot_A'),
                         int(self.plan['pilot_gpu_hours_limit'] * 3600))
            pilot = json.loads((self.batch / 'pilot_A/RESULT.json').read_text())
            if pilot['status'] != 'complete' or pilot['common_burnin']['controls'] != 150:
                raise ValueError('A interface diagnostic incomplete')
            # No benefit gate, no Actor/window selection, no change to statistics after held-out use.
            atomic(self.batch / 'A_DIAGNOSTIC_ADMISSION.json', {
                'status': 'completed_interface_diagnostic', 'pilot_result_sha256':
                __import__('hashlib').sha256((self.batch / 'pilot_A/RESULT.json').read_bytes()).hexdigest(),
                'reward_used_for_admission': False, 'extended_statistical_stationarity_proven': False,
                'physical_warmup_sufficiency_proven': False,
                'boundary': '150-control matching-state paired protocol; burnin trace retained for physical diagnosis.'})
            for f in verification:
                f.result()
        self.phase = 'frozen_BC_paired_evaluation'
        self.phase_started_gpu_seconds = self.gpu_seconds
        self.status('running')
        atomic(self.batch / 'BC_UNBLIND_STARTED.json', {
            'unix': time.time(), 'actor_sha256': self.plan['actor_final']['sha256'],
            'protocol_amendment': self.plan['protocol_amendment'],
            'gain_gate': None, 'contexts': self.plan['contexts']})

        def queue(gpu):
            for context in self.plan['contexts'][gpu::4]:
                if self.cancel.is_set():
                    raise RuntimeError('evaluation cancelled or budget exhausted')
                self.run_job(context['context_id'], gpu,
                             self.argv(sources[context['source_id']], context['window_start'],
                                       context['context_id'], self.plan['evaluation']['scored_interactions'],
                                       'contexts/' + context['context_id']), 10800)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(queue, gpu) for gpu in range(4)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except BaseException:
                    self.cancel.set()
                    raise
        self.phase = 'paired_result_accounting'
        self.cpu('summarize', ['python', 'codes/summarize_mole_n81_final_evaluation.py', '--batch', '/batch'], 600)
        self.phase = 'evaluation_complete'
        self.status('complete')
        atomic(self.batch / 'WORKFLOW_RESULT.json', {'status': 'complete', 'contexts': 8,
               'paired_comparisons': 8, 'independent_physical_sources': 2,
               'allocated_gpu_hours': self.gpu_seconds / 3600,
               'efficiency_comparison': 'separate bounded queue', 'training_started': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--legacy-root', type=Path, required=True)
    a = p.parse_args()
    (a.batch / 'logs').mkdir(exist_ok=True)
    with (a.batch / 'RUN_LOCK').open('x') as f:
        f.write(str(os.getpid()))
    try:
        stage = FinalEvaluation(a.batch, a.legacy_root)
        stage.execute()
    except BaseException as exc:
        atomic(a.batch / 'WORKFLOW_RESULT.json', {'status': 'failed_or_budget_stopped',
               'type': type(exc).__name__, 'reason': str(exc), 'unix': time.time()})
        (a.batch / 'supervisor.exit').write_text('1\n')
        raise
    (a.batch / 'supervisor.exit').write_text('0\n')


if __name__ == '__main__':
    main()
