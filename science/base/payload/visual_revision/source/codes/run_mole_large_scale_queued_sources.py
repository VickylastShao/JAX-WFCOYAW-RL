"""Prepare independent C roots after known parent jobs release GPU1/GPU3.

This finite supervisor does not certify physical maturity or start training.
It uses only the existing allocation and the verified project Docker runtime.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_mole_large_scale_initial_stage import Stage, IMAGE, atomic, command


class QueuedSources(Stage):
    def await_parent(self, spec):
        started = time.monotonic()
        expected = spec['parent_container_id']
        while not self.cancel.is_set():
            parent = json.loads(command(['docker', 'inspect', expected]))[0]
            if (parent['Id'] != expected or parent['Image'] != IMAGE
                    or parent['Config']['Labels'].get('research.task') != self.plan['parent_task_id']):
                raise ValueError('parent container identity differs; no GPU process stopped')
            if not parent['State']['Running']:
                if parent['State']['OOMKilled']:
                    raise RuntimeError('parent hit OOM; new source not launched')
                if parent['State']['ExitCode'] not in (0, 137, 143):
                    raise RuntimeError('parent failed unexpectedly; inspect before rescheduling')
                metadata = Path(spec['parent_source_meta'])
                source = json.loads(metadata.read_text())
                history = source.get('diagnostic_history', [])
                if not history or not history[-1]['metrics']['finite']:
                    raise RuntimeError('parent has no finite saved real-flow block')
                running = command(['nvidia-smi', '-i', str(spec['gpu']),
                                   '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
                if running:
                    raise RuntimeError('GPU reused by another process; allocation not free')
                return
            if time.monotonic() - started > spec['wait_seconds_limit']:
                raise TimeoutError('bounded queue wait expired; parent was not interrupted')
            with self.lock:
                self.status('running_and_queued' if self.active else 'waiting_for_known_parent')
            time.sleep(15)
        raise RuntimeError('queue cancelled before launch')

    def source(self, spec):
        self.await_parent(spec)
        output = self.batch / 'sources' / spec['source_id']
        output.mkdir(parents=True, exist_ok=True)
        argv = ['python', 'codes/generate_mole_precursor_resumable.py',
                '--state', '/batch/sources/' + spec['source_id'] + '/state.npz',
                '--output', '/batch/sources/' + spec['source_id'] + '/pilot200k.json',
                '--batch', '1', '--seed', str(spec['seed']),
                '--target-steps', str(spec['target_steps']), '--chunk-steps', '20000',
                '--profile', 'article_224d_inferred', '--target-turbines', str(spec['turbines']),
                '--compact-solver', 'dense_inverse', '--compact-stdout']
        self.run_job(spec['source_id'], spec['gpu'], argv, spec['run_seconds_limit'])
        meta = json.loads((output / 'state.meta.json').read_text())
        if int(meta['completed_steps']) != spec['target_steps']:
            raise ValueError('finite source task exited before its declared target')
        atomic(output / 'COMPLETE.json', {'source_id': spec['source_id'],
               'completed_steps': meta['completed_steps'], 'state_sha256': meta['npz_sha256'],
               'maturity_qualified': False, 'training_started': False})
        return spec['source_id']

    def execute(self):
        (self.batch / 'logs').mkdir(exist_ok=True)
        with self.lock:
            for spec in self.plan['sources']:
                self.jobs[spec['source_id']] = {'gpu': spec['gpu'], 'status': 'queued',
                                              'parent_container_id': spec['parent_container_id']}
            self.status('queued_sources')
        completed = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(self.source, spec): spec['source_id'] for spec in self.plan['sources']}
            for future in concurrent.futures.as_completed(futures):
                try:
                    completed.append(future.result())
                except BaseException as error:
                    errors.append({'source_id': futures[future], 'error': str(error)})
                    self.cancel.set()
        result = {'status': 'complete' if not errors else 'stopped_or_failed',
                  'task_id': self.task, 'completed_sources': completed, 'errors': errors,
                  'allocated_gpu_hours': self.gpu_seconds / 3600,
                  'maturity_qualified': False, 'training_started': False,
                  'scope': 'Independent seed C preparation only; no scientific gain claim.'}
        atomic(self.batch / 'SOURCE_PREPARATION_RESULT.json', result)
        self.status(result['status'])
        return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, required=True)
    args = parser.parse_args()
    code = QueuedSources(args.batch, args.legacy_root).execute()
    (args.batch / 'supervisor.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
