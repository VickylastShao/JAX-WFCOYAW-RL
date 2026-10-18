"""Bounded XT-H20 N81/N100 runtime admission and real precursor preparation."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

IMAGE = 'sha256:a72fc0a237072f4177524eab7896b3d5ab8d1e2d7d62eb1bf45ff163e9c5729d'
GPU_IDS = (0, 1, 2, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=35)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: ' + result.stderr[-2000:])
    return (result.stdout + (result.stderr if args[:2] == ['docker', 'logs'] else '')).strip()


def load_factory(legacy):
    source = legacy / 'temporal_credit_20260925/codes/launch_mole_n9_temporal_credit_rank_gate.py'
    spec = importlib.util.spec_from_file_location('_large_scale_verified_runtime', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.command_for_gpu


def prefix(factory, gpu, batch):
    if gpu not in GPU_IDS:
        raise ValueError('GPU outside authorized 0-3 scope')
    old = list(factory(gpu))
    if old[:2] != ['docker', 'run']:
        raise ValueError('unexpected verified Docker factory')
    result, i = old[:2], 2
    while i < len(old):
        token = old[i]
        if token in ('-v', '--volume'):
            destination = old[i+1].split(':')[1]
            if not destination.startswith('/project'):
                raise ValueError('unapproved inherited project mount')
            i += 2
            continue
        if token in ('-e', '--env'):
            key, sep, value = old[i+1].partition('=')
            if not sep:
                raise ValueError('ambiguous inherited environment')
            if key not in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
                           'XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX_ENABLE_X64'):
                result += [token, old[i+1]]
            i += 2
            continue
        if token in ('--rm', '-d', '--detach'):
            i += 1
            continue
        if token in ('--privileged', '--gpus', '--user', '-u', '--name'):
            raise ValueError('unexpected inherited ownership/device option')
        result.append(token)
        i += 1
    devices = [result[i+1].split(':')[0] for i in range(len(result)-1)
               if result[i] == '--device' and re.fullmatch('/dev/nvidia[0-9]+', result[i+1].split(':')[0])]
    if devices != ['/dev/nvidia' + str(gpu)]:
        raise ValueError('runtime factory exposes the wrong physical GPU')
    if not any('libnvidia-ml.so.595.91.07' in s for s in result):
        raise ValueError('verified matching NVML mount missing')
    return result + ['--user', f'{os.getuid()}:{os.getgid()}',
        '-e', 'PYTHONPATH=/project/codes', '-e', 'CUDA_VISIBLE_DEVICES=0',
        '-e', 'JAX_PLATFORMS=cuda', '-e', 'JAX_ENABLE_X64=0',
        '-e', 'XLA_PYTHON_CLIENT_MEM_FRACTION=.60',
        '-v', str(batch / 'stage') + ':/project:ro', '-v', str(batch) + ':/batch:rw']


class Stage:
    def __init__(self, batch, legacy):
        self.batch = batch
        self.plan = json.loads((batch / 'PLAN.json').read_text())
        self.manifest = json.loads((batch / 'CODE_MANIFEST.json').read_text())
        for name, expected in self.manifest.items():
            if sha(batch / 'stage' / name) != expected:
                raise ValueError('frozen source checksum mismatch: ' + name)
        if tuple(self.plan['gpu_ids']) != GPU_IDS or self.plan['image_digest'] != IMAGE:
            raise ValueError('resource scope changed')
        self.factory = load_factory(legacy)
        self.task = self.plan['task_id']
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.jobs = {}
        self.started = time.monotonic()
        self.gpu_seconds = 0.
        self.accounted = self.started
        self.active = 0
        signal.signal(signal.SIGTERM, lambda *_: self.cancel.set())
        signal.signal(signal.SIGINT, lambda *_: self.cancel.set())
        stat = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
        atomic(batch / 'SUPERVISOR.json', {'pid': os.getpid(), 'proc_starttime': stat[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'task_id': self.task, 'plan_sha256': sha(batch / 'PLAN.json'),
            'started_unix': time.time(), 'command': sys.argv})

    def status(self, label):
        now = time.monotonic()
        self.gpu_seconds += (now - self.accounted) * self.active
        self.accounted = now
        if self.gpu_seconds >= self.plan['initial_stage_gpu_hours_limit'] * 3600:
            self.cancel.set()
        atomic(self.batch / 'STATUS.json', {'task_id': self.task, 'status': label,
            'updated_unix': time.time(), 'elapsed_seconds': now-self.started,
            'allocated_gpu_hours': self.gpu_seconds/3600, 'active_containers': self.active,
            'jobs': self.jobs, 'scientific_control_results': False,
            'training_started': False})

    def inspect(self, cid):
        raw = json.loads(command(['docker', 'inspect', cid]))[0]
        if raw['Id'] != cid or raw['Image'] != IMAGE or raw['Config']['Labels'].get('research.task') != self.task:
            raise ValueError('container ownership identity mismatch')
        return raw

    def docker_prefix(self, gpu):
        """Allow a derived finite stage to add explicitly scoped read-only inputs."""
        return prefix(self.factory, gpu, self.batch)

    def run_job(self, job, gpu, argv, timeout):
        rows = command(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'])
        if rows:
            raise ValueError(f'GPU {gpu} already has a compute process; no process stopped')
        name = 'shao-large-' + self.task[:10] + '-' + job
        cid = command(self.docker_prefix(gpu) + ['-d', '--name', name,
            '--label', 'research.task=' + self.task, '-w', '/project', IMAGE] + argv)
        inspected = self.inspect(cid)
        if not inspected['State']['Running'] and inspected['State']['ExitCode'] != 0:
            raise RuntimeError('container exited during launch')
        started = time.monotonic()
        with self.lock:
            self.status('running')
            self.active += 1
            self.jobs[job] = {'gpu': gpu, 'container_id': cid, 'container_name': name,
                'argv': argv, 'started_unix': time.time(), 'status': 'running'}
            self.status('running')
        log = self.batch / 'logs' / (job + '.log')
        try:
            while True:
                raw = self.inspect(cid)
                log.write_text(command(['docker', 'logs', cid]) + '\n')
                if not raw['State']['Running']:
                    break
                with self.lock:
                    self.status('running')
                if self.cancel.is_set() or time.monotonic()-started > timeout:
                    self.inspect(cid)
                    command(['docker', 'stop', '--time', '25', cid])
                    break
                time.sleep(10)
            raw = self.inspect(cid)
            log.write_text(command(['docker', 'logs', cid]) + '\n')
            code = int(raw['State']['ExitCode'])
            with self.lock:
                self.status('running')
                self.active -= 1
                self.jobs[job].update(status='complete' if code == 0 else 'stopped_or_failed',
                    exit_code=code, oom_killed=raw['State']['OOMKilled'],
                    elapsed_seconds=time.monotonic()-started)
                self.status('running')
            if code != 0:
                raise RuntimeError(f'{job}: exit {code}, see {log}')
            return job
        except BaseException:
            raw = self.inspect(cid)
            if raw['State']['Running']:
                command(['docker', 'stop', '--time', '25', cid])
            with self.lock:
                if self.jobs[job]['status'] == 'running':
                    self.status('running')
                    self.active -= 1
                    self.jobs[job].update(status='failed',
                        exit_code=self.inspect(cid)['State']['ExitCode'])
                    self.status('running')
            raise

    def main(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            probes = [executor.submit(self.run_job, 'probe-n'+str(n), gpu,
                ['python', 'codes/probe_mole_large_scale_runtime.py', '--turbines', str(n),
                 '--output', '/batch/probes/n'+str(n)+'.json'], 1800)
                for gpu, n in enumerate((81, 100))]
            for future in concurrent.futures.as_completed(probes):
                future.result()
        for n in (81, 100):
            if json.loads((self.batch / f'probes/n{n}.json').read_text())['status'] != 'pass':
                raise ValueError('full runtime smoke failed; no precursor jobs admitted')
        atomic(self.batch / 'RUNTIME_ADMISSION.json', {'status': 'pass',
            'probe_sha256': {str(n): sha(self.batch/f'probes/n{n}.json') for n in (81,100)},
            'boundary': 'Numerical execution and learning-update smoke only.'})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            for gpu, (n, seed) in enumerate(((81,26100181),(81,26100281),(100,26100100),(100,26100200))):
                job = f'precursor-n{n}-s{seed}'
                out = self.batch / 'precursors' / job
                out.mkdir(exist_ok=False)
                argv = ['python', 'codes/generate_mole_precursor_resumable.py',
                    '--state', f'/batch/precursors/{job}/state.npz',
                    '--output', f'/batch/precursors/{job}/pilot200k.json',
                    '--batch', '1', '--seed', str(seed), '--target-steps', '200000',
                    '--chunk-steps', '20000', '--profile', 'article_224d_inferred',
                    '--target-turbines', str(n), '--compact-solver', 'dense_inverse', '--compact-stdout']
                tasks.append(executor.submit(self.run_job, job, gpu, argv, 21600))
            failures = []
            for future in concurrent.futures.as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
            if failures:
                raise RuntimeError('; '.join(failures))
        self.status('initial_stage_complete_not_training')
        atomic(self.batch / 'COMPLETE_INITIAL_STAGE.json', {'status': 'complete',
            'training_started': False, 'scientific_control_results': False,
            'next': 'Evaluate real precursor statistics and actual cost before generating training archives.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--legacy-root', type=Path, required=True)
    a = p.parse_args()
    (a.batch / 'logs').mkdir(exist_ok=True)
    lock = a.batch / 'RUN_LOCK'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        stage = Stage(a.batch, a.legacy_root)
        stage.main()
    except BaseException as exc:
        atomic(a.batch / 'INITIAL_STAGE_FAILURE.json', {'status': 'failed_or_budget_stopped',
            'type': type(exc).__name__, 'reason': str(exc), 'time_unix': time.time()})
        (a.batch / 'supervisor.exit').write_text('1\n')
        raise
    (a.batch / 'supervisor.exit').write_text('0\n')


if __name__ == '__main__':
    main()
