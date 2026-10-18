"""Finite phase supervisor; P2/P3/P4 caps include failed allocations and cleanup."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from run_mole_large_scale_initial_stage import IMAGE, atomic, command
from run_mole_n81_final_evaluation import FinalEvaluation


CAPS = {'P2': 5.0, 'P3': 4.0, 'P4': 8.0}


class EvidencePhase(FinalEvaluation):
    def __init__(self, batch, legacy):
        super().__init__(batch, legacy)
        self.role = self.plan['phase_id']
        if self.role not in CAPS or self.plan['initial_stage_gpu_hours_limit'] != CAPS[self.role]:
            raise ValueError('phase budget differs from the authorized 5+4+8 GPUh roster')
        self.dispatch_extra_seconds = 0.0

    def docker_prefix(self, gpu):
        return super().docker_prefix(gpu) + [
            '-v', self.plan['efficiency_input_batch']+':/effinputs:ro',
            '-v', self.plan['evaluation_input_batch']+':/evalinputs:ro']

    def cpu_step(self, name, argv, timeout, memory):
        cid = command(['docker','run','-d','--name','shao-'+self.task+'-'+name,
            '--label','research.task='+self.task,'--network','none',
            '--user','%d:%d'%(os.getuid(),os.getgid()),'--cpus','4','--memory',memory,
            '-e','JAX_PLATFORMS=cpu','-e','CUDA_VISIBLE_DEVICES=',
            '-e','PYTHONPATH=/project/codes','-e','OMP_NUM_THREADS=2',
            '-v',str(self.batch/'stage')+':/project:ro','-v',str(self.batch)+':/batch:rw',
            '-v',str(self.inputs)+':/inputs:ro',
            '-v',self.plan['efficiency_input_batch']+':/effinputs:ro',
            '-v',self.plan['evaluation_input_batch']+':/evalinputs:ro',
            '-w','/project',IMAGE]+argv)
        started=time.monotonic()
        self.cpu_jobs[name]={'container_id':cid,'argv':argv,'started_unix':time.time(),'status':'running'}
        atomic(self.batch/'CPU_JOBS.json',self.cpu_jobs)
        try:
            while self.inspect(cid)['State']['Running']:
                self.status('running')
                if self.cancel.is_set() or time.monotonic()-started>timeout:
                    self.inspect(cid)
                    command(['docker','stop','--time','20',cid])
                    raise RuntimeError('bounded CPU validation stopped: '+name)
                time.sleep(5)
            state=self.inspect(cid)['State']
            if state['ExitCode'] or state['OOMKilled']:
                raise RuntimeError('CPU contract failed: '+name)
        finally:
            state=self.inspect(cid)['State']
            if state['Running']:
                command(['docker','stop','--time','20',cid]);state=self.inspect(cid)['State']
            (self.batch/'logs'/(name+'.log')).write_text(command(['docker','logs',cid]))
            self.cpu_jobs[name].update(status='complete' if state['ExitCode']==0 else 'failed',
                exit_code=state['ExitCode'],oom_killed=state['OOMKilled'],
                elapsed_seconds=time.monotonic()-started)
            atomic(self.batch/'CPU_JOBS.json',self.cpu_jobs)

    def execute(self):
        for index, step in enumerate(self.plan['steps']):
            self.phase=self.role+'-'+step['name']
            self.status('running')
            if self.cancel.is_set():raise RuntimeError('phase cancelled before a new allocation')
            argv=list(step['argv'])
            for item in step.get('argv_json_bindings',[]):
                value=json.loads((self.batch/item['relative_file']).read_text())
                for key in item['keys']:value=value[key]
                argv+= [item['flag'],str(value)]
            if step['kind']=='cpu':
                self.cpu_step(step['name'],argv,step.get('timeout_seconds',1200),step.get('memory','8g'))
            elif step['kind']=='gpu':
                remaining=CAPS[self.role]*3600-self.gpu_seconds-self.dispatch_extra_seconds
                timeout=min(float(step['allocation_seconds_limit']),remaining-120.)
                if not math.isfinite(timeout) or timeout<=60:
                    raise RuntimeError('insufficient allocation and cleanup reserve')
                before=time.monotonic()
                counted_before=self.gpu_seconds
                try:self.run_job(step['name'],step['gpu'],argv,timeout)
                finally:
                    # One physical GPU per step; include dispatch/start/inspection tail.
                    extra=max(0.,time.monotonic()-before-(self.gpu_seconds-counted_before))
                    self.dispatch_extra_seconds+=extra
                    atomic(self.batch/'ALLOCATION_ACCOUNTING.json',{
                        'phase':self.role,'cap_gpu_hours':CAPS[self.role],
                        'allocated_gpu_hours':(self.gpu_seconds+self.dispatch_extra_seconds)/3600,
                        'dispatch_tail_gpu_seconds':self.dispatch_extra_seconds,
                        'active_integral_gpu_seconds':self.gpu_seconds,
                        'all_attempts_included':True})
            else:raise ValueError('unknown step kind')
            for output in step.get('required_outputs',[]):
                if not (self.batch/output).is_file():raise ValueError('missing phase output: '+output)
        self.phase=self.role+'-complete'
        self.status('complete')
        atomic(self.batch/'WORKFLOW_RESULT.json',{'status':'complete','phase':self.role,
            'allocated_gpu_hours':(self.gpu_seconds+self.dispatch_extra_seconds)/3600,
            'cap_gpu_hours':CAPS[self.role],'no_automatic_retry':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path,required=True)
    a=p.parse_args();(a.batch/'logs').mkdir(exist_ok=True)
    with (a.batch/'RUN_LOCK').open('x') as f:f.write(str(os.getpid()))
    stage=None
    try:
        stage=EvidencePhase(a.batch,a.legacy_root);stage.execute()
    except BaseException as exc:
        if stage is not None:
            stage.cancel.set();stage.status('failed_or_budget_stopped')
        atomic(a.batch/'WORKFLOW_RESULT.json',{'status':'failed_or_budget_stopped',
            'reason':str(exc),'type':type(exc).__name__,
            'allocated_gpu_hours':((stage.gpu_seconds+stage.dispatch_extra_seconds)/3600 if stage else 0.),
            'checkpoints_and_logs_preserved':True})
        (a.batch/'supervisor.exit').write_text('1\n');raise
    (a.batch/'supervisor.exit').write_text('0\n')


if __name__=='__main__':main()
