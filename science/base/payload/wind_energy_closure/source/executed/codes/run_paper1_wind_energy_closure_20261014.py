"""Finite 4+8 GPUh closure supervisor using the verified XT-H20 runtime."""
import argparse
import json
from pathlib import Path
import time
from run_mole_n81_final_evaluation import FinalEvaluation
from run_mole_large_scale_initial_stage import atomic


class Closure(FinalEvaluation):
    def docker_prefix(self, gpu):
        return super().docker_prefix(gpu) + ['-v', self.plan['evaluation_input_batch']+':/evalinputs:ro']

    def execute(self):
        base = ['python', '-u', 'codes/paper1_wind_energy_closure_20261014.py']
        self.phase = 'CPU_contracts_and_common_state_surface_flux'
        self.cpu('contracts', ['python', 'codes/test_paper1_wind_energy_closure_20261014.py'], 600)
        # CPU helper inherited above lacks /evalinputs, so use the already mounted
        # /inputs only for contracts; run common-state audit through a GPU-hidden
        # CPU container with explicit read-only scoring inputs below.
        self.cpu_scoring('common-states', base + ['cpu-audit', '--output', '/batch/numerics/common_states'], 1800)
        self.phase = 'E1_microbatch_component'
        e1_start = self.gpu_seconds
        for m in (1, 2, 4):
            self.run_job('mb'+str(m), 0, base + ['microbatch', '--microbatch', str(m),
                          '--output', '/batch/execution/mb'+str(m)], 4400)
            if self.gpu_seconds - e1_start > 4*3600:
                raise RuntimeError('E1 total 4 GPUh exhausted')
        self.cpu_scoring('equivalence', base+['compare', '--output', '/batch/execution'], 1800, '32g')
        equivalence=json.loads((self.batch/'execution/COMPARISON.json').read_text())
        if not equivalence['all_equivalent']:
            raise ValueError('microbatch physical-state equivalence failed; no numeric extension')
        self.phase = 'N1_matched_physical_filter_cadence'
        numeric_start=self.gpu_seconds
        for dt in (.2, .1):
            self.run_job('dt'+str(dt).replace('.',''), 1, base+['numeric', '--dt', str(dt),
                '--archive', '/inputs/archives/n81_train_A', '--actor', '/inputs/runs/n81_train_r1/actor.pkl',
                '--output', '/batch/numerics/dt'+str(dt).replace('.','')], 14000)
            if self.gpu_seconds-numeric_start > 8*3600:
                raise RuntimeError('N1 total 8 GPUh exhausted')
        self.status('complete')
        atomic(self.batch/'WORKFLOW_RESULT.json', {'status':'complete', 'allocated_gpu_hours':self.gpu_seconds/3600,
             'no_training':True, 'no_grid_convergence_claim':True, 'automatic_retry':False})

    def cpu_scoring(self, name, argv, timeout, memory='8g'):
        from run_mole_large_scale_initial_stage import IMAGE, command
        import os
        cid=command(['docker','run','-d','--name','shao-'+self.task+'-'+name,
             '--label','research.task='+self.task,'--network','none','--cpus','4','--memory',memory,
             '--user',f'{os.getuid()}:{os.getgid()}', '-e','JAX_PLATFORMS=cpu','-e','CUDA_VISIBLE_DEVICES=',
             '-e','PYTHONPATH=/project/codes','-e','OMP_NUM_THREADS=2',
             '-v',str(self.batch/'stage')+':/project:ro','-v',str(self.batch)+':/batch:rw',
             '-v',self.plan['evaluation_input_batch']+':/evalinputs:ro','-w','/project',IMAGE]+argv)
        started=time.monotonic()
        try:
            while self.inspect(cid)['State']['Running']:
                self.status('running')
                if self.cancel.is_set() or time.monotonic()-started>timeout:
                    self.inspect(cid);command(['docker','stop','--time','20',cid])
                    raise RuntimeError('bounded CPU step stopped: '+name)
                time.sleep(10)
            if self.inspect(cid)['State']['ExitCode']:
                raise RuntimeError('CPU step failed: '+name)
        finally:
            state=self.inspect(cid)['State']
            if state['Running']:
                command(['docker','stop','--time','20',cid]);state=self.inspect(cid)['State']
            (self.batch/'logs'/(name+'.log')).write_text(command(['docker','logs',cid]))
            self.cpu_jobs[name]={'container_id':cid,'exit_code':state['ExitCode'],
                'elapsed_seconds':time.monotonic()-started,'oom_killed':state['OOMKilled']}
            atomic(self.batch/'CPU_JOBS.json',self.cpu_jobs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path,required=True);a=p.parse_args()
    with (a.batch/'RUN_LOCK').open('x') as f:f.write(str(__import__('os').getpid()))
    supervisor=None
    try:
        supervisor=Closure(a.batch,a.legacy_root);supervisor.execute()
    except BaseException as exc:
        if supervisor:
            supervisor.cancel.set();supervisor.status('failed_or_budget_stopped')
        atomic(a.batch/'WORKFLOW_RESULT.json',{'status':'failed_or_budget_stopped','type':type(exc).__name__,
            'reason':str(exc),'allocated_gpu_hours':supervisor.gpu_seconds/3600 if supervisor else 0.,
            'logs_and_states_preserved':True})
        (a.batch/'supervisor.exit').write_text('1\n');raise
    (a.batch/'supervisor.exit').write_text('0\n')


if __name__=='__main__':main()
