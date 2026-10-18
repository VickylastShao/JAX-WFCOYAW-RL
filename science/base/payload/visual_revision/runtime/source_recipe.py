#!/usr/bin/env python3
"""Resolve source generation/acquisition recipes only. This tool never launches LES."""
import argparse
import json
from pathlib import Path
import sys
from resolve_run import PACK, parser_from_frozen_source, json_value, source_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('A','B','C'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output-recipe', type=Path)
    args = parser.parse_args()
    config = json.loads((PACK/'configs/runtime_sources.json').read_text())
    source = next(s for s in config['sources'] if s['source_id']==args.source)
    root = args.root.resolve()
    jobs = []
    for label,key in [('spinup','source_spinup_recipe'),('archive','archive_recipe'),
                      ('verify_archive','verification_recipe')]:
        original=source[key]
        argv=[args.python, *original[1:]]
        entry_index=next(i for i,v in enumerate(argv) if v.startswith('codes/'))
        entry=PACK/'source'/argv[entry_index]
        argv[entry_index]=str(entry)
        for i,v in enumerate(argv):
            if v.startswith('/batch/'):
                argv[i]=str(root/v[len('/batch/'):])
        frozen_parser=parser_from_frozen_source(entry)
        parsed=frozen_parser.parse_args(argv[entry_index+1:])
        jobs.append({'step':label,'argv':argv,
                     'all_arguments_parsed_by_frozen_registration':True,
                     'resolved_defaults':json_value(vars(parsed))})
    # Source-state verification accepts a plan; bind the exact recorded physics,
    # geometry, bytes identity and seed instead of inventing a qualification rule.
    source_plan={'sources':[{'source_id':source['archive_name'],'turbines':81,
        'role':source['source_id'],'seed':source['seed'],
        'original_state':str(root/source['source_state_relative_path']),
        'expected_sha256':source['source_state_sha256'],'saved_steps':200000,
        'case':source['case'],'provenance':{
            'target_turbines':81,'target_layout_name':'mole_rect_9x9',
            'target_layout_sha256':source['layout_sha256'],
            'target_domain_lz_m':5922.0,'target_nz':483}}]}
    qualification_entry=PACK/'source/codes/verify_mole_precursor_state.py'
    verification_plan=root/'qualified_sources'/source['archive_name']/'VERIFY_FROZEN_SOURCE.json'
    qualification_argv=[args.python,str(qualification_entry),'--plan',str(verification_plan),
                        '--output',str(verification_plan.parent/'VERIFIED.json')]
    qualification_parser=parser_from_frozen_source(qualification_entry)
    qualification_parser.parse_args(qualification_argv[2:])
    result={'schema_version':1,'status':'source_recipes_resolved_not_executed',
        'source_id':args.source,'root':str(root),'frozen_identity':source_checks(),
        'source_state_contract':{'relative_path':source['source_state_relative_path'],
            'sha256':source['source_state_sha256'],'completed_steps':200000,
            'availability':'needs_external_input','public_fetch_url':None},
        'archive_contract':{'relative_path':source['archive_relative_path'],
            'manifest_sha256':source['manifest_sha256'],
            'planes':source['completed_planes'],'chunks':source['chunk_files'],
            'nominal_chunk_bytes':source['chunk_bytes']},
        'source_qualification':{'plan_path':str(verification_plan),'plan_payload':source_plan,
            'argv':qualification_argv,'all_arguments_parsed_by_frozen_registration':True,
            'scope':'Finite state, complete history, geometry and frozen SHA; not proof of statistical stationarity.'},
        'jobs':jobs,'environment':{'JAX_PLATFORMS':'cuda','JAX_ENABLE_X64':'0',
            'XLA_PYTHON_CLIENT_MEM_FRACTION':'.60','PYTHONPATH':str(PACK/'source/codes')},
        'execution_guard':'No execution branch in this recipe tool. Independent operator budget required before running generated GPU argv.',
        'important_boundary':config['regeneration_boundary'],
        'source_initialization':'If state.npz is absent, the frozen generator starts its recorded seed-dependent log-profile/noise initial state. If present, it resumes and validates matching source metadata/history. Do not mix new spinup and original byte-identity qualification.'}
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output_recipe:
        with args.output_recipe.open('x') as stream:stream.write(encoded)
    print(encoded,end='')


if __name__=='__main__':
    main()
