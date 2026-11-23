"""Public execution interface. Heavy libraries are imported only by commands."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main(argv=None):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    parser=argparse.ArgumentParser(description='RC-MLQ Meta-World T128 S5 experiment')
    parser.add_argument('command',choices=['inspect','test','profile','run','validate','plot','status'])
    parser.add_argument('--config',default='metaworld_fast/config/final.yaml')
    parser.add_argument('--run-dir')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    from .util import load_config, hardware
    config=load_config(args.config)
    if args.command=='inspect':
        from .artifact_reuse import inspect_reuse
        print(json.dumps({'hardware':hardware(),'reuse':inspect_reuse(),'config':config},indent=2))
    elif args.command=='test':
        return subprocess.call([sys.executable,'-m','pytest','-q','metaworld_fast/tests'])
    else:
        if not args.run_dir: parser.error('--run-dir is required for this command')
        run_dir=Path(args.run_dir).resolve()
        if args.command=='status':
            path=run_dir/'status.json'
            if not path.exists(): parser.error(f'No status file at {path}')
            data=json.loads(path.read_text())
            try: os.kill(data['pid'],0);data['supervisor_alive']=True
            except ProcessLookupError: data['supervisor_alive']=False
            print(json.dumps(data,indent=2))
        elif args.command=='run':
            from .scheduler import run_pipeline
            run_pipeline(config,run_dir)
        elif args.command=='profile':
            from .profiling import profile_workload
            print(json.dumps(profile_workload(run_dir,config),indent=2))
        elif args.command=='validate':
            from .validation import validate_run
            print(json.dumps(validate_run(run_dir,config=config),indent=2))
        elif args.command=='plot':
            from .reporting import generate_report
            generate_report(run_dir,config=config,make_plots=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
