import json
from pathlib import Path
import time
import numpy as np
from metaworld_fast.scheduler import jobs_from_config, make_requests, status_snapshot
from metaworld_fast.util import load_config

def test_requested_job_counts_and_common_anchor_once():
    config=load_config(Path(__file__).parents[1]/'config/final.yaml')
    jobs=jobs_from_config(config)
    assert len(jobs)==210
    assert sum(j['M']*j['answers'] for j in jobs)==161280
    assert sum(j['answers'] for j in jobs)==26880
    assert len({(j['setting_id'],j['method'],j['interaction_id']) for j in jobs})==210
    assert sum(j['setting_id']=='m04_b04' for j in jobs)==30

def test_reference_request_subset_and_preserved_duplicate_slots(tmp_path):
    path=tmp_path/'m04_b04__PS__i0.npz'
    q=np.zeros((128,4,4));refs=np.zeros((128,7,4,4))
    np.savez(path,q=q,reference_queries=refs)
    requests=make_requests(path,tmp_path)
    assert sum(len(r['q']) for r in requests if r['kind']=='selected')==512
    assert sum(len(r['q']) for r in requests if r['kind']=='reference')==56
    assert {r['t'] for r in requests if r['kind']=='reference'}=={0,127}
    assert all(len(r['q'])==4 for r in requests)

def test_live_progress_does_not_double_count_finished_checkpoints(tmp_path):
    for folder in ('statistical','checkpoints','rollouts'): (tmp_path/folder).mkdir()
    key='m04_b04__SG__i0'
    np.savez(tmp_path/'statistical'/f'{key}.npz',y=np.zeros(128))
    (tmp_path/'checkpoints'/f'{key}.progress.json').write_text(json.dumps({'run_key':key,'answers_completed':128}))
    (tmp_path/'checkpoints'/'another.progress.json').write_text(json.dumps({'run_key':'another','answers_completed':16}))
    result=status_snapshot(tmp_path,'testing',time.time())
    assert result['answers_complete']==144
    assert result['statistical_runs_complete']==1
