import copy
import json
from pathlib import Path
import time
import numpy as np
import yaml
from metaworld_fast.extension import extension_config, NEW_IDS, OLD_IDS, status
from metaworld_fast.scheduler import jobs_from_config
from metaworld_fast.reporting import endpoint_tables
from metaworld_fast.util import load_config
from metaworld_fast.model import stable_seed


def frozen_source(tmp_path):
    original = load_config(Path(__file__).parents[1]/'config/final.yaml')
    (tmp_path/'config_frozen.yaml').write_text(yaml.safe_dump(original))
    return original


def test_extension_only_changes_seed_budget_and_preserves_original_config(tmp_path):
    original = frozen_source(tmp_path)
    frozen_bytes=(tmp_path/'config_frozen.yaml').read_bytes()
    config=extension_config(tmp_path)
    assert config['experiment']['interaction_ids']==list(range(10))
    for section in ('queries','inference','measurements','validation','resources'):
        assert config[section]==original[section]
    before=copy.deepcopy(original['experiment']);after=copy.deepcopy(config['experiment'])
    for item in (before,after):item.pop('interaction_ids');item.pop('name')
    assert before==after
    all_jobs=jobs_from_config(config)
    additional=[j for j in all_jobs if j['interaction_id'] in NEW_IDS]
    assert len(all_jobs)==420 and len(additional)==210
    assert sum(j['answers'] for j in additional)==26880
    assert sum(j['M']*j['answers'] for j in additional)==161280
    assert not any(j['interaction_id'] in OLD_IDS for j in additional)
    assert (tmp_path/'config_frozen.yaml').read_bytes()==frozen_bytes
    theta=[]
    for i in range(10):
        seed=stable_seed(20261004,'m04_b04',i,None,'theta')
        theta.append(np.random.default_rng(seed).uniform(-1,1,4))
    assert len(np.unique(theta,axis=0))==10


def test_ten_seed_mean_uses_all_ten_and_marks_missing_value(tmp_path):
    frozen_source(tmp_path)
    config=extension_config(tmp_path)
    (tmp_path/'config_frozen.yaml').write_text(yaml.safe_dump(config))
    runs=[]
    for i in range(10):
        eta=np.full(128,float(i))
        if i==9:eta[31]=np.nan
        runs.append(dict(setting_id='m04_b04',method='RC',interaction_id=i,path=Path(f'run{i}.npz'),
                         data=dict(y=np.zeros(128),risk=np.full(129,float(i)),error=np.zeros(129),eta=eta)))
    _,means=endpoint_tables(runs,tmp_path)
    risk=next(r for r in means if r['answer_count']==32 and r['metric']=='posterior_covariance_trace')
    assert risk['mean']==4.5 and risk['n_effective']==10 and risk['n_expected']==10 and risk['complete_expected']
    eta=next(r for r in means if r['answer_count']==32 and r['metric']=='realized_information_ratio')
    assert eta['mean']==4 and eta['n_effective']==9 and not eta['complete_expected']


def test_extension_progress_separates_reused_and_new_runs(tmp_path):
    (tmp_path/'statistical').mkdir();(tmp_path/'checkpoints').mkdir()
    for i in (0,5):
        np.savez(tmp_path/'statistical'/f'm04_b04__RC__i{i}.npz',y=np.zeros(128))
    (tmp_path/'checkpoints/a.progress.json').write_text(json.dumps(dict(run_key='m04_b04__RC__i5',answers_completed=128)))
    (tmp_path/'checkpoints/b.progress.json').write_text(json.dumps(dict(run_key='m04_b04__RC__i6',answers_completed=16)))
    state=status(tmp_path,'testing',time.time())
    assert state['statistical_runs_complete']==2
    assert state['additional_statistical_runs_complete']==1
    assert state['answers_complete']==272
    assert state['statistical_runs_total']==420
