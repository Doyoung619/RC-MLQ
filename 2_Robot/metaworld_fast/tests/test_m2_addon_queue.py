"""Admission invariants: time gating, unfinished predecessors, and correctness seal."""
import json
from pathlib import Path

from metaworld_fast.m2_addon_queue import gate_decision, predecessor_complete, verify_ready, timestamp, completion_stage


def test_not_before_never_reads_predecessor(tmp_path):
    plan={'not_before_utc':'2026-10-06T00:57:46Z',
          'predecessors':[{'kind':'continuous_ofat','status_path':str(tmp_path/'absent.json')}]}
    result=gate_decision(plan,timestamp('2026-10-06T00:57:45Z'))
    assert not result['eligible'] and result['stage']=='delayed'
    assert result['seconds_until_queue_admission']==1


def test_stopped_preexisting_work_keeps_priority(tmp_path):
    path=tmp_path/'prior.json'
    path.write_text(json.dumps({'state':'STOPPED','physical':
                    {'expected':22528,'completed':106,'pending':22422,'running':0,'failed':0}}))
    plan={'not_before_utc':'2026-10-06T00:57:46Z',
          'predecessors':[{'kind':'continuous_ofat','status_path':str(path)}]}
    assert gate_decision(plan,timestamp('2026-10-06T01:00:00Z'))['stage']=='waiting_prior_queue'


def test_only_actual_success_releases_predecessor():
    physical={'expected':10,'completed':10,'pending':0,'running':0,'failed':0}
    assert predecessor_complete('continuous_ofat',{'physical':physical})
    assert not predecessor_complete('continuous_ofat',{'physical':dict(physical,failed=1)})
    assert not predecessor_complete('real_a',{'jobs':{}})
    assert predecessor_complete('real_a',{'jobs':{'a':{'status':'succeeded'}}})
    assert not predecessor_complete('real_a',{'jobs':{'a':{'status':'failed'}}})


def test_readiness_rejects_changed_config(tmp_path):
    (tmp_path/'frozen_config.yaml').write_text('changed: true\n')
    (tmp_path/'READY.json').write_text(json.dumps({'status':'queue_ready','tests_passed':True,
                                                 'frozen_config_sha256':'wrong'}))
    assert verify_ready(tmp_path)[0] is False


def test_exit_zero_does_not_erase_unresolved_or_stopped_status():
    assert completion_stage(0,{'stage':'complete'})=='complete'
    assert completion_stage(0,{'stage':'complete_with_unresolved'})=='complete_with_unresolved'
    assert completion_stage(0,{'stage':'stopped_resumable'})=='stopped_resumable'
    assert completion_stage(0,{})=='run_finished_without_completion_status'
    assert completion_stage(1,{'stage':'complete'})=='run_failed'
