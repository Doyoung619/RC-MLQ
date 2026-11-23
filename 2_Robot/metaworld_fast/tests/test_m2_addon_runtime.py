"""Admission and cached-failure checks for an unattended incremental run."""
import json
from pathlib import Path

import pytest

from metaworld_fast import m2_addon_cli as cli
from metaworld_fast import m2_addon_runner as runner


def test_delayed_admission_cannot_be_bypassed_by_queue_token(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'load_config', lambda _: {'addon': {'not_before_utc': '2999-01-01T00:00:00+00:00'}})
    monkeypatch.setenv('M2_ADDON_QUEUE_ADMITTED', str(tmp_path.resolve()))
    with pytest.raises(RuntimeError, match='cannot begin before'):
        cli.require_admission(tmp_path)


def test_admission_is_bound_to_output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'load_config', lambda _: {'addon': {'not_before_utc': '2000-01-01T00:00:00+00:00'}})
    monkeypatch.delenv('M2_ADDON_QUEUE_ADMITTED', raising=False)
    with pytest.raises(RuntimeError, match='through m2_addon_queue'):
        cli.require_admission(tmp_path)
    monkeypatch.setenv('M2_ADDON_QUEUE_ADMITTED', str(tmp_path/'different'))
    with pytest.raises(RuntimeError, match='through m2_addon_queue'):
        cli.require_admission(tmp_path)
    monkeypatch.setenv('M2_ADDON_QUEUE_ADMITTED', str(tmp_path.resolve()))
    cli.require_admission(tmp_path)


def test_failed_physical_preflight_is_not_accepted_as_cached_pass(tmp_path, monkeypatch):
    from metaworld_fast import contexts, renderer
    (tmp_path/'validation').mkdir()
    (tmp_path/'validation'/'physical_preflight.json').write_text(json.dumps({'status':'unresolved', 'records':[]}))
    monkeypatch.setattr(runner, 'load_config', lambda _: {'experiment':{'interaction_ids':[0]},'addon':{'context_dir':'dummy'}})
    monkeypatch.setattr(contexts, 'load_contexts', lambda _: [])
    def retried(_):
        raise RuntimeError('preflight retried')
    monkeypatch.setattr(renderer, 'RobotAdapter', retried)
    with pytest.raises(RuntimeError, match='preflight retried'):
        runner.preflight(tmp_path)


def test_readiness_rejects_modified_implementation(tmp_path, monkeypatch):
    (tmp_path/'frozen_config.yaml').write_text('config')
    import hashlib
    ready = dict(status='queue_ready',tests_passed=True,source_files={'selector.py':'old'},
                 frozen_config_sha256=hashlib.sha256(b'config').hexdigest())
    (tmp_path/'READY.json').write_text(json.dumps(ready))
    monkeypatch.setattr(runner, 'source_code_inventory', lambda: {'selector.py':'new'})
    with pytest.raises(RuntimeError, match='changed since readiness'):
        runner.check_ready(tmp_path)


def test_failed_or_unpublished_replay_remains_resumable(tmp_path):
    folder=tmp_path/'rollouts'; folder.mkdir()
    key='m02_b04__mem_line__i0'
    (folder/f'{key}.done.json').write_text(json.dumps({'result':{'failed':1,'logical_requests':256,'simulation_valid':255}}))
    assert not runner.replay_is_complete(tmp_path,key)
    (folder/f'{key}.npz').write_bytes(b'engineering marker')
    records=[dict(simulation_valid=True,command_reconstruction_ok=True,action_bounds_ok=True,actual_steps=50)]*256
    (folder/f'{key}.json').write_text(json.dumps({'per_candidate':records}))
    assert not runner.replay_is_complete(tmp_path,key)
    (folder/f'{key}.done.json').write_text(json.dumps({'result':{'failed':0,'logical_requests':256,'simulation_valid':256}}))
    assert runner.replay_is_complete(tmp_path,key)
    records[0]=dict(records[0],command_reconstruction_ok=False)
    (folder/f'{key}.json').write_text(json.dumps({'per_candidate':records}))
    assert not runner.replay_is_complete(tmp_path,key)
