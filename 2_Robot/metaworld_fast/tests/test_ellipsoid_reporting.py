import json
from pathlib import Path

import numpy as np

from metaworld_fast.ellipsoid_reporting import endpoint_tables, generate_ellipsoid_report, load_artifacts, mean_curves


def fixture(tmp_path, evaluations=10):
    source = tmp_path / "original"
    target = tmp_path / "reevaluation"
    (source / "statistical").mkdir(parents=True)
    (source / "figures").mkdir()
    (source / "figures/old.png").write_bytes(b"original figure unchanged")
    (target / "evaluated").mkdir(parents=True)
    (target / "manifest.json").write_text(json.dumps(dict(source_run=str(source))))
    for interaction in range(10):
        name = f"m04_b04__RC__i{interaction}.npz"
        np.savez(source / "statistical" / name, q=np.zeros((128,4,4)), y=np.zeros(128), mu=np.zeros((129,4)),
                 cov=np.tile(np.eye(4)/3,(129,1,1)), risk=np.arange(129,dtype=float)+interaction,
                 error=np.arange(129,dtype=float)*2, eta=np.full(128,2.), eig=np.full(128,.2))
        if interaction < evaluations:
            ratio = np.full(128, .2+interaction/100)
            np.savez(target / "evaluated" / name, t=np.arange(128), eig_actual=ratio,
                     eig_grid=np.full(128,.4), eig_achieved=np.ones(128), eig_upper=np.full(128,np.log(4)),
                     eta_ellipsoid_approx=ratio, eta_grid_recomputed=ratio/.4,
                     eta_lower_plugin=ratio/np.log(4), eta_upper_plugin=ratio,
                     achieved_q=np.zeros((128,4,4)), actual_inside=np.ones(128,dtype=bool))
    return source,target


def test_ten_seed_ratio_average_and_endpoint_indexing(tmp_path):
    _, target=fixture(tmp_path)
    _,_,_,runs=load_artifacts(target)
    individuals,mean,n=mean_curves(runs,"m04_b04","RC","eta_ellipsoid_approx")
    assert individuals.shape==(10,128)
    assert np.all(n==10)
    assert np.allclose(mean,.245)
    _,means=endpoint_tables(target,runs)
    risk=next(r for r in means if r["metric"]=="risk" and r["answer_count"]==32)
    assert risk["mean"]==36.5
    assert risk["n_effective"]==10


def test_incomplete_reevaluation_keeps_old_grid_and_risk_read_only(tmp_path):
    source,target=fixture(tmp_path,evaluations=9)
    original_bytes=(source/"figures/old.png").read_bytes()
    _,_,_,runs=load_artifacts(target)
    _,mean,n=mean_curves(runs,"m04_b04","RC","eta_ellipsoid_approx")
    assert np.all(n==9) and np.allclose(mean,.24)
    _,oldmean,oldn=mean_curves(runs,"m04_b04","RC","eta_grid_original")
    assert np.all(oldmean==2.) and np.all(oldn==10)
    report=generate_ellipsoid_report(target,make_plots=False)
    assert report["complete_evaluated_histories"]==9
    assert report["status"]=="incomplete_or_unresolved"
    assert report["new_actual_answers"]==0 and not report["global_optimum_certified"]
    assert (source/"figures/old.png").read_bytes()==original_bytes
    assert "not a certified exact global-oracle ratio" in (target/"REPORT.md").read_text()


def test_geometry_and_value_inclusion_failures_are_reported(tmp_path):
    _,target=fixture(tmp_path,evaluations=1)
    path=target/"evaluated/m04_b04__RC__i0.npz"
    with np.load(path) as archive:
        data={k:archive[k] for k in archive.files}
    data["achieved_q"][0,0,0]=100.
    data["eig_achieved"][1]=.1
    np.savez(path,**data)
    result=generate_ellipsoid_report(target,make_plots=False)
    assert result["diagnostic_violations"]==3
