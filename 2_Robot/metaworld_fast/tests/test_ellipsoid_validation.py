from metaworld_fast.ellipsoid_validation import numerical_comparison, optimizer_comparison


def test_independent_reference_uncertainty_and_ratio_tolerance():
    result=numerical_comparison(.51,[.50,.50],absolute_tolerance=.005)
    assert result["decision"]=="unresolved" and result["reference_se"]==0
    result=numerical_comparison(.51,[.49,.51],absolute_tolerance=.005)
    assert result["decision"]=="within_reference_uncertainty"
    assert result["reference_se"]>0
    assert "not a confidence certificate" in result["interpretation"]


def test_refinement_preserves_original_and_reports_remaining_global_gap():
    result=optimizer_comparison(.7,.75,1.)
    assert result["warm_witness_retained"] and result["budget_sensitive"]
    assert result["remaining_gap"]==.25
    assert not result["global_optimum_certified"]
    assert not optimizer_comparison(.7,.6,1.)["warm_witness_retained"]
