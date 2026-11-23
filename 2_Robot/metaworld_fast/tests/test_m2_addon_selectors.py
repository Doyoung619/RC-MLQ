"""CPU-only correctness checks for the frozen M=2 addon selector contract."""
import copy
import itertools
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from metaworld_fast.m2_addon_selectors import (
    NEW_METHODS, GCQ_STAGES, INITIAL_STD, AddonSelectorState,
    stage_schedule, stage_at, mem_objective, solve_mem, solve_moving_center,
    solve_gmem, solve_two_mem, binary_entropy_from_logits,
    ib_conditional_entropy, ib_raw_conditional_entropy,
    minimize_bounded_log_objective, select_ib, select_query,
)
from metaworld_fast.model import coordinate_of_query, log_probabilities
from metaworld_fast.posterior_marginal import MarginalPosterior
from metaworld_fast.selectors import sg_stage


@pytest.mark.parametrize("stages,lengths,starts", [
    (1, [128], [0]), (3, [42, 42, 44], [0, 42, 84]),
    (5, [25, 25, 25, 25, 28], [0, 25, 50, 75, 100]),
])
def test_author_code_schedule_lengths_and_all_boundaries(stages, lengths, starts):
    actual_lengths, actual_starts = stage_schedule(128, stages)
    np.testing.assert_array_equal(actual_lengths, lengths)
    np.testing.assert_array_equal(actual_starts, starts)
    for stage, (start, length) in enumerate(zip(starts, lengths)):
        assert stage_at(start, 128, stages) == stage
        assert stage_at(start+length-1, 128, stages) == stage
    for invalid in (-1, 128, 3.5):
        with pytest.raises(ValueError):
            stage_at(invalid, 128, stages)
    assert [sg_stage(t) for t in range(128)] == [0]*128


@pytest.mark.parametrize("method", list(GCQ_STAGES))
def test_gcq_fixed_gaussian_translation_centers_and_checkpoint_resume(method):
    """Independent fixed-input MATLAB-equivalent schedule translation check."""
    stages = GCQ_STAGES[method]
    expected_lengths = [128//stages]*stages
    expected_lengths[-1] += 128-sum(expected_lengths)
    starts = np.r_[0, np.cumsum(expected_lengths[:-1])]
    rng = np.random.default_rng(44)
    reference = np.random.default_rng(44)
    state = AddonSelectorState()
    for t in range(128):
        mean = np.array([.001*t, -.002*t, .003*t, -.001*t])
        template = SimpleNamespace(mean=mean, covariance=np.eye(4)/3,
                                   direction=np.array([1., 0., 0., 0.]))
        j = int(np.searchsorted(starts, t, side="right")-1)
        center = np.array([.001*starts[j], -.002*starts[j], .003*starts[j], -.001*starts[j]])
        fixed_normal = reference.standard_normal((2, 4))
        expected = center+INITIAL_STD*2.**(-j)*fixed_normal
        raw, q = select_query(method, None, 2, t, rng, state, template)
        np.testing.assert_array_equal(raw, expected)
        np.testing.assert_allclose(q, raw, atol=2e-16)
        np.testing.assert_array_equal(state.center, center)
        assert state.std == INITIAL_STD*2.**(-j)
        # Save and restore both sides of every boundary without changing draws.
        if t in set(starts) | set(starts[1:]-1):
            state, rng = pickle.loads(pickle.dumps((state, rng)))
    assert not state.diagnostics["cap_active"][0]


def test_gcq_resume_requires_saved_state_and_consistent_horizon():
    p = MarginalPosterior()
    with pytest.raises(ValueError, match="midstage resume"):
        select_query("gausscloud_paper_L3", p, 2, 10, np.random.default_rng(1), AddonSelectorState())
    state = AddonSelectorState()
    select_query("gausscloud_paper_L3", p, 2, 0, np.random.default_rng(1), state)
    with pytest.raises(ValueError, match="horizon mismatch"):
        select_query("gausscloud_paper_L3", p, 2, 1, np.random.default_rng(1), state, horizon=256)


def test_gcq_current_mean_cap_and_shared_state_no_diagnostic_mutation():
    p = MarginalPosterior()
    state = AddonSelectorState(stage=0, center=np.full(4, 100.), method="gausscloud_paper_L3", horizon=128)
    before = pickle.dumps(state)
    clone = copy.deepcopy(state)
    raw, q = select_query("gausscloud_paper_L3", p, 2, 3, np.random.default_rng(2), clone)
    assert pickle.dumps(state) == before
    mean, _ = p.moments()
    np.testing.assert_allclose(np.linalg.norm(q-mean, axis=1), 32, atol=1e-13)
    assert all(clone.diagnostics["cap_active"])
    expected = np.broadcast_to(32/np.linalg.norm(raw-mean, axis=1)[:, None], raw.shape)
    np.testing.assert_allclose((q-mean)/(raw-mean), expected)


def test_sm_hyperplane_reflection_preserves_tangent_and_midpoint_bisector():
    mean = np.array([.1, -.2, .3, -.4])
    cov = np.array([[.4, .1, 0., 0.], [.1, .2, 0., 0.], [0., 0., .1, 0.], [0., 0., 0., .05]])
    p = SimpleNamespace(moments=lambda: (mean, cov), factorized=False)
    state = AddonSelectorState()
    raw, displayed = select_query("sm_continuous", p, 2, 0, np.random.default_rng(1), state)
    v = np.array(state.diagnostics["direction"])
    z1, z2 = raw-mean
    np.testing.assert_allclose(z2, z1-2*(v@z1)*v, atol=2e-16)
    np.testing.assert_allclose(z1-(v@z1)*v, z2-(v@z2)*v, atol=2e-16)
    assert np.linalg.norm(z1+z2) > .05  # Not point reflection mu-z.
    np.testing.assert_allclose(v @ (raw.mean(axis=0)-mean), 0, atol=1e-16)
    np.testing.assert_allclose(np.linalg.norm(z1), np.linalg.norm(z2), atol=1e-15)
    np.testing.assert_allclose(displayed, raw, atol=2e-16)


def test_sm_cap_same_radial_factor_preserves_coordinate_factorization():
    mean = np.zeros(4)
    cov = np.diag([1e6, 2e5, 1e5, 1e4])
    p = SimpleNamespace(moments=lambda: (mean, cov), factorized=True)
    state = AddonSelectorState()
    raw, q = select_query("sm_continuous", p, 2, 0, np.random.default_rng(11), state)
    assert all(state.diagnostics["cap_active"])
    assert coordinate_of_query(q) == 0
    factors = np.linalg.norm(q, axis=1)/np.linalg.norm(raw, axis=1)
    np.testing.assert_allclose(factors[0], factors[1], atol=1e-16)
    assert np.linalg.norm(q[0]+q[1]) > .01


def test_mem_global_solver_exhaustive_all_small_assignments():
    x = np.array([-.95, -.6, -.05, .22, .7, .91])
    w = np.array([.09, .31, .07, .19, .13, .21])
    pair, loss = solve_mem(x, w)
    brute = np.inf
    for assignment in itertools.product((0, 1), repeat=len(x)):
        mask = np.array(assignment, dtype=bool)
        if mask.all() or not mask.any():
            continue
        a = np.average(x[mask], weights=w[mask])
        b = np.average(x[~mask], weights=w[~mask])
        brute = min(brute, mem_objective(x, w, a, b))
    assert abs(loss-brute) < 1e-14
    assert pair[0] <= pair[1]
    np.testing.assert_array_equal(solve_mem(x[::-1], w[::-1])[0], pair)


@pytest.mark.parametrize("fixed", [-1., -.7, 0., .43, 1.])
def test_moving_center_global_solver_exhaustive_dense_grid(fixed):
    x = np.array([-.99, -.45, -.13, .08, .36, .92])
    w = np.array([.05, .14, .27, .09, .15, .3])
    value, loss = solve_moving_center(x, w, fixed)
    grid = np.linspace(-1, 1, 20001)
    losses = np.minimum((x-fixed)**2, (x[None, :]-grid[:, None])**2) @ w
    assert loss <= losses.min()+1e-13
    assert -1 <= value <= 1
    assert abs(loss-mem_objective(x, w, fixed, value)) < 1e-15


def test_mem_uniform_limits_degenerate_zero_mass_and_ties():
    p = MarginalPosterior(nodes=513)
    m = p.marginals[0]
    pair, _ = solve_mem(m.x, m.weights)
    np.testing.assert_allclose(pair, [-.5, .5], atol=.002)
    greedy, _ = solve_gmem(m.x, m.weights, 0.)
    assert greedy[0] == 0.
    assert abs(abs(greedy[1])-2/3) < .002
    np.testing.assert_array_equal(solve_mem([.2, .2, -.9], [1, 2, 0])[0], [.2, .2])
    np.testing.assert_array_equal(solve_mem([-1, 1], [1, 1])[0], [-1, 1])
    for bad in ([1, -1], [0, 0]):
        with pytest.raises(ValueError):
            solve_mem([-1, 1], bad)


def test_two_mem_exactly_two_sweeps_and_frozen_objective_nonincrease():
    p = MarginalPosterior()
    q = np.array([[-.6, 0, 0, 0], [.8, 0, 0, 0]])
    p.update(q, 0)
    marginal = p.marginals[0]
    before = pickle.dumps(p)
    pair, details = solve_two_mem(marginal.x, marginal.weights, marginal.mean)
    assert details["slot_order"] == [0, 1, 0, 1]
    assert len(details["objectives"]) == 5
    assert np.all(np.diff(details["objectives"]) <= 5e-13)
    np.testing.assert_array_equal(pair, details["centers"][-1])
    assert pickle.dumps(p) == before


def test_it_uses_trace_of_all_dimensions():
    p = MarginalPosterior()
    state = AddonSelectorState()
    raw, q = select_query("infosynth_trace", p, 2, 0, np.random.default_rng(1), state)
    mean, cov = p.moments()
    np.testing.assert_allclose(np.linalg.norm(q-mean, axis=1), np.sqrt(4/3), atol=1e-14)
    assert state.diagnostics["halfspacing"] > 1.9*np.sqrt(cov[0, 0])
    np.testing.assert_array_equal(raw, q)


def test_ib_normalization_entropy_sign_and_stability():
    draws = np.random.default_rng(5).normal(size=(4096, 4))*np.array([.2, .4, .6, .8])
    s = np.sqrt(1.2)
    direction = np.array([0., 0., 0., 1.])
    for relative in (.01, .1, 1., 10.):
        normalized = ib_conditional_entropy(relative, draws/s, direction)
        raw = ib_raw_conditional_entropy(s*relative, draws, direction)
        assert abs(normalized-raw) < 2e-16
        assert 0 <= raw <= np.log(2)
    entropy = binary_entropy_from_logits([0., 1., -1., 1e6, -1e6])
    np.testing.assert_allclose(entropy, [np.log(2), entropy[1], entropy[1], 0, 0])
    assert 0 < entropy[1] < entropy[0]


def test_ib_minimizer_bounds_both_endpoints_and_all_local_minima():
    for slope, expected in ((1, .01), (-1, 10.)):
        seen = []
        def objective(x):
            seen.append(x)
            assert .01-1e-15 <= x <= 10.+1e-14
            return slope*x
        selected, details = minimize_bounded_log_objective(objective, .01, 10.)
        assert abs(selected-expected) < 1e-14
    selected, details = minimize_bounded_log_objective(lambda x: (np.log(x)**2-1)**2+.01*np.log(x), .01, 10.)
    assert len(details["local_minimum_indices"]) == 2
    assert len(details["refinement_status"]) == 2
    assert selected < 1.


def test_ib_reproducible_common_sobol_draws_and_physical_bound():
    cov = np.diag([.01, .03, .08, .4])
    direction = np.array([0., 0., 0., 1.])
    spacing, metadata = select_ib(cov, direction, np.random.default_rng(37))
    spacing2, metadata2 = select_ib(cov, direction, np.random.default_rng(37))
    assert spacing == spacing2 and metadata == metadata2
    assert .01 <= metadata["relative_halfspacing"] <= 10
    assert spacing <= 32
    assert metadata["selected_integration_points"] in (1024, 4096)
    huge_spacing, huge_metadata = select_ib(cov*1000, direction, np.random.default_rng(37))
    assert huge_spacing <= 32+1e-13
    assert huge_metadata["relative_bounds"][1] == 32/np.sqrt(np.trace(cov*1000))


@pytest.mark.parametrize("method", NEW_METHODS)
def test_real_posterior_frozen_selection_binary_guard_and_no_model_change(method):
    p = MarginalPosterior()
    before = pickle.dumps(p)
    rng = np.random.default_rng(89)
    state = AddonSelectorState()
    raw, q = select_query(method, p, 2, 0, rng, state)
    assert raw.shape == q.shape == (2, 4)
    assert np.isfinite(q).all()
    assert pickle.dumps(p) == before
    actual = log_probabilities(np.array([.1, .2, -.3, .4]), q, p.beta)
    direct = -4*np.sum((q-np.array([.1, .2, -.3, .4]))**2, axis=1)
    np.testing.assert_allclose(actual[1]-actual[0], direct[1]-direct[0], atol=2e-14)
    if method not in GCQ_STAGES:
        coordinate_of_query(q)  # Actual displayed geometry supports marginal update.
        p.update(q, 0)
        assert len(p.history) == 1
        select_query(method, p, 2, 1, rng, state)
    with pytest.raises(ValueError, match="M=2"):
        select_query(method, p, 4, 0, rng, state)
    with pytest.raises(ValueError, match="dummy"):
        select_query(method, p, 2, 128, rng, state)
