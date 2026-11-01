"""Tests for the rate-distortion theory: closed forms vs simulation/solver."""
import numpy as np
import pytest

from rwvae import theory

D, DS, DN, RHO = 40, 6, 10, 6.0
N = 60_000  # smaller than the script, for fast tests


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("k", [DN // 2, DN, DN + DS // 2, DN + DS])
def test_pca_closed_form_matches_simulation(k, rng):
    rel_sim, r2_sim = theory.pca_simulate(N, D, DS, DN, RHO, k, rng=rng)
    assert abs(rel_sim - theory.pca_relative_error(k, DS, DN, RHO)) < 0.03
    assert abs(r2_sim - theory.pca_task_r2(k, DS, DN)) < 0.05


def test_pathology_recon_zero_but_task_zero():
    # k = d_n: relative error -> 0 as rho -> inf, yet task R^2 stays exactly 0
    assert theory.pca_relative_error(DN, DS, DN, 1e6) < 1e-4
    assert theory.pca_task_r2(DN, DS, DN) == 0.0


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.9, 1.0])
def test_reverse_water_filling_matches_closed_form(frac):
    lambdas = np.concatenate([np.ones(DS), np.full(DN, RHO)])
    Rstar = theory.rd_threshold(DN, RHO)
    _, D_i = theory.reverse_water_filling(lambdas, frac * Rstar)
    rel_solver = D_i.sum() / lambdas.sum()
    assert abs(rel_solver - theory.rd_relative_distortion(frac * Rstar, DS, DN, RHO)) < 1e-3
    # signal gets zero rate for R <= R*  => I(Z;S)=0
    signal_rate = 0.5 * np.sum(np.log(lambdas[:DS] / D_i[:DS]))
    assert signal_rate < 1e-6


def test_distortion_at_threshold():
    Rstar = theory.rd_threshold(DN, RHO)
    expected = (DS + DN) / (DS + DN * RHO)
    assert abs(theory.rd_relative_distortion(Rstar, DS, DN, RHO) - expected) < 1e-9


def test_above_threshold_signal_gets_rate():
    lambdas = np.concatenate([np.ones(DS), np.full(DN, RHO)])
    _, D_i = theory.reverse_water_filling(lambdas, 1.1 * theory.rd_threshold(DN, RHO))
    assert 0.5 * np.sum(np.log(lambdas[:DS] / D_i[:DS])) > 1e-6


def test_weighted_threshold(rng):
    w = 2.0
    assert theory.weighted_pca_threshold(w) == pytest.approx(w ** 2)
    # below rho* = w^2 weighting recovers the task; above it the task is still lost
    _, r2_lo = theory.pca_simulate(N, D, DS, DN, w ** 2 - 1.0, k=DS, weight_w=w, rng=rng)
    _, r2_hi = theory.pca_simulate(N, D, DS, DN, w ** 2 + 2.0, k=DS, weight_w=w, rng=rng)
    assert r2_lo > 0.9
    assert r2_hi < 0.1


def test_rho_emp_identity(rng):
    emp = theory.rho_emp_simulate(N, D, DS, DN, RHO, rng=rng)
    assert abs(emp - theory.rho_emp_closed(DS, DN, RHO)) / theory.rho_emp_closed(DS, DN, RHO) < 0.03
