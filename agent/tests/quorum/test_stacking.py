"""Tests for Quorum's fixed-penalty ridge stacker."""

from __future__ import annotations

import numpy as np
import pytest

from src.quorum.stacking import RidgeStacker


def _data(n: int = 2_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, (n, 3))
    target = 0.01 + features @ np.array([0.02, 0.0, -0.03]) + rng.normal(0, 0.01, n)
    return features, target


def test_ridge_recovers_coefficient_signs_and_unpenalized_intercept() -> None:
    features, target = _data()
    fitted = RidgeStacker(alpha=1e-6).fit(features, target)

    raw_coef = fitted.coef / fitted.scale  # back to unstandardized feature units
    assert raw_coef == pytest.approx([0.02, 0.0, -0.03], abs=2e-3)
    assert fitted.intercept == pytest.approx(target.mean())


def test_penalty_shrinks_toward_zero_without_changing_signs() -> None:
    features, target = _data()
    loose = RidgeStacker(alpha=1e-6).fit(features, target)
    tight = RidgeStacker(alpha=10.0).fit(features, target)

    assert np.abs(tight.coef).sum() < np.abs(loose.coef).sum()
    assert np.sign(tight.coef[[0, 2]]).tolist() == np.sign(loose.coef[[0, 2]]).tolist()


def test_scores_are_monotone_bounded_and_deterministic() -> None:
    features, target = _data()
    fitted = RidgeStacker(alpha=0.1).fit(features, target)
    again = RidgeStacker(alpha=0.1).fit(features.copy(), target.copy())

    raw = fitted.expected_return(features)
    scores = fitted.predict(features)
    assert np.all(np.abs(scores) < 1.0)
    assert np.array_equal(
        np.argsort(raw, kind="stable"), np.argsort(scores, kind="stable")
    )
    assert np.array_equal(scores, again.predict(features))


def test_fit_fails_closed_on_non_finite_or_too_few_rows() -> None:
    features, target = _data(n=10)
    with pytest.raises(ValueError, match="rows"):
        RidgeStacker().fit(features[:3], target[:3])
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RidgeStacker().fit(features, target)
    with pytest.raises(ValueError, match="alpha"):
        RidgeStacker(alpha=-1.0)
