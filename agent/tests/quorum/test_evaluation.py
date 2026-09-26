"""Known-answer tests for Quorum's out-of-fold prediction evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.quorum.evaluation import evaluate_predictions

UTC = timezone.utc
RETURNS = (0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.05, -0.03)


def _rows(predictor: str, scores: list[float], *, asset: str = "A") -> list[dict]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        {
            "predictor": predictor,
            "asset": asset,
            "event_at": t0 + timedelta(days=i),
            "fold_index": i // 4,
            "role": "test",
            "score": score,
            "forward_return": RETURNS[i],
        }
        for i, score in enumerate(scores)
    ]


def _evaluate(*groups: list[dict], slots: int = 8) -> dict:
    frame = pd.DataFrame([row for group in groups for row in group])
    return evaluate_predictions(frame, slot_counts={"test": slots})


def test_perfect_and_inverted_predictors_have_known_ic_and_hit_rate() -> None:
    report = _evaluate(
        _rows("perfect", [r * 10 for r in RETURNS]),
        _rows("inverted", [-r * 10 for r in RETURNS]),
    )
    perfect = report["roles"]["test"]["predictors"]["perfect"]
    inverted = report["roles"]["test"]["predictors"]["inverted"]

    assert perfect["ic"] == pytest.approx(1.0)
    assert perfect["hit_rate"] == 1.0
    # Hit rate needs the base rate of rising outcomes to be interpretable.
    assert perfect["up_rate"] == 0.5
    assert perfect["fold_count"] == 2
    assert perfect["fold_ic_mean"] == pytest.approx(1.0)
    assert inverted["ic"] == pytest.approx(-1.0)
    assert inverted["hit_rate"] == 0.0
    assert report["roles"]["test"]["score_correlation"]["perfect"][
        "inverted"
    ] == pytest.approx(-1.0)


def test_constant_scores_report_no_ic_or_hit_rate_instead_of_nan() -> None:
    report = _evaluate(_rows("flat", [0.0] * 8))
    flat = report["roles"]["test"]["predictors"]["flat"]

    assert flat["ic"] is None
    assert flat["hit_rate"] is None
    assert flat["hit_count"] == 0
    json.dumps(report, allow_nan=False)


def test_coverage_saturation_and_per_asset_ic() -> None:
    clipped = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5]
    report = _evaluate(
        _rows("partial", clipped, asset="A"),
        slots=8,
    )
    partial = report["roles"]["test"]["predictors"]["partial"]

    assert partial["count"] == 6
    assert partial["coverage"] == pytest.approx(0.75)
    assert partial["saturation"] == pytest.approx(4 / 6)
    assert report["roles"]["test"]["asset_ic"]["partial"]["A"] == pytest.approx(
        partial["ic"]
    )


def test_sign_agreement_and_regime_ic() -> None:
    perfect = _rows("perfect", [r * 10 for r in RETURNS])
    inverted = _rows("inverted", [-r * 10 for r in RETURNS])
    for group in (perfect, inverted):
        for i, row in enumerate(group):
            row["regime"] = "high" if i % 2 else "low"
    report = _evaluate(perfect, inverted)
    test = report["roles"]["test"]

    assert test["sign_agreement"]["perfect"]["inverted"] == 0.0
    assert test["sign_agreement"]["perfect"]["perfect"] == 1.0
    assert test["regime_ic"]["perfect"] == {
        "high": pytest.approx(1.0),
        "low": pytest.approx(1.0),
    }


def test_paired_fold_delta_compares_the_same_folds() -> None:
    from src.quorum.evaluation import paired_fold_delta

    noisy = [0.1, -0.2, 0.3, 0.1, 0.2, -0.1, -0.4, -0.3]
    frame = pd.DataFrame(
        _rows("base", noisy) + _rows("better", [r * 10 for r in RETURNS])
    )
    delta = paired_fold_delta(frame, base="base", other="better", role="test")

    assert delta["fold_count"] == 2
    assert delta["positive_fraction"] == 1.0
    assert delta["mean_delta"] > 0.0
    assert set(delta) == {
        "fold_count",
        "mean_delta",
        "t",
        "positive_fraction",
        "fold_deltas",
    }


# ------------------------------------------------ Ledoit-Wolf Sharpe difference


def _paired_null(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-Sharpe paired returns with shared volatility clustering and 0.8 corr."""
    log_vol = np.zeros(n)
    for t in range(1, n):
        log_vol[t] = 0.95 * log_vol[t - 1] + 0.25 * rng.standard_normal()
    sigma = 0.01 * np.exp(log_vol - log_vol.mean())
    z = rng.standard_normal(n)
    e1, e2 = rng.standard_normal(n), rng.standard_normal(n)
    a = 0.0004 + sigma * (0.8 * z + 0.6 * e1)
    b = 0.0004 + sigma * (0.8 * z + 0.6 * e2)
    return a, b


def test_sharpe_difference_test_is_deterministic_and_matches_sample_sharpes() -> None:
    from src.quorum.evaluation import sharpe_difference_test

    a, b = _paired_null(np.random.default_rng(1), 300)
    first = sharpe_difference_test(a, b, resamples=499, seed=7)
    again = sharpe_difference_test(a, b, resamples=499, seed=7)
    assert first == again
    daily = a.mean() / a.std() - b.mean() / b.std()
    assert first["sharpe_difference_daily"] == pytest.approx(daily)
    assert first["sharpe_difference_annualized"] == pytest.approx(daily * 252**0.5)
    assert first["block_length"] == round(300 ** (1 / 3))
    assert 0.0 < first["p_value_one_sided"] <= 1.0


def test_sharpe_difference_test_holds_size_under_a_dependent_null() -> None:
    from src.quorum.evaluation import sharpe_difference_test

    rng = np.random.default_rng(2026)
    rejections = sum(
        sharpe_difference_test(*_paired_null(rng, 250), resamples=299, seed=i)[
            "p_value_one_sided"
        ]
        <= 0.05
        for i in range(120)
    )
    # Nominal 5%; binomial sd over 120 draws is ~2%. JK-style IID tests drift.
    assert rejections / 120 <= 0.12


def test_sharpe_difference_test_detects_a_real_improvement() -> None:
    from src.quorum.evaluation import sharpe_difference_test

    rng = np.random.default_rng(3)
    base, _ = _paired_null(rng, 500)
    better = base + 0.0015  # same risk, clearly higher mean
    result = sharpe_difference_test(better, base, resamples=999, seed=1)
    assert result["sharpe_difference_annualized"] > 1.0
    assert result["p_value_one_sided"] < 0.01
