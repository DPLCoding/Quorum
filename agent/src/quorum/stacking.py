"""Fixed-penalty ridge stacker over expert scores.

The first Quorum model that learns parameters. It is fitted per fold on that
fold's purged training rows only; the caller owns row selection. The penalty is
a declared research parameter, never tuned on evaluation rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class FittedRidge:
    """Immutable fitted stacker: standardization, coefficients, and score scale."""

    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    score_scale: float
    train_rows: int

    def expected_return(self, features: np.ndarray) -> np.ndarray:
        """Return the fitted forward-return estimate for each feature row."""
        return self.intercept + ((features - self.mean) / self.scale) @ self.coef

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return scores in (-1, 1): evidence relative to the training mean.

        Zero means no view beyond the training-period average return (drift);
        the transform is monotone, so rank IC equals that of the raw estimate.
        """
        if self.score_scale == 0.0:
            return np.zeros(len(features))
        deviation = self.expected_return(features) - self.intercept
        return np.tanh(deviation / self.score_scale)

    def to_dict(self, feature_names: tuple[str, ...]) -> dict[str, Any]:
        """Return a JSON-safe model card for one fold."""
        return {
            "alpha": self.alpha,
            "train_rows": self.train_rows,
            "intercept": self.intercept,
            "score_scale": self.score_scale,
            "standardized_coef": dict(zip(feature_names, map(float, self.coef))),
            "feature_mean": dict(zip(feature_names, map(float, self.mean))),
            "feature_scale": dict(zip(feature_names, map(float, self.scale))),
        }


@dataclass(frozen=True, slots=True)
class RidgeStacker:
    """Ridge regression of forward return on standardized expert scores.

    Minimizes ``||y - b - Z w||^2 + alpha * n * ||w||^2`` with ``Z`` the
    training-standardized features, so ``alpha`` is scale-free in both the
    feature units and the row count. The intercept is not penalized.
    """

    alpha: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")

    def fit(self, features: np.ndarray, target: np.ndarray) -> FittedRidge:
        features = np.asarray(features, dtype="float64")
        target = np.asarray(target, dtype="float64")
        if features.ndim != 2 or target.shape != (len(features),):
            raise ValueError("features must be (rows, k) and target (rows,)")
        if not (np.isfinite(features).all() and np.isfinite(target).all()):
            raise ValueError("features and target must be finite")
        rows, width = features.shape
        if rows < 2 * (width + 1):
            raise ValueError(f"ridge needs at least {2 * (width + 1)} training rows")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0  # constant feature: centered to zero, no effect
        z = (features - mean) / scale
        intercept = float(target.mean())
        gram = z.T @ z + self.alpha * rows * np.eye(width)
        coef = np.linalg.solve(gram, z.T @ (target - intercept))
        fitted_deviation = z @ coef
        return FittedRidge(
            alpha=self.alpha,
            mean=mean,
            scale=scale,
            coef=coef,
            intercept=intercept,
            score_scale=float(fitted_deviation.std()),
            train_rows=rows,
        )


__all__ = ["FittedRidge", "RidgeStacker"]
