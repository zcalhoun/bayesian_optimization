"""Small, sequential Bayesian optimization example using a Gaussian process.

Original implementation by Zach Calhoun, with help from Claude.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class Trial:
    """One proposed parameter set and its observed objective value."""

    def __init__(self, parameters: dict[str, float], trial_type: str = "sobol"):
        self.parameters = parameters
        self.type = trial_type
        self.result: float | None = None
        self.skipped = False

    def update(self, result: float) -> None:
        if self.skipped:
            raise RuntimeError("Cannot update a skipped trial.")
        if self.result is not None:
            raise RuntimeError("This trial already has a result.")
        if not np.isfinite(result):
            raise ValueError("The result must be finite.")
        self.result = float(result)

    def skip(self) -> None:
        if self.result is not None:
            raise RuntimeError("Cannot skip a completed trial.")
        self.skipped = True


class BayesianOptimization:
    """Suggest points from a fixed Sobol pool, then rank them by expected improvement.

    Complete or skip each suggested trial before requesting another one. The
    objective is supplied by the caller through ``Trial.update``.
    """

    def __init__(
        self,
        variables: dict[str, tuple[float, float]],
        minimize: bool = True,
        random_inits: int = 5,
        n_candidates: int = 60000,
        random_state: int | None = None,
    ):
        if not variables:
            raise ValueError("At least one variable is required.")
        self.variables = dict(variables)
        bounds = np.asarray(list(self.variables.values()), dtype=float)
        if bounds.shape != (len(variables), 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("Each variable needs two finite bounds.")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each lower bound must be below its upper bound.")
        if random_inits < 1 or n_candidates < random_inits:
            raise ValueError("Require 1 <= random_inits <= n_candidates.")

        self.lower, self.upper = bounds.T
        self.minimize = minimize
        self.random_inits = random_inits
        self.trials: list[Trial] = []
        self._used_indices: set[int] = set()
        self._sobol_idx = 0
        self.random_state = random_state
        self.X_candidates = self._create_candidates(n_candidates)

    def _create_candidates(self, n_candidates: int) -> np.ndarray:
        # A power of two preserves the balance property of a Sobol sequence.
        count = 2 ** math.ceil(math.log2(max(n_candidates, 2)))
        samples = Sobol(
            d=len(self.variables), scramble=True, seed=self.random_state
        ).random_base2(int(math.log2(count)))
        return self.lower + samples * (self.upper - self.lower)

    def _to_unit_cube(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lower) / (self.upper - self.lower)

    def _completed(self) -> list[Trial]:
        return [t for t in self.trials if t.result is not None and not t.skipped]

    def _collect_trials(self) -> tuple[np.ndarray, np.ndarray]:
        completed = self._completed()
        X = np.asarray(
            [[t.parameters[name] for name in self.variables] for t in completed]
        )
        y = np.asarray([t.result for t in completed])
        return X, y

    def fit_surrogate(self) -> GaussianProcessRegressor:
        """Fit a GP to completed trials; its inputs use normalized bounds."""
        X, y = self._collect_trials()
        if not len(y):
            raise RuntimeError("Complete a trial before fitting the surrogate.")
        return _fit_gp(self._to_unit_cube(X), y, self.random_state)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return GP mean and standard deviation at points in original units."""
        points = np.asarray(X, dtype=float).reshape(-1, len(self.variables))
        return self.fit_surrogate().predict(self._to_unit_cube(points), return_std=True)

    def acquisition(self, X: np.ndarray) -> np.ndarray:
        """Return expected improvement at points in original units."""
        points = np.asarray(X, dtype=float).reshape(-1, len(self.variables))
        return expected_improvement(
            self._to_unit_cube(points),
            self.fit_surrogate(),
            self.get_best_trial().result,
            minimize=self.minimize,
        )

    def get_next_trial(self) -> Trial:
        if any(t.result is None and not t.skipped for t in self.trials):
            raise RuntimeError("Complete or skip the pending trial first.")

        if len(self._completed()) < self.random_inits:
            if self._sobol_idx >= len(self.X_candidates):
                raise RuntimeError("The candidate pool is exhausted.")
            index = self._sobol_idx
            self._sobol_idx += 1
            trial_type = "sobol"
        else:
            available = np.array(
                [i for i in range(len(self.X_candidates)) if i not in self._used_indices]
            )
            if not len(available):
                raise RuntimeError("The candidate pool is exhausted.")
            gp = self.fit_surrogate()
            best = self.get_best_trial().result
            ei = expected_improvement(
                self._to_unit_cube(self.X_candidates[available]),
                gp,
                best,
                minimize=self.minimize,
            )
            index = int(available[np.argmax(ei)])
            trial_type = "ei"

        self._used_indices.add(index)
        point = self.X_candidates[index]
        trial = Trial(
            {name: float(point[i]) for i, name in enumerate(self.variables)},
            trial_type=trial_type,
        )
        self.trials.append(trial)
        return trial

    def get_best_trial(self) -> Trial:
        """Return the best completed trial so far."""
        completed = self._completed()
        if not completed:
            raise RuntimeError("No completed trials to select from.")
        values = [t.result for t in completed]
        return completed[int(np.argmin(values) if self.minimize else np.argmax(values))]


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    f_best: float,
    minimize: bool = True,
) -> np.ndarray:
    """Expected positive improvement over the best observed value."""
    mu, sigma = gp.predict(X, return_std=True)
    improvement = f_best - mu if minimize else mu - f_best
    with np.errstate(divide="ignore", invalid="ignore"):
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma > 1e-9, ei, np.maximum(improvement, 0.0))


def _fit_gp(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> GaussianProcessRegressor:
    # A fixed length scale keeps the sparse-data example interpretable in 2D.
    # Inputs have already been scaled to [0, 1] in each dimension.
    kernel = kernels.ConstantKernel(1.0) * kernels.RBF(
        length_scale=np.full(X.shape[1], 0.25),
        length_scale_bounds="fixed",
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )
    gp.fit(X, y)
    return gp
