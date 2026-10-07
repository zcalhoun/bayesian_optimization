"""Focused checks for acquisition direction and the sequential trial API."""

import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from bopt import BayesianOptimization, Trial, expected_improvement


class FakeGP:
    def predict(self, X, return_std=False):
        mean = np.asarray(X)[:, 0]
        std = np.full_like(mean, 0.2)
        return (mean, std) if return_std else mean


class CertainGP:
    def predict(self, X, return_std=False):
        mean = np.asarray(X)[:, 0]
        std = np.zeros_like(mean)
        return (mean, std) if return_std else mean


class BayesianOptimizationTests(unittest.TestCase):
    def test_ei_respects_optimization_direction(self):
        points = np.array([[0.1], [0.9]])
        minimize_ei = expected_improvement(points, FakeGP(), 0.5)
        maximize_ei = expected_improvement(points, FakeGP(), 0.5, minimize=False)
        self.assertGreater(minimize_ei[0], minimize_ei[1])
        self.assertGreater(maximize_ei[1], maximize_ei[0])
        self.assertAlmostEqual(expected_improvement(points, CertainGP(), 0.5)[0], 0.4)

    def test_trial_must_be_resolved_and_candidates_are_unique(self):
        opt = BayesianOptimization(
            {"x": (2.0, 10.0)}, random_inits=1, n_candidates=4, random_state=5
        )
        first = opt.get_next_trial()
        with self.assertRaisesRegex(RuntimeError, "pending"):
            opt.get_next_trial()
        first.skip()
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            first.update(1.0)

        points = {first.parameters["x"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(3):
                trial = opt.get_next_trial()
                x = trial.parameters["x"]
                self.assertNotIn(x, points)
                points.add(x)
                trial.update((x - 5) ** 2)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            opt.get_next_trial()
        self.assertEqual(len(points), 4)

    def test_best_trial_and_reproducibility(self):
        a = BayesianOptimization({"x": (0, 1)}, random_state=11)
        b = BayesianOptimization({"x": (0, 1)}, random_state=11)
        self.assertTrue(np.array_equal(a.X_candidates, b.X_candidates))
        trial = a.get_next_trial()
        trial.update(0.0)
        self.assertIs(a.get_best_trial(), trial)
        with self.assertRaisesRegex(RuntimeError, "already"):
            trial.update(1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            Trial({"x": 0.5}).update(float("nan"))

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            BayesianOptimization({"x": (1, 1)})


if __name__ == "__main__":
    unittest.main()
