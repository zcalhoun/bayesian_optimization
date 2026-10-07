# Bayesian optimization classroom notebook

[Open the notebook](bayesian_optimization_demo.ipynb) to walk through Bayesian optimization on a two dimensional loss surface. The notebook creates a synthetic surface with two valleys, displays it with `imshow`, evaluates eight Sobol starting points, and then plots seven expected improvement (EI) decisions before evaluating them. Each iteration shows the true surface, the Gaussian process (GP) prediction, and the EI map. A final figure shows the observations and best loss over time.

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook bayesian_optimization_demo.ipynb
```

Run the cells from top to bottom. The notebook imports `BayesianOptimization` from `bopt.py`, so keep both files in this directory. You can change `SEED`, `N_INITIAL`, or `N_EI_STEPS` in the setup cell and rerun all cells.

The default run's figures and trace are saved in the notebook, so they are visible as soon as it opens.

## Teaching points

- The full surface is shown as a teaching reference. The optimizer only sees the objective values at requested points.
- A scrambled Sobol sequence spreads the initial observations around the domain.
- The GP supplies a predicted mean and uncertainty. For this visualization, its RBF length scale is fixed at 0.25 of each normalized parameter range.
- EI balances promising predictions and uncertainty, choosing the next point from a finite candidate pool.
- The default run improves the best observed loss from about -0.47 after initialization to about -1.71. The dense display grid has a minimum near -1.72; this grid is never used to choose trials.

The optimizer is sequential and assumes continuous parameters and a deterministic objective. It does not handle constraints or parallel evaluations. To check its focused tests, run `python -m unittest -v`.
