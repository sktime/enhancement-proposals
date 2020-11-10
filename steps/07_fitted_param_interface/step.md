# A Uniform Interface for Fitted Parameters
Contributors: @fkiraly, @mloning

## Motivation
Why do we need a unified interface for fitted parameters?

## Use cases
* Feature extraction when used via reduction to solve related learning tasks using fitted parameters as features,
* Fittable heuristic to set parameters for subsequent transformers/estimators (median distance heuristic for setting kernel parameters),
* Inspection/Interpretability of models,

## Other advantages
* Would simplify [`check_fit_idempotent`](https://github.com/scikit-learn/scikit-learn/blob/3a6c8c4b6ce17b84b01d47b66ead1797c04931bc/sklearn/utils/estimator_checks.py#L2925)

## Implementation details
* Add class attribute `_fitted_parameters` to all estimators, list/tuple of names of fitted parameters
* Add method `get_fitted_params()`, optionally also `set_fitted_params()`, to `BaseEstimator` with analogous behaviour to `get_params()` and `set_params()`, i.e.`get_fitted_params()` returns dictionary of fitted parameter names and values. 
