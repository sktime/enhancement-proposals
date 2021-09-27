# probabilistic forecasting API

Contributors: @fkiraly, @sveameyer13

## High-level summary 

### The Aim

`sktime` has a predictions interval interface which is a bit clunky to use and suffers from proliferation of methods in the base class.
Further, we would like to introduce a distribution return interface for probabilistic forecasts, a highly requested feature.

See also https://github.com/alan-turing-institute/sktime/issues/984

### The proposed solution

Our proposed solution consists of the following components:

* splitting of the prediction interval return functionality from `predict` in a function `predict_interval`, while keeping downwards compatibility
* adding a function with distribution return type `predict_proba`
* adding a shorthand for predictive variance return, `predict_var`

## Refactor design: probabilistic forecasting

We proceed outlining the refactor target interface.

### Conceptual model: predict return types

The conceptual idea is to map different return types onto different variants of `predict`, denoted by an underscore suffix. If we want to predict/forecast a random variable $Y$ (at a single horizon), then:

* `predict` returns the predictive expectation $\mathbb{E}(Y)$
* `predict_var` returns the predictive variance $\mbox{Var}(Y)$
* `predict_interval` returns predictive quantiles $F_Y^{-1}(\alpha_i)$, given requested quantiles $\alpha_1,\dots, \alpha_k$ (following usual conventions on the pseudo-inverse of the cdf $F_Y$ of $Y$ rendering it well-defined)
* `predict_proba` returns the predictive law of the random variable $Y$, i.e., a full distribution object

By default, the predictive variances and distributions are point-wise (per-horizon, marginal), but the interface should also accommodates joint predictions, i.e., covariance matrix returns and joint predictive distributions.

NOTE: this convention allows the `predict` return to lie outside, say, 5/95% prediction quantiles from `predict_interval`, since the return of `predict` is the predictive mean, not the predictive median; the predictive mean can, in general, be the prediction quantile at any percentage between 0 and 100.

### new methods: signature specification

We specify the function signature of the new methods:

* `predict(self, fh=None, X=None) -> Series` is the target signature of `predict`. This differs from temporary signature to be deprecated (below).

* `predict_var(self, fh=None, X=None, cov=False) -> np.ndarray` returns: if `cov=False`, an vector of same length as `fh` with predictive marginal variances; if `cov=True`, a square matrix of size `len(fh)` with predictive covariance matrix

* `predict_interval(self, fh=None, X=None, alpha=Union[float, np.ndarray[float, dim=1]]) -> pd.DataFrame` returns a `pd.DataFrame` with rows having `fh` as index, and columns being a multi-index. First column multi-index level has are names of variables/columns of `y` (passed to `fit`); second column multi-index level has floats corresponding to requested quantiles (as a fraction). If `y` has no variable names (e.g., a `Series` without `name` attribute), the column name is `"quantiles"` instead. Entries are the predictive quantile for the horizon in the row, for the variable in the first column index at the quantile in the second column index. If `alpha` is a float, then the returned quantiles are `alpha`, `0.5` and `1-alpha` for all variables in `y`. If `alpha` is a vector, then the returned quantiles are exactly the elements of `alpha`, for all variables in `y`.

* `predict_interval` example: if `y` has column names `"foo"`, `"bar"`, and `alpha=0.05` the return has column indices `("foo", 0.05)`, `("foo", 0.5)`, `("foo", 0.95)`, `("bar", 0.05)`,`("bar", 0.5)`, and `("bar", 0.95)`.

* `predict_proba(self, fh=None, X=None, joint=False) -> tensorflow-probability.Distribution` returns a `tensorflow-probability` `Distribution` object. If `joint=False`, it is a vectorized distribution with one vector dimension, elements corresponding to elements of `fh`, in same sequence. If `joint=True`, it is a single joint distribution over a `len(fh)`-dimensional domain. Marginals of the `joint=True` return must be identical with the `joint=False` case.


### base class defaulting

If `predict_proba` is implemented, the other three `predict` functions default to calls in the following way:

* `predict` calls `predict_proba` and then the mean
* `predict_var` calls `predict_proba` and then the (co-)variance
* `predict_interval` calls `predict_proba` and then the quantile function

(of the distribution object returned by `predict_proba`)
 
### new tags

Each estimator may or may not have the capability to predict variances, intervals, etc. For this, the following boolean tags are introduced, with their obvious meaning:

* `capability:predict_var`
* `capability:predict_interval`
* `capability:predict_proba`

### public/private interface

Each new method has a private counterpart which is called internally, i.e., `_predict_var`, `_predict_interval`, `_predict_proba`. As in the generic `sktime` design, the public method contains "plumbing", while the private class is the extension locus and contains only method logic.

### Downwards compatibility: `predict`

To ensure downwards compatibility, the `return_pred_int` and `alpha` parameters will still be accepted by `predict` until the end of the next deprecation cycle.

If these are passed and `return_pred_int=True`, the base class directs the `predict` call to `predict_interval` where possible; Note: this is always possible once a specific forecaster is refactored. Whether a forecaster is already refactored will be checked for in an additional class method `_has_refactored_predict_interval()`;

Since the return type of prediction intervals also changes, a boolean tag `keep_old_return_type=True` will temporarily be passed to `predict`, allowing the user to decide whether he wants to use the old return type (also possible until the next deprecation cycle". 

### removal/refactor of unnecessary `BaseForecaster` methods

Currently, there are a number of unnecessary methods in the `BaseForecaster` which need to be refactored.

* `compute_pred_int`, seems to duplicate the current interface point in `predict`, used only by `theta`, move there
* `_compute_pred_int`, called by `compute_pred_int` only, move to `theta`
* `_get_pred_int`, used only by `_tbats` - move to `_tbats`
* `_compute_pred_err`, called by `compute_pred_err`, used for override in some descendants, move there or keep
