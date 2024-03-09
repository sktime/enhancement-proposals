# probabilistic forecasting API, part 2 - distribution forecasting

Contributors: @fkiraly

## High-level summary 

### The Aim

`sktime` now has streamlined functionality for interval, quantile, and variance forecasts. However, distribution forecasts via `predict_proba` remain hard to use, and are not compatible with hierarchical features and broadcasting that is mostly `pandas` driven.

The distribution forecast interface should be reworked, with requirements detailed in the next section.

References:

* STEP 13 on probabilistic forecasting: https://github.com/sktime/enhancement-proposals/blob/main/steps/13_proba_forecasting/step.md
* original issue on probabilistic forecasting https://github.com/alan-turing-institute/sktime/issues/984
* issue on distribution repreesentations: https://github.com/sktime/sktime/issues/1746


### requirements

* `pandas` based interface for batches of distributions: row/column index, hierarchical index, sub-setting - needed for compatibility with hierarchical features, metrics interface
* backend agnosticity in-principle - current `tensorflow-probability` dependency is heavy, especially for users who do not want to use deep learning forecasters
* design should allow easy extension of loss functions to the distribution forecasting case
* design which is extensible to adding "exotic" distribution defining functions such as energy statistics, integrated cdf as required by CRPS and common losses

### The proposed solution

Our proposed solution consists of the following components:

* `BaseDistribution` inheriting from `BaseObject`, representing collections of distributions with a batch shape, and `pandas` row and column index
* a `tensorflow-probability` backend adapter inheriting from `BaseDistribution`
* loss functions extending `BaseMetric` and a sub-base class `BaseProbaMetric`, interfacing the `BaseDistribution` interface
* exotic distribution defining functions are added to `BaseDistribution`, which can also hold default approximation methods; overrides can be added in descendants

## Design: distribution forecasting

We proceed outlining the refactor target interface.

### Conceptual model: `pandas`-indexed distributions

Conceptually, we want to model "distribution objects" that are probability distributions, taking values in `pandas.DataFrame`-s, possibly with hierarchical row index.

For this, we need to keep track of the following information:

* a batch/instance structure of the distribution, e.g., 2D matrix shape
* distribution defining information, i.e., parameters of the distribution
* row index and column index of the distribution, both `pd.Index`.

Row and column index shape need to agree with the batch/instance shape of the distribution.

Typically but not nececssarily, different rows will be independent, but different columns in the same row will not be independent.


### `BaseDistribution` interface

* `BaseDistribution`-s are constructed with `column` and `index` (like `pd.DataFrame`-s), and any number of named parameters
* named parameters can be scalar, or array-likes of dimension 1 or 2 broadcastable to the shape defined by `column` and `index`
* `BaseDistribution`-s inherit from `BaseObject`, and therefore possess `get_params`, `set_params`, etc.
* `BaseDistribution`-s possess distribution defining functions `pdf`, `cdf`, etc, that `tensorflow-probability` distributions also possess. However, by default, these return `pd.DataFrame` with index naturally determined by `column`, `index` of the parent distribution.
* `BaseDistribution` can be sub-set by `loc` and `iloc` indexers like `pandas.DataFrame`; the result being another `BaseDistribution` of the same class

### `predict_proba`: signature specification

The signature of `predict_proba` changes to:

`predict_proba(self, fh=None, X=None, joint=False) -> Base Distribution`

If `joint=False`, it is a vectorized distribution with one vector dimension, elements corresponding to elements of `fh`, in same sequence. If `joint=True`, it is a single joint distribution over a `len(fh)`-dimensional domain. Marginals of the `joint=True` return must be identical with the `joint=False` case.

### Distribution metrics and losses

Distribution metrics will closely follow the interface of forecasting metrics, with `evaluate`/`_evaluate` and the same parameter structure (e.g., `multioutput` and `multilevel`).

The signature will be

`evaluate(self, y_true: BaseDistribution, y_pred, **kwargs)`,

where `y_true` is an `sktime` compatible time series type, and `y_pred` is a `BaseDistribution` with same `index` and `columns` as `y_true`, if the latter is `pandas` based (otherwise of same shape).

## Change and deprecation

### Change of `predict_proba` signature

During a deprecation period, `predict_proba` will have an additional boolean argument `legacy_interface`. If `True`, the function will behave as before. The default will change from `True` to `False` over one minor cycle, and the parameter will be removed after another cycle.


## Implementation phases

### Phase 1 - distributions and losses

Phase 1 is a proof-of-concept, implementing:

* 2 or 3 distributions classes, interfacing most frequently used `tensorflow-probability` distributions
* basic distribution methods: `pdf`, `cdf`, `logpdf`
* a test class with the `TestAll...` pattern, for distributions
* a base class for probabilistic losses
* the logarthmic loss for distributional forecasts, inheriting from this base class
* new tests in the metric test suite, for probabilistic losses

### Phase 2 - exotic methods and CRPS

Phase 2 aims at implementing the CRPS, using methods of the distribution class:

* energy function default, by approximation or quadrature, in `BaseDistribution`
* energy functions for 1 or some of the distribution classes
* the CRPS as a metric class, calling the energy functions

### Phase 3 - restart `skpro`

In phase 3, `BaseDistribution` and probabilistic losses are moved to `skpro`.

`skpro` is rebooted with a dependency on `skbase`.

Requires:

* professional package structure, CI/CD - `sktime` or `skbase` can be used as template
* live refactor, with experimental branch of `sktime` depending on `skpro`
* dependency change over multiple release cycles, with deprecation/change management

(the above is similar to refactoring `BaseObject` into `skbase`)

### Phase 4 - extend distributions

* more metrics/losses, e.g., squared integral loss; survival loss
* more distributions
* more distribution methods, e.g., integrated cdf, integrated survival function
* optional: additional distribution backends
