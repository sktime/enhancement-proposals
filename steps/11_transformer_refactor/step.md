# refactor of transformers interface

Contributors: @fkiraly, 

## High-level summary 

### The Aim

The transformers interface has grown organically and is difficult to use, interface, and extend.

This proposal tries to alleviate the problem by removing mixins from transformers and using the new tag system to provide a unified and easy-to-use interface.

### The proposed solution

Our proposed solution follows principles used in the refactor of the forecasters:

* methods are separated into outer `fit`/`transform` and inner `_fit`/`_transform` etc
* the four mixins are removed, and replaced by tags
* the four transformer mixins are united under a single interface
* addition of `update` methods for transformers, to be usable with forecaster `update`

The unification introduces type inhomogeneity in the `transform` method - this is addressed by allowing both `Series` and `Panel` inputs and outputs, and dropping the `Z` argument in favour of `X` (keeping the `Z` as a downwards compatible alias).

## Refactor design: transformers

We proceed outlining the refactor interface.

### Conceptual model: types of transformers

Conceptually, we change our view of transformer types. Before the refactor, these weree:

* series-to-primitives
* panel-to-tabular
* series-to-series
* panel-to-panel
* series-to-panel

Where series-to-panel occurs only in two places currently and not formalised as a transformer type:
* sliding window forecasting (in `update_predict`)
* epoching, in `sktime-neuro`

We replace the old taxonomy with an input/output and rowwise yes/no taxonomy - based on what type an individual instance has, and whether `fit`/`transform` is independent per-instance or not:

| `transform-input` | `transform-output` | `rowwise` | old name |
|---|---|---|---|
| `Series` | `Primitives` | True | series-to-primitives |
| `Series` | `Primitives` | False | panel-to-tabular |
| `Series` | `Series` | True | series-to-series |
| `Series` | `Series` | False | panel-to-panel |
| `Series` | `Panel` | True | N/A, this is new |
| `Primitives` | `Primitives` | True | unfittable scikit-learn transformer |
| `Primitives` | `Primitives` | False | fittable scikit-learn transformer|

Implications on the signature will be discussed below, and we will repeat the table after a longer explanation.

### Fit/transform signature

We begin by outlining the expected fit/transform signature for each of the "old" transformer types.

#### series-to-primitives and panel-to-tabular

`fit(X: Panel or Series, y=None) -> self`

`transform(X: Panel or Series, y=None) -> res: pd.DataFrame`

where in `transform`, rows in `res` correspond to instances in `X`.
If `X` is a `Panel`, `transform` is applied per row; if `X` is a `Series` it is considered as a single-series-panel.
The exogeneous input `y` can be any label vector but is usually `None` (discussed later).

The distinction between series-to-primitives and panel-to-tabular lies not in the `transform` interface, but in assumptions
about the statistical relation between input and output:
* all pairs of input rows of `fit` and output rows of `transform` are assumed statistically independent for series-to-primitives - this is modelled as a boolean property tag `scitype:rowwise = True`.
* this assumption is not made for panel-to-tabular - this is modelled as a boolean property tag `scitype:rowwise = False`.

Series-to-primitives transformers should be considered vectorized in `fit`, with `Series` being the "native" input and `Panel` being
a vectorized

#### series-to-series and panel-to-panel

`fit(X: Panel or Series, y=None) -> self`

`transform(X: Panel or Series, y=None) -> res: Panel or Series`

where instances in `res` correspond to instances in `X`; and `X` in `fit` and `transform` must have the same number of instances and variables.
If `X` is a `Panel`, `transform` is applied per row; if `X` is a `Series` it is considered as a single-series-panel.
The exogeneous input `y` can be any label vector but is usually `None` (discussed later).

The distinction between series-to-series and panel-to-panel lies not in the `transform` interface, but in assumptions
about the statistical relation between input and output:
* all pairs of input rows of `fit` and output rows of `transform` are assumed statistically independent for series-to-series - this is modelled as a boolean property tag `scitype:rowwise = True`.
* this assumption is not made for panel-to-panel -  this is modelled as a boolean property tag `scitype:rowwise = False`.

#### series-to-panel

`fit(X: Series, y=None) -> self`

`transform(X: Series, y=None) -> res: Panel`

The exogeneous input `y` can be any label vector but is usually `None` (discussed later).

### Scitype tags

Which type a transformer has is determined by inspectable tags:

* `scitype:transform-input`, which can have values `Series`, or `Primitives`
* `scitype:transform-output`, which can have values `Series`, `Panel`, or `Primitives`
* `scitype:rowwise`, which is bool as described above.

Only the five combinations above are allowed (implemented), plus "primitives-to-primitives" which is the tag combination for scikit-learn type transformers.

To avoid confusion, the old terminology "series-to-series" etc is no longer used, instead we talk about input/output types.
The following table gives a translation between old and new scheme.

| `transform-input` | `transform-output` | `rowwise` | old name |
|---|---|---|---|
| `Series` | `Primitives` | True | series-to-primitives |
| `Series` | `Primitives` | False | panel-to-tabular |
| `Series` | `Series` | True | series-to-series |
| `Series` | `Series` | False | panel-to-panel |
| `Series` | `Panel` | True | N/A, this is new |
| `Primitives` | `Primitives` | True | unfittable scikit-learn transformer |
| `Primitives` | `Primitives` | False | fittable scikit-learn transformer|

The other five combinations are not used - four would involve `Panel` as input, which, with vectorization would assume a 4D data container; and one is "primitives-to-panel" for which there do not seem to be useful examples..

### Handling of `y`

Normally, `y` will be `None` for most concrete transformers, but the argument exists for compatibility and for examples of labelled transforms. We don't design this explicitly, due to a lack of complete view of different relevant cases.

However, to handle the implied taxonomy, we introduce an additional tag 

* `scitype:transform-y`, which holds the (scitype) of `y`, usually `None` but possibly `Series` or `Segmentation`

### Updating

Updating always means an update in *temporal* index direction, i.e., observation of more/later time points, not observations of more samples (in a panel).

All update methods (except of series-to-panel which admits only `Series`) have the signature `update(X: Panel or Series, y=None) -> self`.
The `X` in update must have the same variables, and same number of instances as the `X` seen in `fit`.

### Downwards compatibility: `Z`

To ensure downwards compatibility, the transformers that currently accept a `Z` argument will still accept `Z` as an alias for `X`.

### `fit`/`_fit` and input/output conversions

The transformers will use the refactored mtype and conversion interface from PR #1225.
This will be handled exactly as in the forecasters, with an `X_inner_type` and a fixed output mtype for a given scitype - `pd.DataFrame` for `Series` and `pd-multiindex` for `Panel`.

Beyond this, some additional features of the conversion interface are required:

* an override for the output type, settable as a dynamic tag. This is useful in pipeline constructs and for ensuring downwards compatibility.
* conversion when `Series` are passed instead of `Panel`. This will convert mtypes for `Series` to "adjacent" mtypes for `Panel` where applicable, i.e., `pd.DataFrame` to `df-list` or 2D numpy to 3D numpy.
* internal types can include mtypes for `Series` and `Panel` both, in which case this conversion may not be necessary.

### Ensuring downwards compatibility in the refactor

As an intermediate step in the refactor, the mixin classes will not be removed, instead replaced by classes that add tags (such as the output override) and mediate the `X`/`Z` aliasing.
