# scityping based input checks

Contributors: @fkiraly, @mloning, 

## The Problem

`sktime` is accumulating an increasing number of implicit scitypes for estimators, including time series prediction estimators for different tasks and transformer estimators of different types.

In the most recent versions, we also have simultaneous support for data containers of the same scitype but of different machine types - `np.array` vs nested `pd.DataFrame` in time series classification, for example.

Recent discussions seem to indicate that supporting multiple machine types may be inevitable when supporting advanced learning tasks, e.g., a single time series being representable by all of `np.array`, `pd.Series`, or `pd.DataFrame` with specific encoding conventions.

Version 0.5 has also introduced implicit scityping using the `typing` module, see the pre-amble of `transformers/base.py`. This is accompanied by type and format checking functionality in `utils/validation/series.py`. The interplay between declarative typing and implicit checking seems a bit bolted together and should be straightened out, also see discussion in [PR 420](https://github.com/alan-turing-institute/sktime/pull/420#issuecomment-716059124).

## The Aim

We suggest to collate scattered type declaration and type/format checking functionality in a coherent module, which also introduces an abstraction layer.

This module - or perhaps package? - should implement:

* declarative scityping in functions or class method arguments, using `typing` syntax, as in current `transformers/base.py`
* shorthand definitions of machine type vs scitype mapping, as in current preamble of `utils/validation/series.py`
* extensible single-argument (sci)type checks for format and adherence to implicit formatting conventions, e.g., whether a `pd.DataFrame` correctly encodes a multivariate time series (`MultivariateSeries` type) or a sample of time series (`Panel` type)
* extensible multi-argument type checks, e.g., checking same length of `X` and `y` in `fit`
* extensible conversion functionality between different machine types encoding the same scitype
* input/output conversion determined by explicit representation of an estimator's default internal machine type and comparison with input/output machine type

## Related Work

MLJ, the "Julia scikit-learn", has a satellite package with comparable functionality, [MLJLScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/stable/). It provides mostly an abstraction layer on top of an existing strong typing system for data types, but does not contain functionality for convention validation, multi-argument type checks, or conversion. 
Further, since the python language does have typing related patterns used therein, such as dispatch, it will not serve as a blueprint, but more of a generic source of inspiration.

## Design principles

We consider the following design principles key

* consistency with existing template interfaces in sktime
* no adverse impact on framework complexity - while some complexity in implementing the type checking module is expected, it should be separate from and not impact the core framework
* no adverse impact on extensibility - the burden on dev users extending the framework should not increase
* avoiding user frustration - natural user expectations on interface behaviour should be met; in particular, neither low-tech user nor dev user should be required to
* adherence with sklearn style design principles - unified interface (strategy pattern), modularity, sensible defaults, etc
* downwards compatibility - impact code written in earlier versions of the interface should be minimized


