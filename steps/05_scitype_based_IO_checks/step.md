# scityping based input checks

Contributors: @fkiraly, @mloning, 

## The Problem

`sktime` is accumulating an increasing number of implicit scitypes for estimators, including time series prediction estimators for different tasks and transformer estimators of different types.

In the most recent versions, we also have simultaneous support for data containers of the same scitype but of different machine types - `np.array` vs nested `pd.DataFrame` in time series classification, for example.

Recent discussions seem to indicate that supporting multiple machine types may be inevitable when supporting advanced learning tasks, e.g., a single time series being representable by all of `np.array`, `pd.Series`, or `pd.DataFrame` with specific encoding conventions; or, a sample of time series being representable by all of `np.array`, nested `pd.DataFrame`, or `pd.DataFrame` in long format.

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

## User journey

We outline intended usage for two use cases:
* the standard user, making use of estimators shipping with `sktime`
* the extender user, writing new `sktime` compatible estimators in accordance with design templates

### User journey: standard user

The standard user's journey involves defining an estimator and invoking it on data.
The specific sequence will depend on the scitype of the estimator.
In the example of time series classification with a `sklearn` compatible design, it may look as follows:

```python
myest = SktimeEstimator(param1=1, param2=42)
myest.fit(X, y)
y_pred = myest.predict(X_pred)
```

where `X`, `X_pred` are samples of sequences, and `y`, `y_pred` are label vectors.

In the "scitpyed" user journey, the user can pass the sample of sequences `X` in different formats, say `np.array`, nested `pd.DataFrame`, or `pd.DataFrame` in long format, to `fit`, and `predict` will return the type accordingly - thus minimizing user frustration by removing the necessity to perform input type conversion to one of multiple equally obvious input formats.

In addition to "same-type-in-as-out" (standard behaviour), the standard user can also specify output types that deviate from input types by parameters of the estimator's constructor - in the example above, by invoking `myest.predict(X_pred, out_type = ClassVec_as_pdSeriesCat)`.

### User journey: extender user

The extender user journey involves writing core logic for the estimator.
In our design, we try to minimize the overhead in terms of input/output checks and conversions on the extender.

The optimal solution here is perhaps the use of class decorators, taking care of input/output conversion.
That is, the extender user would:
* inherit from a template base class
* implement mandatory abstracts such as `fit` and `predict` in time series classification
* in said implementations, type hint which machine types the implementation supports
* decorate the class with a scitype specific decorator which takes care of conversions and checks

In particular, there is no burden of doing the right type checks in the right place on the extender user.

For example, for time series classifiers, a custom class could look as follows:

```python

@scitype_TimeSeriesClassifier
class MyTimeSeriesClassifier(BaseTSC):

    def __init__(etc):
        #in sklearn convention

    def fit(X: TSS_as_NestedDF, y: ClassVec_as_nparray):
        # implement logic here

    def predict(X: TSS_as_NestedDF) -> ClassVec_as_nparray:
        # implement logic here
        return y_pred
```

In the above:
* the class decorator `scitype_TimeSeriesClassifier` takes care of input/output type conversion; it can simultaneously be read by the extender user as an explicit scitype declaration.
* the type hints are used to specify what input/output types are passed to or returned by the implementation.

Types come in two forms:
* pure scitypes - for example, `TSS` for "sample of time series" or `ClassVec` for "vector of categoricals"
* machine-type/scitype combination hints - for example, `TSS_as_NestedDF` is the type of `TSS` encoded as nested `pd.DataFrame` according to a specified convention (nested format); similarly `TSS_as_LongDF` could be `TSS` encoded as `pd.DataFrame` according to a sepcified convention (long format)

The extender user can specify machine-type/scitype combination hints, in which case this means that the implementation assumes these input/output types. Alternatively, they can also supply the pure scitype as a hint, which means the implementation natively supports all input/output formats (where the hint is given).





