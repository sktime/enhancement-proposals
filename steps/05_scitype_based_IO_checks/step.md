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
* no adverse impact on extensibility - the burden on dev users extending the framework should not increase; optimally, it is lightened by removing boilerplate code, e.g., input checks
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
* the class decorator `scitype_TimeSeriesClassifier` takes care of input/output type conversion; it can simultaneously be read by the extender user as an explicit scitype declaration. The naming convention here is `scitype_[name of estimator scitype]`.
* the type hints are used to specify what input/output types are passed to or returned by the implementation.

Types come in two forms:
* pure scitypes - for example, `TSS` for "sample of time series" or `ClassVec` for "vector of categoricals"
* machine-type/scitype combination hints - for example, `TSS_as_NestedDF` is the type of `TSS` encoded as nested `pd.DataFrame` according to a specified convention (nested format); similarly `TSS_as_LongDF` could be `TSS` encoded as `pd.DataFrame` according to a sepcified convention (long format). The naming convention here is 
`[name of pure scitype]_as_[name of implementation case]` - note that there can be multiple implementation conventions with the same machine type, e.g., two conventions of machine type `pd.DataFrame` in the time series classification case (nested format and long format).

The extender user can specify machine-type/scitype combination hints, in which case this means that the implementation assumes these input/output types. Alternatively, they can also supply the pure scitype as a hint, which means the implementation natively supports all input/output formats (where the hint is given).

If no type hints are given, then documented conventions or the base class may impose some default type assumptions here (to be determined).

## Implementation design

Implementation of the STEP is divided in two parts:
* generic (sci)type checking and conversion functionality for single typed arguments
* scitype class decorators, including multi-argument input checks and compatible input-output conversions

The scitype decorators make use of the generic type checking and conversion functionality, thus depend on it as a module.

### Generic type checking and conversion

The generic type checking module contains a list of argument scitypes, as constants.
It lists:
* pure scitypes, e.g., `TSS` for "sample of time series"
* machine-type/scitype combinations, e.g., `TSS_as_NestedDF` for "sample of time series, encoded as nested `pd.DataFrame`"; for abbreviation we call these "mascitypes"
* for each mscitype, which scitype it implements; e.g., the information that `TSS_as_NestedDF` is an encoding convention for `TSS`
* in documentation: encoding convention of machine-type/scitype combinations, and meaning of pure scitypes

The module implements the following features:
* `puretypeof(mascitype: Type) -> Type` produces the pure scitype for a mascitype `mascitype`.
* `checkmascitype(obj, mascitype: Type) -> Boolean` checks whether `obj` is of mascitype `mascitype`
* `infermtype(obj, puretype: Type) -> Type` infers the mascitype combination type of `obj` which is known/assumed to be of pure type `puretype`. E.g., if `obj` is a `pd.DataFrame` in nested format, `infermtype(obj, TSS)` would return `TSS_as_NestedDF`
* `convert(obj, from_mascitype: Type, to_mascitype: Type) -> to_mascitype` converts `obj` from mascitype `from_mascitype` to mascitype `to_mascitype`. It is assumed that `obj` has mascitype `from_mascitype`.

The above likely have to make use of dispatch mechanisms, on mascitype arguments. E.g., `convert` should dispatch on `from_mascitype` and `to_mascitype`, for example to a function `convert_from_TSS_as_LongDF_to_TSS_as_NestedDF`. The dispatch mechanism is to be determined.

### Scitype class decorators

Each scitype has its own scitype class decorator, using the naming convention `scitype_[name of estimator scitype]`.

The scitype class decorator decorates a rump extension class, as detailed in the user journey of the "extender user".
It implements:
* multi-argument type checks and input/output conversions
* any other "under the hood" functionality specific to the scitype

We describe in detail only functionality for the type checks and conversions, as other functionality will be scitype specific.
Examples for other "under the hood" functionality may be:
* internal dispatch functionality for core methods, e.g., choosing one or the other variant of a private `predict`
* complex data scientific wrapper functionality, e.g., `update_predict` in forecasters

The multi-argument type checks and conversions are implemented as follows.

The scitype class decorator has consant class variables `default_mascitype_[methodname]_[argumentname]` whose values are the default mascitype of argument `[argumentname]` in method `[methodname]`, for all methods and arguments that have mascitypes.

When the class decorator wraps an extension class, it does the following:

1. for every method 

### Optional in the generic module: type conversion class decorator

This is optional (may simplify things?)

The generic type checking module also contains a generic class decorator factory that can be used to create class decorators by inheritance, with the functionality to use type hints to to convert mascitypes of inputs and outputs.

The factory creates a decorator from a template with type hints, which is assumed to have the same method/argument set as any class to be decorated later on.

The decorator performs type conversions from input mascitypes to the template mascitypes for inputs, and from decorated class mascitypes to the template mascitypes for outputs. 

Template defaults can be overridden using class variables `default_mascitype_[methodname]_[argumentname]` as above.
