# Addressing `pandas` `freq` deprecation within `ForecastingHorizon`

Contributors: @khrapovs, @fkiraly

## Introduction

`ForecastingHorizon` has an internal concept of frequency, mirroring the `freq` attribute of `pandas` indices and time stamps.

This is inferred depending on construction case. `ForecastingHorizon` can be constructed with:
* a time stamp absolute or relative index set. Then `freq` is indirectly inferred from this index.
* an integer-like index set. If the time series index (of the forecasting time series `y`) is time-like, then `freq` is inferred from the time series index. If the time series index is integer-like, there is no `freq`.

The `freq` is required for conversions of `ForecastingHorizon` to a relative/absolute version of itself (`.to_relative`/`.to_absolute` methods), anchored at a cutoff. 
In the conversion methods, the `freq` information is currently obtained from `cutoff` argument, which in all uses is equal to the `cutoff` of the forecaster, which is expected to be either `int`, `pd.Timestamp`, or `pd.Period`.

However, the frequency attribute (`.freq`) of `pd.Timestamp` will be deprecated in `pandas` in the near future.
In that situation, `freq` can no longer be inferred from the `cutoff`.
The information is, however, still available in the direct source, i.e., the forecasting time series `y` or the `ForecastingHorizon` constructor argument.

The problem is: we need to address the deprecation. This document outlines possible solutions to this deprecation problem.

Related issue: [#1750](https://github.com/alan-turing-institute/sktime/issues/1750)


## Relevant use cases

We describe the most important internal use case examples to cover.

### iterable of `int` passed to `fit`/`predict`, converted to `ForecastingHorizon`

Users can pass `int` iterables of `int` such as `[1, 2, 3, 5]`, to `fit` or `predict` of a forecaster, as `fh` argument.
`fh` can be passed early, in `fit`, or late, in `predict` (depending on forecaster).

Internally, iterable `fh` are converted to `ForecastingHorizon`, before reaching `_fit` or `_predict`.
The `fh` seen in `_fit` and `_predict` are guaranteed to be `None` or `ForecastingHorizon`


### iterable of `timestamp` passed to `fit`/`predict`, converted to `ForecastingHorizon`

Users can pass iterables of timestamps, to `fit` or `predict` of a forecaster, as `fh` argument.
`fh` can be passed early, in `fit`, or late, in `predict` (depending on forecaster).

Internally, iterable `fh` are converted to `ForecastingHorizon`, before reaching `_fit` or `_predict`.
The `fh` seen in `_fit` and `_predict` are guaranteed to be `None` or `ForecastingHorizon`

### `ForecastingHorizon` passed directly to `fit`/`predict`

Users can also create a `ForecastingHorizon` with iterables, including indices.

In this case, the `freq` may not be known at construction, if it needs to be inferred from `y`.


### current situation and current solution

The `freq` concept of the `ForecastingHorizon` is currently:

* not stored in `ForecastingHorizon`
* inferred in its methods indirectly, via the forecaster's `cutoff` passed to methods. In all uses, this is is equal to the `cutoff` of the forecaster, which is the last time index of `y` seen in `fit`. This currently carries the `freq` for all cases of time-like indexing, but will lose it after deprecation, since single time indices will no longer carry `freq`.

### post-deprecation situation

Generally, in the post-deprecation situation, `freq` can be inferred from:

* `y.index` directly in `fit`, or via `self._y.index`, as remembered, in `predict`
* the constructor argument of `fh`, if it is a `pandas` index (and not a different form of iterable)
* the constructor argument of `fh`, via `pandas.infer_freq`


## Solution options

### 1. Prohibit `pd.Timestamp` type for `cutoff` argument

Implemented in [PR 2694](https://github.com/alan-turing-institute/sktime/pull/2694).

Passing `pd.Timestamp`, for example, in `.to_absolute`
```python
fh.to_absolute(cutoff=pd.Timestamp("2022-01-01"))
```
raises `AssertionError`. Once is forced to pass `pd.Period` instead which has a perfectly valid `.freq` attribute. The consequence is that the return type of `fh.to_absolute(cutoff=pd.Period("2022-01-01"))` id `pd.PeriodIndex` which may end up incompatible with `pd.DatetimeIndex` of the time series index itself (`y.index`). Hence, we implement a new standalone function `convert_fh_to_datetime_index` that converts absolute `ForecastingHorizon` object to `pd.DatetimeIndex`, if that is required for compatibility with time series `y`.


### 2. Pass frequency to `ForecastingHorizon` as an optional argument

This type of solution ensures `freq` is passed to the `ForecastingHorizon`:

### 2a. earlies possible pass, in constructor

New argument in constructor:
```python
fh = ForecastingHorizon(values=[1], is_relative=True, freq="30T")
```

Advantages:
* simpler solution, passed only once, "minimum information"

Disadvantages:
* `freq` needs to be as soon as the `ForecastingHorizon` is created, this may be too early.
  Example: `ForecastingHorizon` is integer, and `freq` first appears in `y.index`, in a forecaster.

### 2b. latest possible pass, in methods

new argument in all public methods that require `cutoff`, e.g.
```python
fh.to_absolute(cutoff=pd.Timestamp("2022-01-01"), freq="30T")
```

This new argument can be optional. In case it is not provided, an attempt will be made to extract frequency from provided `cutoff` argument. If neither frequency is provided explicitly, nor `cutoff` has a valid frequency attribute, then an error should be risen.

Advantages:
* remains flexible, "latest possible"

Disadvantages:
* `freq` has to be passed multiple times
* handling different `freq` passed at multiple instances seems like a logic and case distinction nightmare


### 2a'. 2a, but with option to set `freq` later

2a has a key disadvantage, namely that `freq` may not always be known at construction.

Therefore we can supplement 2a by an option to set it later.

A concrete design could be

```python
def _check_freq(freq):
    if self.freq is None:
        self.freq = freq
    if self.freq is not None and freq is not None and freq != self.freq:
        raise ValueError("all passed freq must be equal")
    return self.freq

def some_method(foo, bar, freq=None):
    freq = self._check_freq(freq)
```

or similar.

### accepting more `freq` inferrable formats

This can be combined with any of the type 2 options.

To make the logic robust w.r.t. future deprecations or changes how `freq` is handled in `pandas`,
there should be a `get_freq` function which can extract `freq` from a large range of objects.

Preferably:

* `pandas.Series` and `pandas.DataFrame`, obtaining it from `index.freq`
* time-like `pandas.Index` objects
* `freq` strings as used in `pandas`

The function `get_freq` can then be invoked at the start of any method with a `freq` input to convert different possible input options to a uniform `freq` representation.