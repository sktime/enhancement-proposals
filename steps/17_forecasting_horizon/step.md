# Addressing `pandas` `freq` deprecation within ForecastingHorizon

Contributors: @khrapovs

## Introduction

Currently, `ForecastingHorizon` object allows conversion to relative/absolute version of itself (`.to_relative`/`.to_absolute` methods). These methods internally require frequency for conversion. This information is obtained from `cutoff` argument which is expected to be either `int`, `pd.Timestamp`, or `pd.Period`. Frequency attribute (`.freq`) of `pd.Timestamp` will be deprecated in `pandas` in the near future. Hence, we need to address this deprecation. This document outlines possible solutions to this deprecation problem.

Related issue: [#1750](https://github.com/alan-turing-institute/sktime/issues/1750)

## Solution options

### 1. Prohibit `pd.Timestamp` type for `cutoff` argument

Implemented in [PR 2694](https://github.com/alan-turing-institute/sktime/pull/2694).

Passing `pd.Timestamp`, for example, in `.to_absolute`
```python
fh.to_absolute(cutoff=pd.Timestamp("2022-01-01"))
```
raises `AssertionError`. Once is forced to pass `pd.Period` instead which has a perfectly valid `.freq` attribute. The consequence is that the return type of `fh.to_absolute(cutoff=pd.Period("2022-01-01"))` id `pd.PeriodIndex` which may end up incompatible with `pd.DatetimeIndex` of the time series index itself (`y.index`). Hence, we implement a new standalone function `convert_fh_to_datetime_index` that converts absolute `ForecastingHorizon` object to `pd.DatetimeIndex`, if that is required for compatibility with time series `y`.

### 2. Pass frequency to `ForecastingHorizon` as an optional argument

New argument in constructor:
```python
fh = ForecastingHorizon(values=[1], is_relative=True, freq="30T")
```
or a new argument in public methods that require `cutoff`, e.g.
```python
fh.to_absolute(cutoff=pd.Timestamp("2022-01-01"), freq="30T")
```

This new argument can be optional. In case it is not provided, an attempt will be made to extract frequency from provided `cutoff` argument. If neither frequency is provided explicitly, nor `cutoff` has a valid frequency attribute, then an error should be risen.
