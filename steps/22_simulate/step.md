# Simulation base class and interface for forecasters

Contributors: fkiraly, ltsaprounis

## Contents

[TOC]

## Overview

### Problem Statement

There are a number of "simulation" elements in `sktime`:

* simulating time series from fitted models, especially `statsmodels` models. Example: simulating from ARIMA with fitted parameters.
* simulating/sampling series as input to unit tests and contract tests, such as `_make_series` and `_make_hierarchical`
* simulating series for the purpose of benchmarking or scientific studies of estimator accuracy, e.g., in the new annotation module, incl `hmmlearn` interfaces

It seems natural to try bring these under the same interface:

* a unified interface is useful for cases of testing, reuse
* a unified interface allows later development of composition patterns, e.g., overlaying two simulators such as "trend plus residual"
* a unified interface allows use of simulators inside estimators, e.g., for sampling based estimators that are common in the annotation space, or Bayesian models

### status quo

Currently, most simulators exist in loose functions without a unified interface.

@ltsaprounis has proposed a simulation interface in https://github.com/sktime/sktime/pull/3462, although this covers only simulation from forecasters, and merges parameters of simulation with those of the forecaster.

### Requirements

* unified, composable interface for simulating time series - a "simulator" as an object
* should give rise to a natural interface to simulate from forecasters, and separate parameters of simulation from those of forecaster
* general compatibility with `sktime` and `sklearn` design patterns, e.g., `BaseObject` interface expectations

### The proposed solution

The proposed solution introduces:

* a `BaseObject` descendant class `BaseSimulator` for time series simulators, with a public/private `simulate`/`_simulate` pair
* a `simulate` interface for `BaseForecaster` which uses the `BaseSimulator` interface for its return

## Proposed solution

### User journey design

We outline user journey designs for:

* simulating one time series
* simulating a batch/panel of time series
* using a forecaster with a `simulate` method 

#### User journey design: simulating one time series

Example: current `_make_series` as a simulator class:

```python
# 1. create a simulator object
my_sim = MakeSeries(
    n_columns=1,
    all_positive=True,
)

# 2a. get a time series from the simulator
# index arg indicates index to simulate for
y = my_sim.simulate(at=50)
# int is interpreted as RangeIndex(50)

# 2b. same, with pandas.Index
y = my_sim.simulate(at=pd.RangeIndex(50))

# 2c. alternative - call is the same as "simulate"
y = my_sim(at=pd.RangeIndex(50))

# 2d. alternative - index determined by params
my_sim = MakeSeries(
    n_timepoints=50,
    n_columns=1,
    all_positive=True,
    index_type="range",
)

y = my_sim.simulate()
```
All the above result in the same `y` as current `y=_make_series(n_timepoints=50, index_type="range", all_positive=True)`

A tag may indicate whether a simulator supports `at` with a flexible index to simulate at.

#### User journey design: simulating multiple samples

Each simulator can be invoked to simulate multiple series at the same time. In this case, one level is added, e.g., simulators producing `Series` produce `Panel` if used with the `n_instances` argument:

```python
# 1. create a simulator object
my_sim = MakeSeries(
    n_timepoints=50,
    n_columns=1,
    all_positive=True,
    index_type="range",
)

# 2. simulate 10 independent instances
y_panel = my_sim.simulate(n_instance=10)
```

`y_panel` is now a collection of time series, of `Panel` scitype, with 10 instances, and an extra instance index (integer with 0 .. 9).

Simulation of multiple instances is i.i.d., from the same simulation process.

#### User journey design: simulating from a forecaster

Simulating from a forecaster works via the `simulate` method:

```python
# 1. create a forecaster
my_arima = ARIMA()

# 2. fit the forecaster
fh = ForecastingHorizon([1, 2, 3], is_relative=True)
my_arima.fit(y, fh=fh)

# 3a. simulate from the fitted model
y_sim = my_arima.simulate(at=fh)

# 3b. simulate multiple instance
y_sim_panel = my_arima.simulate(
    at=fh,
    n_instance=10,
)
```

When using exogeneous data, it should be passed to both `fit` and `simulate` (in the common case of "requires same index"):

```python
# 1. create a forecaster
my_arima = ARIMA()

# 2. fit the forecaster
my_arima.fit(y_train, fh=[1, 2, 3], X=X_train)

# 3. simulate from the fitted model
y_sample = my_arima.simulate(
    at=fh,
    X=X_sim,
    n_instances=10,
)
```

Simulators from forecasters may require extra parameters to be passed, required only for the simulator:

```python
# 1. create a forecaster
my_arima = ARIMA()

# 2. fit the forecaster
my_arima.fit(y, fh=[1, 2, 3])

# 3. simulate from the fitted model
# special simulation parameters are passed as kwargs
y_sample = my_arima.simulate(
    at=fh,
    n_instances=10,
    simulator_param=param_value,
)
```

If `simulate` is called without args, a simulator class is returned:

```python
# 1. create a forecaster
my_arima = ARIMA()

# 2. fit the forecaster
my_arima.fit(y, fh=[1, 2, 3])

# 3. get simulator class
y_simulator = my_arima.simulate(simulator_param=param_value)

# 4. simulate from class
y_sample = y_simulator(n_instances=10)
```


### Code design: simulator base class




### Code design: simulating from a fitted forecaster



### Alternative designs considered
