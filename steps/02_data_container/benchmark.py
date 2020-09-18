#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import awkward1 as ak
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForest
from sktime.utils._testing.series_as_features import \
    make_classification_problem
from sktime.utils.data_container import nested_to_3d_numpy

from tsf_awkward import TimeSeriesForestAwkward


def _make_ak_array(X):
    Xc = X.copy()
    n_instances, n_variables = Xc.shape
    n_timepoints = Xc.iloc[0, 0].shape[0]

    # convert data into nested list
    Xl = []
    for i in range(n_instances):
        variables = []
        for v in range(n_variables):
            series = Xc.iloc[i, v]
            assert series.shape == (n_timepoints,)
            variables.append(series)
        Xl.append(variables)
    assert len(Xl) == n_instances

    # convert into awkward array
    instances = []
    for i in range(n_instances):
        instance = []
        for v in range(n_variables):
            # get time series
            series = Xl[i][v]

            # separate values/index
            values = series.to_numpy()
            times = series.index.to_numpy()

            variable = [{"time": time, "value": value}
                        for time, value in zip(times, values)]
            instance.append(variable)
        instances.append(instance)
    return ak.Array(instances)


def _fit_predict(estimator, X_train, y_train, X_test):
    return estimator.fit(X_train, y_train).predict_proba(X_test)


X, y = make_classification_problem(n_instances=100, n_timepoints=100)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_ak, X_test_ak = _make_ak_array(X_train), _make_ak_array(X_test)

expected = _fit_predict(TimeSeriesForest(random_state=1),
                        X_train, y_train, X_test)


def ak_3d_arr(X):
    return ak.Array(nested_to_3d_numpy(X))


def ak_record_arr(X):
    return _make_ak_array(X)


def np_arr(X):
    return nested_to_3d_numpy(X)


def _mean(X, axis=-1):
    return np.mean(X, axis=axis)


def _slice(X):
    return X[1:4, :, 1:4]


def test_ak_3d_mean(benchmark):
    X_ak = ak_3d_arr(X)
    benchmark(_mean, X_ak)


def test_ak_3d_slice(benchmark):
    X_ak = ak_3d_arr(X)
    benchmark(_slice, X_ak)


def test_ak_record_mean(benchmark):
    X_ak = ak_record_arr(X)
    benchmark(_mean, X_ak)


def test_ak_record_slice(benchmark):
    X_ak = ak_record_arr(X)
    benchmark(_slice, X_ak)


def test_np_mean(benchmark):
    X_np = np_arr(X)
    benchmark(_mean, X_np)


def test_np_slice(benchmark):
    X_np = np_arr(X)
    benchmark(_slice, X_np)


# def test_tsf(benchmark):
#     estimator = TimeSeriesForest(random_state=1)
#     actual = benchmark(_fit_predict, estimator, X_train, y_train, X_test)
#     np.testing.assert_array_equal(actual, expected)
#
#
# def test_tsf_ak_rec(benchmark):
#     estimator = TimeSeriesForestAwkward(random_state=1)
#     actual = benchmark(_fit_predict, estimator, X_train_ak, y_train, X_test_ak)
#     np.testing.assert_array_equal(actual, expected)
