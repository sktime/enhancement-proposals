#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForest
from sktime.utils._testing.series_as_features import \
    make_classification_problem

from .tsf import TimeSeriesForest_3d_np
from .tsf import TimeSeriesForest_ak_3d
from .tsf import TimeSeriesForest_ak_record
from .utils import ak_3d_arr
from .utils import ak_record_arr
from .utils import np_3d_arr


def _fit_predict(estimator, X_train, y_train, X_test):
    return estimator.fit(X_train, y_train).predict_proba(X_test)


PARAMS = {"n_estimators": 100, "random_state": 1}
X, y = make_classification_problem(n_instances=100, n_timepoints=200)
X_train, X_test, y_train, y_test = train_test_split(X, y)

expected = _fit_predict(TimeSeriesForest(**PARAMS),
                        X_train, y_train, X_test)


def test_tsf_3_np(benchmark):
    X_train_np, X_test_np = np_3d_arr(X_train), np_3d_arr(X_test)
    estimator = TimeSeriesForest_3d_np(**PARAMS)
    actual = benchmark(_fit_predict, estimator, X_train_np, y_train, X_test_np)
    np.testing.assert_array_equal(actual, expected)


def test_tsf_tabularize(benchmark):
    estimator = TimeSeriesForest(**PARAMS)
    actual = benchmark(_fit_predict, estimator, X_train, y_train, X_test)
    np.testing.assert_array_equal(actual, expected)


def test_tsf_ak_record(benchmark):
    X_train_ak, X_test_ak = ak_record_arr(X_train), ak_record_arr(X_test)
    estimator = TimeSeriesForest_ak_record(**PARAMS)
    actual = benchmark(_fit_predict, estimator, X_train_ak, y_train, X_test_ak)
    np.testing.assert_array_equal(actual, expected)


def test_tsf_ak_3d(benchmark):
    X_train_ak, X_test_ak = ak_3d_arr(X_train), ak_3d_arr(X_test)
    estimator = TimeSeriesForest_ak_3d(**PARAMS)
    actual = benchmark(_fit_predict, estimator, X_train_ak, y_train, X_test_ak)
    np.testing.assert_array_equal(actual, expected)
