#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
from sktime.utils._testing.series_as_features import \
    make_classification_problem
from sktime.utils.data_container import tabularize

from benchmarks.utils import ak_3d_arr
from benchmarks.utils import ak_record_arr
from benchmarks.utils import np_arr


def _mean(X, axis=-1):
    return np.mean(X, axis=axis)


def _tabularize_mean(X, axis=-1):
    return tabularize(X).to_numpy().mean(axis=axis)


def _nested_mean(X):
    return np.asarray([X.iloc[i, 0].mean() for i in range(X.shape[0])])


X, y = make_classification_problem(n_instances=100, n_columns=1, n_timepoints=100)

expected = _mean(np_arr(X))


def test_ak_3d_mean(benchmark):
    x = ak_3d_arr(X)
    actual = benchmark(_mean, x)
    np.testing.assert_array_almost_equal(expected, np.array(actual))


def test_ak_record_mean(benchmark):
    x = ak_record_arr(X)
    actual = benchmark(_mean, x)
    np.testing.assert_array_almost_equal(expected, actual["value"])


def test_np_mean(benchmark):
    x = np_arr(X)
    actual = benchmark(_mean, x)
    np.testing.assert_array_equal(expected, actual)


def test_nested_mean(benchmark):
    actual = benchmark(_nested_mean, X)
    np.testing.assert_array_equal(expected, actual.reshape(-1, 1))


def test_tabularize_mean(benchmark):
    actual = benchmark(_tabularize_mean, X)
    np.testing.assert_array_equal(expected, actual.reshape(-1, 1))
