#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
from sktime.utils._testing.series_as_features import \
    make_classification_problem

from benchmarks.benchmark import ak_3d_arr
from benchmarks.benchmark import ak_record_arr
from benchmarks.benchmark import np_arr


def _slice(X):
    return X[10:20, 5:15, 50:60]


X, y = make_classification_problem(n_instances=100,
                                   n_timepoints=100,
                                   n_columns=20)

expected = _slice(np_arr(X))


def test_ak_3d_slice(benchmark):
    x = ak_3d_arr(X)
    actual = benchmark(_slice, x)
    np.testing.assert_array_equal(actual, expected)


def test_ak_record_slice(benchmark):
    x = ak_record_arr(X)
    actual = benchmark(_slice, x)
    np.testing.assert_array_equal(actual["value"], expected)


def test_np_slice(benchmark):
    x = np_arr(X)
    actual = benchmark(_slice, x)
    np.testing.assert_array_equal(actual, expected)
