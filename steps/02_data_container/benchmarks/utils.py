#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import awkward1 as ak
from sktime.utils.data_container import nested_to_3d_numpy


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


def ak_3d_arr(X):
    return ak.Array(nested_to_3d_numpy(X))


def ak_record_arr(X):
    return _make_ak_array(X)


def np_arr(X):
    return nested_to_3d_numpy(X)
