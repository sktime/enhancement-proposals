#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import math

import awkward1 as ak
import numpy as np
from sklearn.base import clone
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state
from sktime.classification.base import BaseClassifier


class TimeSeriesForestAwkward(ForestClassifier, BaseClassifier):

    def __init__(self,
                 random_state=None,
                 min_interval=3,
                 n_estimators=200
                 ):
        super(TimeSeriesForestAwkward, self).__init__(
            base_estimator=DecisionTreeClassifier(criterion="entropy"),
            n_estimators=n_estimators)

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.classifiers = []
        self.intervals = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y) using random
        intervals and summary features
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. RISE has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        # X, y = check_X_y(X, y, enforce_univariate=True)
        # X = tabularize(X, return_array=True)
        assert isinstance(X, ak.highlevel.Array)
        n_instances, self.series_length = X.shape[0], X[0, 0].shape[0]

        rng = check_random_state(self.random_state)

        self.n_classes = np.unique(y).shape[0]

        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.n_intervals = int(math.sqrt(self.series_length))
        if self.n_intervals == 0:
            self.n_intervals = 1
        if self.series_length < self.min_interval:
            self.min_interval = self.series_length
        self.intervals = np.zeros((self.n_estimators, self.n_intervals, 2),
                                  dtype=int)
        for i in range(self.n_estimators):
            transformed_x = np.empty(shape=(3 * self.n_intervals, n_instances))
            # Find the random intervals for classifier i and concatentate
            # features
            for j in range(self.n_intervals):
                self.intervals[i][j][0] = rng.randint(
                    self.series_length - self.min_interval)
                length = rng.randint(
                    self.series_length - self.intervals[i][j][0] - 1)
                if length < self.min_interval:
                    length = self.min_interval
                self.intervals[i][j][1] = self.intervals[i][j][0] + length
                # Transforms here, just hard coding it, so not configurable
                means = np.mean(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"], axis=1)
                std_dev = np.std(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"], axis=1)
                slope = self._lsq_fit(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"])
                transformed_x[3 * j] = means
                transformed_x[3 * j + 1] = std_dev
                transformed_x[3 * j + 2] = slope
            tree = clone(self.base_estimator)
            tree.set_params(**{"random_state": self.random_state})
            transformed_x = transformed_x.T
            tree.fit(transformed_x, y)
            self.classifiers.append(tree)
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        proba = self.predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Local variables
        ----------
        n_test_instances     : int, number of cases to classify
        series_length    : int, number of attributes in X, must match
        _num_atts determined in fit

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        self.check_is_fitted()
        # X = check_X(X, enforce_univariate=True)
        assert isinstance(X, ak.highlevel.Array)
        # X = tabularize(X, return_array=True)

        n_test_instances, series_length = X.shape[0], X[0, 0].shape[0]
        if series_length != self.series_length:
            raise TypeError(
                " ERROR number of attributes in the train does not match "
                "that in the test data")
        sums = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for i in range(0, self.n_estimators):
            transformed_x = np.empty(
                shape=(3 * self.n_intervals, n_test_instances),
                dtype=np.float32)
            for j in range(0, self.n_intervals):
                means = np.mean(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"], axis=1)
                std_dev = np.std(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"],
                    axis=1)
                slope = self._lsq_fit(
                    X[:, 0, self.intervals[i][j][0]:self.intervals[i][j][
                        1], "value"])
                transformed_x[3 * j] = means
                transformed_x[3 * j + 1] = std_dev
                transformed_x[3 * j + 2] = slope
            transformed_x = transformed_x.T
            sums += self.classifiers[i].predict_proba(transformed_x)

        output = sums / (np.ones(self.n_classes) * self.n_estimators)
        return output

    def _lsq_fit(self, Y):
        """ Find the slope for each series (row) of Y
        Parameters
        ----------
        Y: array of shape = [n_samps, interval_size]

        Returns
        ----------
        slope: array of shape = [n_samps]

        """
        x = np.arange(Y[0].shape[0]) + 1
        slope = (np.mean(x * np.array(Y), axis=1)
                 - np.mean(x) * np.mean(Y, axis=1)) / (
                        (x * x).mean() - x.mean() ** 2)
        return slope
