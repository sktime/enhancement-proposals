# probabilistic forecasting API

Contributors: @fkiraly, @danbartl, @ltsaprounis, @mloning, @aiwalter, @satyapattnaik

## High-level summary 

### The Aim

`sktime` has robust functionality for forecasting, but that covers only forecasting of a single time series, without hierarchy information.

It would be desirable to introduce functionality that also allows for panel and hierarchical forecasting.

### The proposed solution

Our proposed solution is a new base class, closely following the current forecaster design. Long-term, this base class may replace the current forecaster base class, since it extends its functionality.

Key design elements:

* allowing `X` and `y` to be panels or multi-indexed with arbitrary number of levels
* introduction of a `Z` argument for time-constant metadata
* downwards compatibility with the current `BaseForecaster`

## Design: hierarchical forecasting

We proceed outlining the refactor target interface.

### data container specification for hierarchical time series

Below, variables `y` and `X` can be hierarchical time series.
For the prototype, there will be only one `mtype` representing hierarchical time series, as follows.

A hierarchical time series is represented by a `pd.DataFrame`, which has a multi-index of `n` levels, `n` being 1 or larger. The last, `n`-th level of the index must be one of the supported temporal indices in `sktime` (see data type specification). The other levels can have arbitrary values.

Post-prototype, further `mtype`-s may bei ntroduced.

### methods: public signature specification

We specify the function signatures of the new methods.
To facilitate testing of the design, we omit special `predict` and `update` variants.

The class inherits from `sktime`'s `BaseEstimator` (and hence `BaseObject`).

```python
    def fit(self, y, X=None, Z=None, fh=None):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh if fh is passed.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
            if self.get_tag("requires-fh-in-fit"), must be passed, not optional
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index
        Z : pd.DataFrame, optional (default=None)
            index of Z must equal index of y with the first level removed
            can only be passed if y is a multi-index with at least 2 levels

        Returns
        -------
        self : Reference to self.
        """
```

```python
    def predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed in _fit.

        Parameters
        ----------
        fh : int, list, np.ndarray or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts at fh
            index is multi-index with non-temporal levels equal to y in fit
                temporal (last) level equal to fh.index
            y_pred has same columns as y passed in fit
        """

```

```python
    def update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:
            update_params=True: fitting to all observed data so far
            update_params=False: updates cutoff and remembers data only

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self. cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
```

Further function with usual default is `fit_predict`.

### public/private interface

Each method has a private counterpart which is called internally, i.e., `_fit`, `_predict`, `_update`. As in the generic `sktime` design, the public method contains "plumbing", while the private class is the extension locus and contains only method logic.

### apply-by-level

To allow leveraging existing forecasters, we introduce default "apply-by-level" functionality.

For this, a boolean tag `scitype:hierarchical` is introduced - if False, the forecaster is not genuinely hierarchical. In this case, `_fit`, `_predict` etc are only defined for "plain" (1 level indexed) time series. If a hierarchical time series with 2 or more levels is passed to `fit`, `predict`, etc, copies of the forecasters are made per instance (in an attribute `hierarchy_estimators_`) and fitted/predicted/updated per instance.

### Downwards compatibility with current `BaseForecaster`

Ultimately, the idea is that the current `BaseForecaster` can be replaced with the new base class.
For this, the following steps need to be taken beyond simple replacement:

* default tag setting `"scitype:hierarchical=False"` for existing non-hierarchical forecasters.
* ensure that additional input/output types are properly handled in `Series` and `Panel` case, this is not detailed in the above prototype design
* ensure that current forecaster tests pass
* implement special `predict` and `update` methods (with appropriate signatures), e.g., `predict_quantiles`, `update_predict`


## Development steps

### Step 1 - base class, mocks, tests

As first step, the base class, and one or two simple estimators should be implemented, together with checks and a test suite closely mimicking the forecaster suite.

This should *exclude* the "downwards compatibility" items.

The following should be taken into account:

* only `pd.DataFrame` should be supported on the first prototype, so `pd.Series` based tests need to be replaced by `pd.DataFrame` based ones
* hierarchical inputs need to be included in the tests
* `Z` needs to be included in checks (e.g., same levels as `y`) and tests
* definition and checks for the hierarchical `mtype` should be included in the `datatypes` module, e.g., checking that the last level is temporal of a supported type
* generators for hierarchical `mtype` are needed for use in the tests
* between `fit` and `update`, it needs to be checked that `y`, `X` have same levels
* the `"scitype:hierarchical"` tag functionality needs to be tested

The "apply-by-level" functionality can also be implemented after step 3.

### Step 2 - concrete estimators

This should now include common classes for hierarchical modelling:

* multi-factor time series models
* reconciliation meta-algorithms, these should be compositors
* reduction approaches using indices as covariates

### Step 3 - tutorial

A tutorial for hierarchical forecasting should be written.

### Step 4 - extending the interface

Once the above is done, the interface should be extended as in the "downwards compatibility" section.

Importantly, the extension should ensure support for additional `Series` and `Panel` `mtype`-s.
In particular, `Panel` `mtype`-s need to tie into the "apply-by-level" functionality.

### Step 5 - merging the forecaster base classes

Once the above is done, the old (non-hierarchical) `BaseForecaster` can be replaced with the new class, as it will satisfy all its contracts while providing additional functionality.
