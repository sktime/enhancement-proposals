# Rethinking sktime documentation structure

In the current state, sktime documentation is comprehensive and explain most of the features of the library. However, it is worth rethinking its structure. Some negative feedbacks about sktime are focused on the documentation, since, for new users, they are really dense in information and can be confusing.

This step proposes rethinking sktime documentation with a focus on the user experience. The first exercise is defining the different personas that might be interested in the package, and how would be the best organization for their purpose. The current state is dense and might be best for the ones already familiar with Python and advanced features of dependencies, such as pandas and multiindex.


## Personas

We can identify the following personas:

* **User A**: Student, Junior, new to timeseries
* **User B**: knows scikit-learn API, needs-production grade pipelines
* **User C**: Academic Researcher, wants to benchmark and create new models
* **User D**: OSS contributor, Python developer
* **User E**: familiar to other timeseries packages, want to compare and get to know sktime.

In general, the user will be interested in a specific timeseries task: Forecasting, Classification, Clustering, Detection, or Regression.

## Proposed information architecture

Home | Get Started | Tutorial | How-to | Estimator Overview | API Reference | Get Involved | More ▾


### Home

This is what users see first when they access sktime's documentation. We should highlight the versatility of the package. A suggestion would be highlighting how one can change `NaiveForecaster` to `Chronos`/`NBEATS` in 1 line of code to produce forecasts with a state-of-the-art model.

We could also have links for the different personas, e.g.:


* New to timeseries forecasting? Go to the univariate forecasting tutorial
* Researcher? Check-out how to benchmark your models
* Scikit-learn user? Learn how to use regressors for forecasting
* Want to contribute? Check-out the contributing guidelines

Also adding discord, linkedin urls.

### Get started

* Installation
* 5-min quick start for any of the 4 tasks (Forecasting, Classification, Clustering, Regression)
* Why sktime? Showcase the motivation of sktime, and why to use it, the benefits of its APIs and community

### Tutorial

Learning-oriented docs, focused on the basics and to make the user learn sktime API.

* Forecasting
    * Forecasting univariate timeseries (15 min)
        * Load of a simple dataset (e.g. Airline), without exogenous variables
        * Brief explanation of pandas structure in sktime
        * Arguments for sktime forecasting API: y, fh
        * Show relative and absolute fh
        * Forecast with a simple model (e.g. exponential smoothing)
    * Forecasting with exogenous variables (10 min)
        * Forecast with a more advanced model (e.g. Chronos, to showcase versatility)
        *  Forecasting with exogenous variables
        * Dataset with exogenous variables
        * `all_estimators` to get exogenous variables
        * Usage of a simple forecaster that uses exogenous variables: AutoREG
    * Transformations (10 min)
        * Context about transformations, motivation
        * Transformations for target variable (differencing, detrending)
        * Feature Engineering: FourierFeatures, Holidays
    * Pipelines (10 min)
        * Motivation: avoid data leakage, reproducibility...
        * Target transformations
        * Transformations for exogenous variables
        * Composition with both types of transformations
        * `get_params` and `set_params` for compositions
    * Cross-validation and metrics (10 min)
        * Splitters, and their plot_windows
        * Metrics
        * `evaluate`
        * Conclusion & call to click to other tutorials
    * Hyperparameter tuning (10 min)
        * Tuners in sktime
        * Tuning of a simple model
        * Tuning of a composition
        * Cross-validation of a tuner
    * Probabilistic forecasting (20 min)
        * Brief Motivation
        * showcase probabilistic forecasting with `predict_interval` using a simple forecaster
        * Enumerate other forecasting methods (quantiles, variance etc)
        * Detail briefly probabilistic forecasting behaviour with compositions
        * Metrics for probabilistic forecasting
        * Conformal wrappers and boostrapping
    * Forecasting multiple series (panel datasets) (15 min)
        * Use a dataset with panel data, but not that large. Something that can be easily executed in a computer, < 1 min for fitting
        * Motivation
        * Details of pandas dataframe structure, and useful operations
        * Call of `fit` and `predict` with a simple univariate model
        * Demonstration of `.forecasters_` attribute
        * Metrics for panel data (aggregation)
        * Conclusion and connection to global forecasting
    * Forecasting with scikit-learn like regressors (15 min)
        * dataset loading
        * `window_summarizer`, `make_reduction`
        * Why differencing can be useful: capturing the trend
        * Pipeline of transformations + reduction
        * `get_params` and its recursive behaviour
    * Global and zero-shot forecasting (15 min)
        * Definition of global forecasting
        * Global forecasting with Reduction Forecasters
        * Global forecasting with Deep Learning
        * Zero-shot forecasting
    * Hierarchical forecasting (15 min)
        * Context of the problem
        * Reconciliation strategies
        * Reconciliation transformations (new API)
        * ReconcileForecaster and mint
    * Machine-types (mtypes) and Scientific Types (scitypes) (10 min)
        * What are mtypes and scitypes
        * How to use polars with sktime
* Detection and segmentation
    * Anomaly Detection
    * Changepoint detection
    * Segmentation
* Classification
    * Introduction to timeseries classification
    * Advanced Classification methods
* Clustering
* Regression


These documentations should feel like a learning experience, focused on learning and for new users and the ones that want a deep understanding


### How-to

Goal oriented documentations, recipes, for users that already have certain knowledge about sktime framework.

* How-to create an AutoML pipeline
* How-to tune hyperparameters
* How-to create custom:
    * forecasters
    * classifiers
    * regressors
    * anomaly detectors
* How-to use clustering with forecasters
* How-to benchmark different models
* How-to cross-validate global models

## Examples

Examples should be application oriented, end-to-end.
We could temporarily leave the existing notebooks here, to be removed after the refactoring is completed.

* Energy Forecasting
* Financial timeseries
* Retail forecasting
* Healthcare applications
* (old notebooks)

### Reference

Standard API reference

### Get Involved

Description of the governance model and how to contribute, benefits of contributing.


### More

Here, we could include the following menus:

* Development
* About
