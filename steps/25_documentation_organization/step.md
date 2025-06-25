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

In general, the user will be interested in a specific timeseries task: Forecasting, Classification, Clustering, or Regression.

## Proposed information architecture

Home | Get Started ▾ | Tutorial | How-to | Estimator Overview | API Reference | More ▾


### Get started

* Installation
* 5-min quick start for any of the 4 tasks (Forecasting, Classification, Clustering, Regression)
* Why sktime? Showcase the motivation of sktime, and why to use it, the benefits of its APIs and community

### Tutorial

Learning-oriented docs, focused on the basics and to make the user learn sktime API.

* Forecasting
    * Forecasting univariate timeseries
    * Transformations and compositions
    * Forecast panel datasets, broadcasting
    * Forecasting hierarchical data
    * Forecasting with scikit-learn like regressors
    * Global and zero-shot forecasting
* Anomaly, segmentation
* Classification
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
* How-to benchmark different models


### Reference

Standard API reference

### Explanation

Explain motivation of choices, architecture design and different aspects of the library.

* Why `y`, `X` and `fh` and not a single dataframe as data in the API?
