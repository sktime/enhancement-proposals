# Panel/supervised forecasting

Contributors: @partmor, @fkiraly, @mloning,

This proposal describes a design for a supervised forecasting module.  
It specifies a template class signature for supervised forecasters.

## Interface design targets

sktime currently has two templates which are closely related:

* supervised time series classification and regression
* forecasting

The supervised forecasting design should be concordant with the two. In particular:

* "apply forecaster by row" should be close to the forecasting interface
* "forecast one time point" should be  close to the supervised time series regression interface

## Design principles

We wish to adhere to multiple principles:

* consistency with existing template interfaces in sktime
* easy extensibility - the ease or difficulty to build custom estimators or extend sktime should not be affected
* avoiding user frustration - natural user expectations on interface behaviour should be met
* adherence with sklearn style design principles - unified interface (strategy pattern), modularity, sensible defaults, etc
* downwards compatibility - any change should not impact code written in earlier versions of the interface

## Conceptual design

The design implements generic behaviour of a supervised forecaster as specified in the AG/FK manuscript.

Key interface elements in the user journey are, in this sequence:

* construction of the supervised forecaster - implemented by constructor `__init__`
* fitting of the forecaster (two steps) - implemented by `fit` and `fit_forecaster` methods
* application of the forecaster - implemented by `predict` method
* optional: updating the model - implemented by `update` and `update_forecaster` methods, then potentially follows by `predict`.

A supervised forecaster has three states:

1. unfitted - no data ingested
2. fitted - training past/future sequence ingested
3. forecaster fitted on test data - "past" of test set ingested

State changes are as follows:

* to 1: construction with parameter and model specification
* from 1 to 2: "fitting", ingesting training data
* from 2 to 3: "fitting forecaster", ingesting past of test set

The design below maps the state changes as follows on methods:

* to 1 is implemented by constructor `__init__`
* 1 to 2 is implemented by the `fit` method
* 2 to 3 is implemented by the `fit_forecaster` method; alternatively, `predict` can be called directly after `fit`, in which case it also runs `fit_forecaster`. If `predict` is run after `fit` and `fit_forecaster`, there is no state change.
* `update` and `update_forecaster` methods do not change the state in the above sense, but ingest more data and update the fitted model

For the statistical/relational data model:

* training data are independent instances of series, with a designated "past" and "future" part
* deployment data are independent instances of series, all in the "past"
* test data are independent instances of series, with a designated "past" and "future" part

Explicitly, observations within series are not assumed independent.  
There is a temporal continuity and association assumed between "past" and "future" part.

## Template design

The template class `BaseSupForecaster` inherits from `sklearn` `BaseEstimator`.



## First steps



## Notes from 1st meeting

Starting point:
* implement reduction to tabular learning

First models to implement:
* reduction to supervised tabular
* mixed effects models
* reduction 2-step: use only constant variables for mean then another estimator for forecast

Other resources
* Master thesis (to be uploaded)
* previous design notes (will see if I can find any)
