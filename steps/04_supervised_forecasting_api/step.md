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

It has the following methods:

`fit(X : Panel, fh=None : ForecastingHorizon, cutoffs=None : np.array) -> self`  
Behaviour: fits the model to training data `X`  
Args:  
`X` - `Panel` type, independent instances of training data
`fh` - forecasting horizon object, optional  
`cutoffs` - same length as `X`, i-th entry is cut-off time points (past vs future) of the i-th sample in `X`

`fit_forecast(X : Panel, fh=None : ForecastingHorizon) -> self`
Behaviour: fits the model to past of prediction data `X` - usually requires `fit`  
Args:  
`X` - `Panel`, independent instances of prediction data  
`fh` - forecasting horizon object, optional - should only be passed if not passed in `fit`  

`predict(X=None : Panel, fh=None : ForecastingHorizon, return_pred_int=False, alpha=DEFAULT_ALPHA)-> Panel`  
Behaviour: makes prediction for prediction series in `X` - requires `fit`, but `fit_forecast` is optional; if requires `fit_forecast` but not run, runs it first  
Args:  
`X` - `Panel`, independent instances of prediction data  
`fh` - forecasting horizon object, required if not passed earlier  
`return_pred_int` - whether prediction intervals should be returned  
`alpha` - prediction interval alpha  
Returns:  
`Panel` type object with predictions  
if `return_pred` is `True`, also returns `Panel` objects that correspond to prediction intervals  

Should we also have  
`update`  
`update_predict`  
?

Tags:

* `requires_fit` - does the model require a call to `fit` before `predict`?
* `requres_fit_forecast` - does the model require a call to `fit_forecast`?
* `requires_cutoff` - does the model need a cutoff to specify in `fit`?

## Implementing complex state behaviour

For usability, users can call `predict` directly after `fit`, although it may be more efficient to call `fit_forecast` first.

This can be accomplished by decorating `fit` with a method `fit_forecast_check` that checks whether `fit_forecast` has been called and needs to be called. If has not been called but needs to, runs `fit_forecast` first with arguments passed.    

## Metrics design

Due to the nature of the task, metrics need to compare samples of series to samples of series, i.e., will be of the signature

`mymetric(y_true: Panel, y_pred: Panel, * , sample_weight=None)`

I would suggest to implement metrics not by hand, but by creating one or few decorator factories, that average `sklearn` metrics over time points.

That is, for example, create a function

`make_idx_avg_panel_metric(sklearn_metric : type(mean_squared_error)) -> type(mymetric)`

which creates the panel metric that is the argument metric averaged first over entries in a series, then over the test samples.

To define metrics, define shorthands and create docstrings for the panel metrics created in this way.

## First steps

Suggested first steps:

* implement template class, perhaps without `update`
* implement decorator `fit_forecast_check` for `fit`
* write template python file with only gaps to fill in
* implement "simple example" estimators
* implement simple metrics - averaged MSE and MAE


## Simple examples estimators to implement

* reduction to tabular supervised regression: splits `X` in `fit` into two, by cut-off - past becomes `X`, future becomes `y` for an `sklearn` supervised learner
* reduction to tabular supervised time series regression: splits `X` in `fit` into two, by cut-off - past becomes `X`, future becomes `y`, or a time series regressor
* "apply forecaster on the test set by row" - does not require `fit`, and applies copies o the wrapped forecaster to each sample in the prediction set
* some linear mixed effects model from `statsmodels`

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
