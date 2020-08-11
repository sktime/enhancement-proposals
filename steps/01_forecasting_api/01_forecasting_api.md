# Forecasting API proposal

Contributors: @mloning, @fkiraly, @matteogales, @big-o, @prockenschaub 

## Overview

### Forecasting
Forecasting describes the learning task in which we want to make temporal forward predictions of a given univariate time series. We may have additional exogenous, potentially multivariate time series which we can use as extra input data to predict the target series.

### API design
Forecasting involves a number of choices on how to fit forecasters and how to make and update forecasts. API design is about how to best map these choices onto classes and methods in Python. 

With sktime, our goal is to design a uniform interface for forecasting, similar to the scikit-learn's uniform interface for tabular (or cross-sectional) prediction setting. All forecasters need to have methods for at least the following functionality: 

1. Model specification (constructor, hyper-parameters)
2. Training/estimation
3. Inspection (fitted parameters)
4. Application (forecasting, in-sample/out-of-sample)
5. Updating
6. Persistence (save, load) 

Before looking at concrete proposals, we discuss these methods and related key concepts and questions below. 

For preliminary discussions of the API design presented here, see issue [#18](https://github.com/alan-turing-institute/sktime/issues/18). 

### Content
**I.** [**Key concepts and questions**](#Key-concepts-and-questions)
1. [Forecasting horizon](#Forecasting-horizon)
2. [Model specificaton](#Model-specification)
3. [Training](#Training)
4. [Inspection](#Inspection)
5. [Prediction](#Prediction)
6. [In-sample predictions](#Insample-predictions)
7. [Updating: online leanring](#Updating:-online-learning)
8. [Prediction-intervals](#Prediction-intervals)
9. [Exogenous variables: nowcasting vs forecasting](#Exogenous-variables:-nowcasting-vs-forecasting)

**II.** [**API proposals**](#API-proposals)
1. [Proposal based on methods signature](#Proposal-based-on-methods/keyword-arguments)
2. [Proposal based on dispatch on task objects](#Proposal-based-on-dispatch-on-task-objects)
3. [Class hierarchy](#Class-hierarchy)


## Key concepts and questions
### Forecasting horizon
The forecasting horizon specifies the time periods (or steps or time points) which we want to predict. In deployment, the forecasting horizon is in the future. In model evaluation, it is a subset of the test set. When assessing the goodness-of-fit, it is often part of the training set. 

With regard to interface design, there are three key questions:
* When do we specify the forecasting horizon - at construction, training or prediction?
* How do we specify the forecasting horizon, e.g. as a new object possibly as part of a full task object or simple positional or keyword arguments?
* Should the interface to specify the forecasting horizon be uniform?

While the first two questions are less clear, most people agree that whatever interface we choose for the forecasting horizon, it should be uniform, i.e. it should be specified and provided in the same way for all forecasters and methods.

For _out-of-sample predictions_, the forecasting horizon could conveniently be specified as an array with the future time periods to forecast, relative to the end of the training set. For example:

```python
fh = np.array([1])
fh = np.array([3, 4, 5])
fh = np.arange(1, 10)
```

Simply specifying the forecasting horizon as the number of steps ahead is not enough, as some forecasters may train a separate models for each step, so it is important to know exactly which time periods we want to forecast.

For *in-sample predictions*, please see the [section below](#Insample-predictions).

__Questions:__
* For convenience, the forecasting horizon may be specified as an integer. Should the given integer in this case indicate the exact step ahead or the number of steps ahead? 
 
While the number of steps ahead may be the more common use case, it risks confusing users, as `fh=2` and `fh=np.array([2])` would no longer denote the same forecasting horizon. 

__Implementation notes:__
* Appropriate input checks for the passed forecasting horizon are needed, checks should return sorted forecasting horizon in form of a numpy array (e.g. transforming string values to corresponding forecasting horizon based on training data). 
* Is any of this affected when we eventually add support for date and time indices (on an absolute time scale)? For example, date/time ranges could be passed which do not necessarily align with the actual observed time points. 
* One could also allow for fractional indices to be given in the forecasting horizon instead of only integers, e.g. one may want to make forecasts on interpolated time points at higher frequencies than the observed frequency.   

### Model specification
To specify a model, user can specify different hyper-parameters, ideally with sensible defaults for all of them. In principle, models specification should be separate from data and separate from task specific information. 

__Questions:__
* Based on what principle to we decide which arguments go in the constructor and in other methods? Typically, (tunable) hyper-parameters and parameters that pick out a particular model instance from a wider model family are defined the the constructor. As an aside, note that scikit-learn, for example, also includes task specific information in the constructor, e.g. the class weights used in classification, possibly because of the lack of a more expressive task specification interface. 
* In some cases of composite and reduction forecasters, task related information specifies a model (e.g. one may want to specify some model to forecast values for a certain forecasting horizon and then apply a different model to these forecasts to obtain the final forecasts). In these cases, this information could be passed to the constructor as it defines the particular composite strategy rather than the ultimate learning task. 
* Should the forecasting horizon be specified in the constructor even though it is part of task specification? One reason may be that for some forecasters, the forecasting horizon changes the structure of the forecasters (e.g. reduction to regression where one model is learnt for each period of the forecasting horizon), but note that this is only one strategy among others to reduce forecasting to regression.

### Training
In training, we fit a forecaster to the available past data (i.e. the training set). 

While most forecasters need the forecasting horizon only during prediction, some need it already during training. Examples of forecasters that require the forecasting horizon during training include: 
* the direct reduction strategy to regression where one forecaster is fitted for each step ahead in the forecasting horizon, 
* tuning algorithms where the a best forecaster is selected for each step ahead.

To create a uniform interface, all forecasters should accept the forecasting horizon in training, even if the particular forecaster does not need it for training. So the training method needs to know about: 
* Training data
* Forecasting horizon (optional)

__Implementation notes:__
* Any forecasting horizon passed in training should be remember by the forecaster and used in prediction. If a different
 forecasting horizon is given in prediction than in training, the new one should be used, but an appropriate warning
  should be raised. 
* Appropriate errors should be raised if no forecasting horizon is passed but required in training.
* For those forecasters that require past data to make forecasts, the necessary data needs to be stored in the forecaster during training, so that it can be accessed during prediction (e.g. the last window of the training set for out-of-sample predictions and the first window for in-sample predictions).

__Questions:__
* Should the whole training data always be stored in the forecaster? 

### Inspection
Forecasters should not only have a uniform interface for hyper-parameters, but also for fitted parameters. 

```python
def get_fitted_params(self):
    pass
```

This is useful for reduction approaches. For example, when solving a time series classification task, one could fit a forecaster on the time series of each instance and use the learnt parameters as features, i.e. a form of feature extraction (see e.g. the RISE classifier or tsfresh library).

In addition, users would ideally be able to manually set fitted parameters. 

```python
def set_fitted_params(self, **params):
    pass
```

__Implementation notes:__
* When interfacing or wrapping other methods, their hyper-parameters and fitted parameters should be exposed in a uniform interface, similar to the forecasters implemented directly in sktime. 

### Prediction
Once we have a fitted forecaster, we can make forecasts for a given forecasting horizon. 

In most situations, users will already know the forecasting horizon of interest when fitting the forecaster. To still allow users to change it without forcing them to re-fit the whole model, `predict()` also needs to accept a forecasting horizon. 

So, the prediction interface needs to know: 
* Forecasting horizon (optional)


__Implementation notes:__
* For those forecasters that require some past data to make forecasts, the required past data has to be stored in training, both the first window for in-sample predictions and the last window for out-of-sample predictions. 
* For those forecasters that depend on the forecasting horizon in training, we need to check that any new forecasting horizon passed in prediciton is a subset of the one passed in training. 
* Warnings should be raised if the forecasting horizon given in training is overwritten with a forecasting horizon passed in prediction.

### Insample predictions
In addition to forecasting future values, we may also be interested in *in-sample predictions* for the time points seen in the training set. This is important for 
* assessing the goodness-of-fit of a model qualitatively by comparing the observed values with the in-sample predictions,
* for computing residuals which are in turn important for obtaining prediction intervals, 
* for detrending pipelines, when we first want to detrend a time series before fitting a forecaster on the detrended series.

#### A few options 
To allow for in-sample predictions, we have a couple of options, including

* an *additional keyword argument* in `fit()` allowing the user to specify whether the passed forecasting horizon is relative to the end or start of the training set,
```python
def fit(y_train, fh=None, insample=False):
    pass
```

* allowing *negative values* in the forecasting horizon,  going backwards from the end of the training series

```python
fh = np.array([-len(y_train), 0])  # from the first to the last time point of the training set
fh = np.array([-2, -1, 0])  # the last three time points of the training set
```

* accepting an *"insample" string value* for the forecasting horizon to return in-sample forecasts for entire training series

```python
fh = "insample"  # convenient keyword arguments
```

* adding a *separate method* specifically for in-sample forecasts, any forecasting horizon passed to this method would be interpreted relative to the start of the training series
```python
def predict_in_sample(y_train, fh=None, return_pred_int=False, alpha=0.05):
    pass
```

__Implementation notes:__
* Forecasting horizons are not only used in forecasters, but also in other classes like temporal cross-validators, but in-sample forecasts only make sense for forecasters. So, extending the accepted input formats to string values or negative values would force us to write separate input validation functions. When using a separate method, input validation does not change. 
* Internally, the routines for in-sample predictions are likely to be factored out into their own methods anyway.
* Having a separate method would allow us to ingest the training data again if necessary, so that we don't have to stored in the forecasters.
* However, having a separate interface point for prediction creates problems: for example, when detrending a series with the predicted values from a forecaster, after the forecaster is fitted on the training series, during `transform()` or `inverse_transform()` the detrender needs to find out whether the series to be transformed is in-sample, out-of-sample or some of both, then call the methods accordingly and finally append the results. With a single method, this is dealt with in the forecaster. 

### Updating: online learning 
While in some cases we only ever want to make a single forecast, in most real-world cases we want to repeatedly update our forecasts as soon as new data becomes available. 

Updating can take two forms. Some forecasters use past data only during training. Once they are fitted, no past data is needed for prediction (e.g. ARIMA). These forecasters can be updated by updating their fitted parameters (e.g. using Kalman smoothing for hidden state representation of ARIMA models or running additional iterations of the optimiser). 

Other forecasters use past data during training and prediction (e.g. reduction approaches to time series regression). These forecasters can be updated by updating the past data that is available to them during prediction and optionally also their fitted parameters (e.g. in the reduced time series regression setting, updating the last window used by a perceptron algorithm to make forecasts and optionally running a batch update to update its fitted weights). 

So, the updating method needs to know: 
* New data 
* Option to specify whether to also update the fitted parameters.

One could of course also update forecasts by completely re-fitting the forecasters to the new data, but this is already covered by the training interface.

In addition to having an `update()` method for making a single update, forecasters need a method to update and make forecasts more dynamically. 

* In model evaluation, an `update_predict()` method gives a more convenient method for obtaining predictions from
 temporal cross-validation over the entire test set. 
* In deployment or for single predictions, many forecasters have more efficient joint update and predict procedures than
 simply calling them iteratively, so we may also want an `update_predict_single()` method.

The joint update-predict method then needs to know:
* New data (e.g. entire test set) 
* Time series splitter or cross-validation iterator specifying how to iterate over the new data

__Question:__
* In some cases, one may want to update forecasters by passing additional past data. Should we still raise warnings if the passed data is not newer than the data seen during training?

### Prediction intervals
In addition to point forecasts, we are also interested in obtaining some measure of uncertainty around our point forecasts.

Prediction intervals require a number of additional arguments. The key design question is whether we should introduce a separate method for prediction intervals or allow `predict()` to have a varying number of return object, either solely returning point forecasts or returning point forecasts and intervals.

__Questions:__
* Do prediction intervals depend on training, i.e. is it necessary in some cases to specify prediction interval arguments in `fit()`? 
* Do prediction intervals depend on point predictions, i.e. is it  always necessary to call `predict()` before computing prediction intervals? 

## Exogenous variables: nowcasting vs forecasting
Many forecasters can also use information in additional exogenous, potentially multivariate time series data. In deployment, these situation are fundamentally different: In nowcasting, exogenous variables are available when making predictions. In forecasting, strictly speaking, the exogenous variables are not available. 

In many popular libraries (e.g. R's forecast or statsmodels in Python), the forecaster becomes a nowcaster by default when exogenous variables are passed, so that we also have to pass values for the exogenous variables for time periods of the forecasting horizon when making predictions (e.g. in ARIMA). By contrast, in forecasting, strictly speaking, the exogenous variables are only available up to the point in time at which we make forecasts and need not be provided for prediction. 

__Implementation notes:__
* The decision whether defaulting to nowcasting when exogenous data is given affects training, updating and prediction interfaces. For example, when passing exogenous variables to `update_predict()` it is not clear how to distinguish between whether the data should only be used for updating, or for both updating and predicting (i.e. nowcasting).

__Questions:__
* How should we distinguish between nowcasters and forecasters? One could adopt the viewpoint that all forecasters may receive exogenous series for the future, but some choose to ignore it. From that viewpoint, the distinction would move to task specification rather than implying a split in the estimator inheritance hierarchy.
* If we followed scitypes, we would keep them separate, but given that we're interfacing existing package, it may require a lot of code duplications to actually keep them separate.

### Model evaluation
In principle, evaluation specific information should be explicit and not encapsulated in classes or methods. For example, training and test split should be specified explicitly outside of methods (e.g. in sliding window model evaluation). 

However, in an online learning setting, we provide the `update_predict()` method for convenience and more efficient dynamic update routines. 

If we have a convenience `score()` method in line with scikit-learn, it needs to know everything `predict
()` needs to know. For dynamic updating, `update_score()` would need to know everything `update_predict()` needs to know.
 
#### Performance metrics
Performance metrics compare a series of forecasts with a series of observed (or "true") values and assign some score to the difference between them. In model evaluation, it is common practice to use temporal cross-validation, i.e. a sliding window to repeatedly update and make forecasts. When forecasting multiple steps ahead, forecasts can overlap so that we have multiple forecasts for the same observed "true" value. As a consequence, performance metrics for forecasting need to handle multiple forecast values and return appropriately aggregated scores. 


## API proposals
We consider two main proposals for the forecasting API. 

1. Based on class methods and method signatures, inspired by [statsmodels](https://www.statsmodels.org/stable/index.html) and [pmdarima](https://github.com/alkaline-ml/pmdarima) among others,
2. Based on task object to encapsulate the different choices in a single object, together with simple class methods, inspired by [mlr](https://mlr.mlr-org.com) and [openML](https://www.openml.org).

While the first proposal tries to develop a uniform forecasting interface by adapting and extending the familiar scikit-learn API, the second proposal follows a different approach. Tasks are an attempt to separate information about the learning tasks from information about the estimator which used to solve a given task. For example, tasks would encapsulate information on which variables are exogenous and which one is the endogenous variable, the forecasting horizon and so on. Tasks may also be useful in the simpler, cross-sectional supervised learning setting. For example, in scikit-learn, the estimators interface confounds both task information (e.g. class weights in classifiers) and estimator parameters. 

However, tasks are not supported by scikit-learn's interface. So the task based proposal will make it harder for developer to use and integrate scikit-learn's existing functionality as well as for new users to become familiar with the new API. 

There are also some open questions about tasks:

*  How could users change task during the machine learning workflow, e.g. changing the forecasting horizon after training?  

Of course, we could also try to combine both approaches giving users the option which one to use, similar to [Keras](https://keras.io) which also offers two ways of defining models (functional and sequential API). In this case, the tasks could allow for greater expressivity and more specialised options, while the scikit-learn familiar approach based on methods is still available for simpler workflows. 
 

### Proposal based on methods/keyword arguments

```python
class Forecaster(BaseForecaster):
    
    def __init__(self, **hyperparams):
        pass
    
    def fit(self, y_train, fh=None, X_train=None):
        pass
    
    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        pass
    
    def update(self, y_new, X_new=None, update_params=False):
        pass
    
    def update_predict(self, y_test, cv=None, X_test=None, udpate_params=False, return_pred_int=False, alpha=0.05):
        pass
        
    def score(self, y_test, fh=None, X_test=None):
        pass

    def update_score(self, y_test, cv=None, X_test=None, update_params=False):
        pass

    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False, alpha=0.05):
        pass
        
    def get_fitted_params(self):
        pass
```

Note that differentiating between the names of the input variables (i.e. `y_train`, `y_new` and `y_test`) can help users understand the difference in the methods and when they are useful. However, the same does not work for the exogenous variables (i.e. `X`), as they may be used for updating only, or also for nowcasting. 

Also note that this approach could be combined with the task based approach outlined below, by adding additional methods (e.g. inherited via a mix-in class), for example: 

```python
def fit_task(self, train, task):
    pass
    
def predict_task(self):
    pass
```


### Proposal based on dispatch on task objects
```python
class ForecastingTask(BaseTask):
    
    def __init__(
        self, 
        target,
        fh,
        X=None,
        return_conf_int=False,
        update_params=False,
        cv=None,
        alpha=0.05,
        ):
        pass
```

```python    
class Forecaster(BaseForecaster):
    
    def __init__(self, **hyperparams):
        pass
    
    def fit(self, train, task):
        pass
    
    def predict(self):
        pass
    
    def update(self, data):
        pass
    
    def udpate_predict(self, test):
        pass   
        
    def modify_task(self, task):
        pass
```

__Implementation notes:__
* To allow user to modify their specified task object, there must be a way to update the task information in the forecaster. This is relevant for example, when one wants to change prediction interval parameters (e.g. alpha) after training if alpha or change the forecasting horizon after training.
* In contrast to the approach based on methods/keyword arguments, the input data format varies. If exogenous variables are available, it has to be a data frame. If no exogenous variables are available, it should be a series.  

### Class hierarchy
What are sensible base classes? 

* BaseTemporalEstimator(BaseEstimator) 
    * inherits from sklearn's BaseEstimator
    * implements common methods for keeping track of observations horizon and current state in time ("now") 
    * parent of transformers (e.g. Detrender) and forecasters

* BaseForecaster(BaseTemporalEstimator)
    * defines abstract methods for all forecasters (shared API) 
    * adds common default routines (e.g. `update_predict()` by calling `update()` and `predict()` iteratively)
    
* BaseForecasterOptionalFHinFit(BaseForecaster)
    * adds setting/getting of fh for forecasters which take the forecasting horizon either in `fit()` or `predict()`

* BaseForecasterRequiredFHinFit(BaseForecaster)
    * adds setting/getting of fh for forecasters which require the forecasting horizon in `fit()`
