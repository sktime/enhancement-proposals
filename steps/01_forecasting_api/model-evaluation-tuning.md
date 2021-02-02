# Forecasting: Model selection & evaluation

Reference issue: [#622](https://github.com/alan-turing-institute/sktime/issues/622), [597](https://github.com/alan-turing-institute/sktime/issues/597)

Contributors: @aiwalter, @mloning, @fkiraly, @pabworks, @ngupta23, @ViktorKaz

## Introduction

We start by making a few conceptual points clarifying (i) the difference between model selection and model evaluation and (ii) different temporal cross-validation strategies. We then suggest possible design solutions. We conclude by highlighting a few technical challenges. 

## Overview
* clarify concepts
* agree on general design approach
* agree on next implementation steps
* techical details

## Concepts

### Model selection vs model evaluation

In model evaluation, we are interested in estimating model performance, that is, how the model is likely to perform in deployment. To estimate model performance, we typically use cross-validation. Our estimates are only reliable if are our assumptions hold in deployment. With time series data, for example, we cannot plausibly assume that our observations are i.i.d., and have to replace traditional estimation techniques such as cross-validation based on random sampling with techniques that take into account the temporal dependency structure of the data (e.g. temporal cross-validation techniques like sliding windows). 

In model selection, we are interested in selecting the best model from a predefined set of possible models, based on the best estimated model performance. So, model selection involves model evaluation, but having selected the best model, we still need to evaluate it in order to estimate its performance in deployment.

Literature references:
* [On the use of cross-validation for time series predictor evaluation](https://www.sciencedirect.com/science/article/pii/S0020025511006773?casa_token=3s0uDvJVsyUAAAAA:OSzMrqFwpjP-Rz3WKaftf8O7ZYdynoszegwgTsb-pYXAv7sRDtRbhihRr3VARAUTCyCmxjAxXqk), comparative empirical analysis of CV approaches for forecasting model evaluation
* [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf)

### Different temporal cross-validation strategies

There is a variety of different approaches to temporal cross-validation.

Sampling: how is the data split into training and test windows
1. blocked cross-validation, random subsampling with some distance between training and test windows,
1. sliding windows, re-fitting the model for each training window (request [#621](https://github.com/alan-turing-institute/sktime/issues/621)),
1. sliding windows with an initial window, using the initial window for training and subsequent windows for updating,
1. expanding windows, refitting the model for each training window.
1. expanding window with an initial window, using the initial window for training and subsequent windows for updating

It is important to document clearly which software specification implements which (statistical) strategy.

## Design
Since there is a clear difference between the concepts of model selection and evaluation, there should arguably also be a clear difference for the software API (following domain-driven design principles, more [here](https://arxiv.org/abs/2101.04938)). 

Tuning is usually represented as a composite estimator wrapping an estimator following the same interface as the component estimator (e.g. scikit-learn). 

By contrast, in model evaluation, we are interested in the performance results on all traing and test folds. 

Potential design solutions:
1. Keep `ForecastingGridSearchCV` and add model evaluation functionality
2. Factor out model evaluation (e.g. `Evaluator`) and reuse it both inside model selection and for model evaluation functionality
3. Keep only `ForecatingGridSearchCV` and use inspection on CV results for model evaluation

### 1. Keep `ForecastingGridSearchCV` and add model evaluation functions 
see e.g. `cross_val_score` as in  [`pmdarima`](https://alkaline-ml.com/pmdarima/auto_examples/model_selection/example_cross_validation.html)


```python
def evaluate(forecaster, y, fh, X=None, cv=None, strategy="refit", scoring=None):
    """Evaluate forecaster using cross-validation"""
    
    # check cv, compatibility with fh
    # check strategy, e.g. assert strategy in ("refit", "update"), compatibility with cv
    # check scoring
    
    # pre-allocate score array
    n_splits = cv.get_n_splits(y)
    scores = np.empty(n_splits)
    
    for i, (train, test) in enumerate(cv.split(y)):
        # split data
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        # split X too

        # fit and predict
        forecaster.fit(y_train, fh) # pass X too
        y_pred = forecaster.predict()

        # score
        scores[i] = scoring(y_test, y_pred)
    
    # return scores, possibly aggregate
    return scores
```


### 2. Factor out model evaluation and reuse it both for model selection and model evaluation functionality
For further modularizations, see current benchmarking module


```python
# using evaluate function from above
class ForecastingGridSearchCV:
    
    def fit(self, y, fh=None, X=None):
        # note that fh is no longer optional in fit here
        
        cv_results = np.empty(len(self.param_grid))
        
        for i, params in enumerate(self.param_grid):
            forecaster = clone(self.forecaster)
            forecaster.set_params(**params)
            scores = evaluate(forecaster, y, fh, cv=self.cv, strategy=self.strategy, scoring=self.scoring)
            cv_results[i] = np.mean(scores)
            # note we need to keep track of more than just scores, including fitted models if we do 
            # not want to refit after model selection
        
        # select best params
        return self
```

### 3. Keep only `ForecatingGridSearchCV` and use inspection on CV results for model evaluation
basically possible now

## Technical 

* Redundancy in model training in nested sliding/expanding window CV due to overlapping training windows, depending on step size, window length of both outer and inner CV, potential solution to avoid redunant training: some optimized class for nested CV that keeps track of windows and associated trained models. 
* Tuning with multi-step horizons required some data wrangling with pandas to present results for multi-windows and multi-step horizons (see [#633](https://github.com/alan-turing-institute/sktime/issues/633))


## Next steps

### Update behaviour

* passing "now" information, i.e. cutoff at which we want to make predictions
* passing new data
* information on whether we want to update parameters 

discussion on `update_params` default behaviour:

Decision A: what is the default of `update_params`?

1. True
2. False

Decision B: what is inherited default behaviour of `update` given `update_params=True`?

1. complete re-fit to entire updated scope
2. raise `NotImplementedError`

Decision C: what is inherited default behaviour of `update` given `update_params=False`?

1. move the cut-off, do not update model parameters
2. raise `NotImplementedError`

status quo: A2 B2 C1

Franz opinion: A2 B1 C1
Franz revised opinion: A1 B1 C1
Markus: A1 B2 C1
Nikhil: A1/A2 B1 C1
Kutay: A2 B1(including a warning instead of an error) C1
Peter: A1 B2 C1
Martin: A1 B1 C1
Viktor: A2 B2 C1

**decision**: A1, B1 with `NotImplementedWarning` if refit, C1

Question D: Definition of update?
1. any algorithm that claims to "improve parameters" after eating new data, also updates cut-off
2. fit(X).update(X2) must result in the same estimator parameter state (analytically identical) as fit(concat(X,X2)), in addition to 1.; also updates cut-off
3. any algorithm that claims to "improve parameters" after eating new data but not fit 
4. move cutoff and update params but might have to pay price in performance
5. 
Franz: 2 is a special case of 1, and 1 excludes common model updates in popular models like ARIMA
3 = "1 but not 2"

### Tuning
**decision**: kwarg to init of `ForecastingGridSearchCV` to specify CV strategy ("refit", "update"), with "refit" as default

### Evaluation
**decision**: implement "evaluate" function, `cross_val_score`

### Actions
* Write interface specification sheet for forecasters, ensure that `update` and `update_params` is properly discussed (documentation) -> Franz
* write algorithm specification for ForecastingGridSearchCV & ForecastingRandomSearchCV depending on CV strategy as documentation of behaviour (documetation) -> Franz
* make issue for model multiplexer (abstract for all tasks?) -> Franz
* issue for sliding window nested redundancy optimized GridSearch classes -> Franz
* create/update issue and implement change for `update`
* create/update issue and implement `evaluate` function (ping @ViktorKaz)
* create/update issue and implement change for `ForecastingGridSearchCV`
* create/update issue and implement change for `ForecastingRandomSearchCV`
* update example notebook for user journeys for model tuning and evaluation


## Multiplexer design for tuning over models
```python
class Multiplexer

    def __init__(self, forecasters, selection):
        self.forecasters = forecasters
        self.selection = selection
        
    def fit(self, ...):
   
        self.forecasters[selection].fit(...)

    def predict(self, ...):

        self.forecasters[selection].predict(...)
        
    def update(self, ...):

        self.forecasters[selection].update(...)
```

```python
f = Multiplexer([ARIMA, ExpSmoothing], "ARIMA")

grid = {"selection": ["ARIMA", "ExpSmoothing"]}
gscv = GridSearchCV(f, grid, ...)
```
