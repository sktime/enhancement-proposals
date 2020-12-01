# Extension of Benchmarking API

@viktorkaz, @fkiraly

## Introduction

The current implementation of the benchmarking module in sktime suffers from several limiations.

* It is not compatible with forecasting tasks;
* It does not support custom benchmarking workflows, users are forced into a set of default choices;
* It does not support different paralelization options.

[concise introduction to problem and overview of proposed solution]

For preliminary discussions of the proposal presented here, see issue: [links to issues/pull requests]

## Contents
[Problem Statement](#Problem statement)
[Description](#Description of proposed solution)

## Problem statement

The current design of the orchestrator enforces a certian workflow for supervised classification and regression tasks which does not give full flexibility to users.

The current design of the orchestrator relies on the following main components:

1. Backend for saving and loading the data
1. Wrapper class for strategies
1. Iterator for going through strategies, datasets and cv folds
1. Procedure for fitting the startegies 
1. Procedure for making predictions
1. Evaluating the trained strategies

Currently the iterator constitutes the main loop of the orchestrator and the procedure for fittigng the strategies and making the predictions is hard coded in the main `fit_predict`. This design choice limits the functionality of the toolbox. For example, currently the UEA benchmarking workflow is not supported. In addition to ths, the benchmarking module does not curently support forecasting as the forecaster takes the number of time steps to be forecasted as an argument to its `predict` method which is unsupported at the moment. In addition to this, different paralization options are not supported either again due to the fact that the process of fitting and predicting are hard coded.


## Description of proposed solution

Our proposal is to abstract the fit and predict logic logic of the orchestrator which will make the workflow more adaptable to varying use cases. The iterator is the other principle component of the orchestrator that remains hard coded. It currently loops through 

1. tasks and datasets
1. strategies
1. cv folds

* Question: Is this sufficient for most use cases or should we think about redesigning the iterator as well?

## Motivation

## Discussion and comparison of alternative solutions

An alternative solution would be to not abstract the logic for fitting and predicting but hard code it in the orchestrator.

As an immediate fix to the forecasting problem we could have a forecasting horison `fh` parameter passed to the `fit_predict` method of the orchestrator that will be used for making the predictions. In a similar way, we can have built-in parallelization functionality in the orchestrator. 

This approach will be easier to implement in the short term but might make the code base unwieldy if more and more functions are being added over time.

## Detailed description of design and implementation of proposed solution 

The current implementation of the orchestrator in pseudocode can be found below:

```Python
class Orchestrator:
    def __init__(self, tasks, datasets, strategies, cv, results):
        # the task, datasets and strategies parameters are passed as lists
        # cv stands for cross validation iterator
        # results is a custom class for persisting the strategies and their predictions

    def fit_predict(self,
        overwrite_predictions=False,
        predict_on_train=False,
        save_fitted_strategies=True,
        overwrite_fitted_strategies=False,
        verbose=False)

        #_iter is a private iterator method 
        for task, dataset, data, strategy, cv_fold, train_idx, test_idx in self._iter():
            train = data.iloc[train_idx]
            strategy.fit(task, train)
            results.save_fitted_strategy()

            test = data.iloc[test_idx]
            strategy.predict(test)
            results.save_predictions()
            
```
The proposed implementation is:

```Python
class Orchestrator:
    def __init__(self, tasks, datasets, strategies, cv, results, fit_logic, predict_logic):
        # the task, datasets and strategies parameters are passed as lists
        # cv stands for cross validation iterator
        # results is a custom class for persisting the strategies and their predictions

        # fit_logic and predict logic are the two new classes that we propose to add to the design.

    def fit_predict(self,
        overwrite_predictions=False,
        predict_on_train=False,
        save_fitted_strategies=True,
        overwrite_fitted_strategies=False,
        verbose=False)

        #_iter is a private iterator method 
        for task, dataset, data, strategy, cv_fold, train_idx, test_idx in self._iter():
            train = data.iloc[train_idx]
            strategy.fit(task, train)
            fit_logic.fit(strategy, task, train)
            results.save_fitted_strategy()

            predict_logic.predict(strategy, train) 
            results.save_predictions()
            
```

The difference is that the fit/predict logic will be specified by the user. This should solve the problem of incompatibility of the orchestrator with forecasting tasks as users will be able to specify the forecating horizon `fh` in the predict_logic object.