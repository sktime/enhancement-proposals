# Extension of Benchmarking API

@viktorkaz, @fkiraly

## Introduction

The current implementation of the benchmarking module in sktime suffers from several limiations.

* It is not compatible with forecasting tasks;
* It does not support custom benchmarking workflows, users are forced into a set of default choices;
* It does not support different paralelization options.

For a preliminary discussion of some initial enhancement ideas for the benchmarking module [see this issue on Github.](https://github.com/alan-turing-institute/sktime/issues/141)

## Contents
[Problem Statement](#Problem-statement)

[Description of the project](#Description-of-proposed-solution)

[Discussion and comparison of alternative solutions](#Discussion-and-comparison-of-alternative-solutions)

[Detailed description of design and implementation of proposed solution](#Detailed-description-of-design-and-implementation-of-proposed-solution)

## Problem statement

The current design of the orchestrator enforces a certian workflow for supervised classification and regression tasks which does not give full flexibility to users.

The current design of the orchestrator relies on the following main components:

1. Backend for saving and loading the data
1. Wrapper class for strategies
1. Iterator for going through strategies, datasets and cv folds
1. Procedure for fitting the startegies 
1. Procedure for making predictions
1. Evaluating the trained strategies

Currently the iterator constitutes the main loop of the orchestrator and the procedure for fittigng the strategies. A `task` object holds information about the type of problem we are trying to solve, i.e. regression, classification, forecasting, prediction. A `strategy` object serves as a wrapper for the underling estimator and holds the `fit_predict` logic.

## Description of proposed solution

Our proposal is to exapnd the task and strategy objects and add a simplified interface for setting up experiments and evaluating the results throught the use of yaml files.


The current design allows to easly add new task objects. If we want to solve the forecasting problem only we can simply use the current design to write a new forecasting task, for example:

```Python

class TSFTask(BaseTask):
"""
Time series regression task.

A task encapsulates metadata information such as the feature and target
variable
to which to fit the data to and any additional necessary instructions on
how
to fit and predict.

Parameters
----------
target : str
    The column name for the target variable to be predicted.
fh: int
    forecasting horison 
features : list of str, optional (default=None)
    The column name(s) for the feature variable. If None, every column
    apart from target will be used as a feature.

metadata : pandas.DataFrame, optional (default=None)
    Contains the metadata that the task is expected to work with.
"""

def __init__(self, target, fh, features=None, metadata=None):
    self._case = 'TSR'
    self._fh = fh
    super(TSRTask, self).__init__(target, features=features,
                                    metadata=metadata)

```
The current design of the benchmarking module allows facilitates implementing custom `fit` and `predict` methods by expanding the strategy objects. 


```Python
class TSFStrategy(BaseSupervisedLearningStrategy):
"""
Strategy for time series classification.

Parameters
----------
estimator : an estimator
    Low-level estimator used in strategy.
name : str, optional (default=None)
    Name of strategy. If None, class name of estimator is used.
"""

def __init__(self, estimator, name=None):
    self._case = "TSC"
    self._traits = {"required_estimator_type": CLASSIFIER_TYPES}
    super(TSCStrategy, self).__init__(estimator, name=name)

def fit(task,train):
    #custom logic goes here

def predict(test):
    #custom logic goes here
```


   
In order to simplify the interface, one option would be to adopt a yaml file or similar approach to defining the experiments. Examples of such an approach to setting up experiments include https://machinable.org/guide/, skll.

From the user point of view setting up a benchmarking experiment can look something like:

```yaml
benchmarking:
    datasets: "path_to_datasets"
    tasks: forecasting
    strategies: 
    TimeSeriesForest:
        n_estimator: 10
        name: time_series_forest
    RandomIntervalSpectralForest:
        n_estimator: 10
        name: time_series_forest
    cv: PresplitFilesCV
    results: "path_to_results" 
evaluation:
    metric: PairwiseMetric
    evaluation_results: `path_where_to_save`
```

The yaml file based and Python interfaces for specifying and evaluating experiments can coexist. For simpler experiements where only sktime modules and standard settings are used, users can use the yaml interface.

For more bespoke experiments, the Python interface can be used.


## User Journey

We will consider the three individual use cases below:

1. Define prediction experiment on multiple data sets, with time series classifiers

    This use case is covered in the exisiting design of the benchmarking module of sktime. Therefore, no changes are required. Please see [this notebook](https://www.sktime.org/en/latest/examples/04_benchmarking.html) for an example of how this can be achieved.

    Things to note:
    1. Users need to define a `TSCTask` object that holds the name of the column in the dataset that holds the target variable.
    1. The  `fit` method of the `TSCStrategy` object takes the task and the training data as arguments. 
    1. The `predict` method of the `TSCStrategy` object takes only the test data as an argument. The `task` object is already saved as a private attribute in the `TSCStrategy` object when the strategy was fitted. Therefore, we can use it to get only the feature columns in the training data.
    1. This procedure is equivalent to making $n$ single step ahead predictions without using the prior predictions as features for making subsequent predictions.

1. Define forecasting experiment, train/test split with sliding window predict/update/predict/etc, multiple datasets
    1. We need to define the forecasting horison `fh` which is comprised of the future time points for which we need to produce forecasts.
    1. The sliding window approach of making predictions and updating the sliding window by adding the previously predicted value is handled by family of `forecaster` strategies through reduction to regression algorithm that are implemented in the `sktime.forecasting.compose` module.


    From a user experience point of view the journey will be identical:
    
    ```Python
    # run orchestrator
    orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategies,
    cv=PresplitFilesCV(),
    results=results)

    orchestrator.fit_predict(save_fitted_strategies=False, overwrite_predictions=True)
    ```

    However, we will need to write a new forecasting task that will take the `fh` as an argument. From a user point of view this will look like this:

    ```Python
        task = TSFTask(tarkget="target", fh=fh)
    ```

    In order to make this work, we need to change very slightly the code in the orchestrator module. This will be invisible to end users.

    Currently the orchestrator performs predictions by simply calling:

    ```Python
    strategy.predict(train)
    #train are the taining samples produced by the cv algorithm
    ```

    In order to make this work with the orchestrator we need to change the above line in the orchestrator to make a case distrinction based on the type of task. For example, this can look like:

    ```Python
    if task._type == 'prediction':
        strategy.predict(train) #same as above
    if task._type == 'forecasting':
        strategy.predict(task._fh) #sktime forecasting strategies take the forecasting horison an argument to predict()
    ```

1. define forecasting experiment, multiple data sets, single train/test split (no sliding)