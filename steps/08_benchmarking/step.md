# Extension of Benchmarking API

@viktorkaz, @fkiraly

## Introduction

The current implementation of the benchmarking module in sktime suffers from several limiations.

* It is not compatible with forecasting tasks;
* It does not support custom benchmarking workflows, users are forced into a set of default choices;
* It does not support different paralelization options.

For preliminary discussions of the proposal presented here, [see this.](https://github.com/alan-turing-institute/sktime/issues/141)

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

Currently the iterator constitutes the main loop of the orchestrator and the procedure for fittigng the strategies and making the predictions is hard coded in the main `fit_predict`. This design choice limits the functionality of the toolbox. For example, currently the UEA benchmarking workflow is not supported. In addition to ths, the benchmarking module does not curently support forecasting as the forecaster takes the number of time steps to be forecasted as an argument to its `predict` method which is unsupported at the moment. In addition to this, different paralization options are not supported either. 

## Description of proposed solution

Our proposal is to abstract the fit and predict logic logic of the orchestrator which will make the workflow more adaptable to varying use cases. The iterator is the other principle component of the orchestrator that remains hard coded. It currently loops through 

1. tasks and datasets
1. strategies
1. cv folds

* Question: Is this sufficient for most use cases or should we think about redesigning the iterator as well?

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

The `fit_logic` function can look something like:

```Python
def fit_logic():
    # this is a pseudocode example for a bagging algorithm pararelized manually over 4 CPU cores
    self.model1 = self.strategy.fit(self.x_train, num_treads=1)
    self.model2 = self.strategy.fit(self.x_train, num_treads=1)
    self.model3 = self.strategy.fit(self.x_train, num_treads=1)
    self.model4 = self.strategy.fit(self.x_train, num_treads=1)

    
```

The above example is only for illustrative purposes, there are libraries that can pararelize trainng over the available CPU cores more efficiently. However, this shows how the fitting logic can be made more flexible. With this design users can create custom fitting pipelines, be able to pararelize the training over cpomputing clusters, use third party liblaries, etc.

Below is an example of a `predict_logic` method that should be compatible with sktime's forceasting framework:

```Python
def predict_logic(fh):
    prediction1 = self.model1.predict(x_test)
    prediction2 = self.model2.predict(x_test) 
    prediction3 = self.model3.predict(x_test) 
    prediction4 = self.model4.predict(x_test) 

    return (prediction1+prediction2+prediction3+prediction4) / 4
```

## User journey

The below code snippet shows the current implementation and the proposed change.
```Python
# Create individual pointers to dataset on the disk
datasets = [
    UEADataset(path=DATA_PATH, name="ArrowHead"),
    UEADataset(path=DATA_PATH, name="ItalyPowerDemand"),
]
#specify learning task
tasks = [TSCTask(target="target") for _ in range(len(datasets))]
#specify learning strategies
strategies = [
    TSCStrategy(TimeSeriesForest(n_estimators=10), name="tsf"),
    TSCStrategy(RandomIntervalSpectralForest(n_estimators=10), name="rise"),
]

# Specify results object which manages the output of the benchmarking
results = HDDResults(path=RESULTS_PATH)


# -------------------proposed change----------------
def fit_logic():
    pass
def predict_logic():
    pass
# -------------------proposed change----------------
# run orchestrator
orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategies,
    cv=PresplitFilesCV(),
    results=results,
    fit_logic=fit_logic, #-------proposed change
    predict_logic=predict_logic #------ proposed change
)
orchestrator.fit_predict(save_fitted_strategies=False, overwrite_predictions=True)

#evaluate results
evaluator = Evaluator(results)
metric = PairwiseMetric(func=accuracy_score, name="accuracy")
metrics_by_strategy = evaluator.evaluate(metric=metric)
metrics_by_strategy.head()
```

## Discussion and comparison of alternative solutions
1. Option 1 - Expand the task and strategy objects
   
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
    If this approach is assumed, we only need to change one line in the orchestrator where the predictions are made.


    ```Python
    def fit_predict():
        #current implementation
        ......

        y_pred = strategy.predict(test)
        .......
        #needs to change to
        y_pred = strategy.predict(task, test)
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
    
    def predict(task, test):
        #custom logic goes here
    ```

1. Option 2 - hard code logic in orchestrator
    
    An alternative solution would be to hard code the logic for fittig and predicting in the orchestrator.

    As an immediate fix to the forecasting problem we could have a forecasting horison `fh` parameter passed to the `fit_predict` method of the orchestrator that will be used for making the predictions. In a similar way, we can have built-in parallelization functionality in the orchestrator. 

    This approach will be easier to implement in the short term but might make the code base unwieldy if more and more functions are being added over time.

1. Option 3 - Add option to specify experiments through yaml files.
   
   One option would be to adopt a yaml file or similar approach to defining the experiments. Examples of such an approach to setting up experiments include https://machinable.org/guide/, skll.

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
   

   

   

   
   

