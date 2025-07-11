# Advanced Benchmarking

Contributors: ["jgyasu"]

## Introduction

For preliminary discussions of the proposal presented here, see issue:

* sktime/sktime#8388

## Problem statement

Users would like to run benchmarking experiments on scale, for statistical performance of estimators, similar to common academic or industrial experiments. This enhancement proposal is about enabling common experimental setups, as described below.

Features and designs are derived from this

## Use cases

* use case 1: reproducing a known set of benchmarks, for instance, M4, M5, TSC
    * this is 1:1 reproduction of a previous experiment
* use case 2: running a typical "academic benchmark experiment": 1 new estimator, plus a common selection of prior art benchmarks and naive benchmarks are run on a common selection of datasets
    * this includes the case where we add a single estimator to a known set of benchmarks
    * there are two sub-cases: 2a re-running the entire experiment; 2b only retrieving results, and runnong only the estimator in addition
* use case 3: running a common set of algorithms on a new dataset (single) or collection thereof. Typical "industry" use case. data is new, but estimators are known sets.
    * this may be repeated regularly as data set grows. (refit, new start, or update)

## Requirements - for this enhancement phase

* use cases as above can be handled easily, intuitive sklearn-like specification syntax (all use cases)
    * "run the TSC bake-off" should not be too many lines!
* user can obtain and use easily well-known sets of benchmark estimators (all use cases)
* user can obtain and use easily well-known sets of benchmark datasets (use case 1 and 2)
* user can obtain and use easily historical configurations of benchmarks, e.g., combination of metrics, cv, and estimators, datasets (use case 1, 2).
    * possibly also, collections of metrics?
* resumability: if experiment breaks, it can be resumed (all use cases)
* sharing of cached benchmarking results: if user does not want to rerun full "standard benchmarks" results can be obtained (2b)
* easily deployable on a cluster - typical benchmark runs take a long time
    * parked until next enhancement

## Conceptual model

MAYBE NOT COMPLETE; PLEASE WORK ON THIS

### relevant objects

#### "primitives" relevant

estimator

Example 1: `HIVECOTEv1()`
Example 2: `ChronosForecaster()`
Example 3: `TransformedTargetForecaster([Differencer(), ExponentialSmoothing()])`

dataset object in-memory, containers

Example 1: return of `load_airline()`
Example 2: returns of `load_arrowhead()`
Example 3: instance `ArrowHead()` - class design
Question: maybe want to introduce additional containers specifically for bm

* dataset file on hard-drive or cloud

Example 1: a collection of `csv` for time series classification
Example 2: the M5 dataset files as downloaded from Monash repo
Example 3: bunch of `tsf`
Question: we may want to host common datasets on HuggingFace - need to think about
how to represent these

* re-sampling specifier
* metric

#### specification of benchmark experiment

is made up of:
* selection of estimators
* selection of datasets
* re-sampling and cv
* selection of metrics
* this constitutes a single benchmark experiment *specification*
* different estimator types will imply different object types above!

"always give three examples"

Example 1: "specification of TSC bake-off"
* list of estimators from paper
* UCR repository datasets state 2018
* etc

#### experiment results
* metric evaluate per estimator, dataset, re-sample fold
* different levels of aggregation of these
    * e.g., mean over all datasets

#### "collection" of objects

* of estimators, etc

Example 1: all the estimators in the TSC bake-off (original)
Example 2: the M4 datasets
Example 3: "common choice of point forecasting metrics", e.g., MAPE, RMSE, MAE




### relevant actions

"specify": experiment specification is constructed from different selections

"execute experiment": benchmark experiment runs

"resume experiment": benchmark experiment restart after abort

## User journey designs

### Use case 1

```python
stuff


```


### Use case 2a

### Use case 2b


### Use case 3



### Resuming experiment after it breaks in middle



## Description of proposed solution

Please see [here](#detailed-description-of-design-and-implementation-of-proposed-solution)

## Motivation

To simplify reproducibility of a benchmark experiment for the user

## Discussion and comparison of alternative solutions

To do

## Detailed description of design and implementation of proposed solution 

This is how I expect the benchmarking interface to be, so that the end user does not have to manually import all the datasets when they want to run the benchmark, they can just use a collection instead

### Expected Benchmarking Interface

```python
from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.dataset_collection import DatasetCollection # To implement
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


benchmark = ClassificationBenchmark()

benchmark.add_estimator(
    estimator=DummyClassifier(),
    estimator_id="DummyClassifier",
)

cv_splitter = KFold(n_splits=3)
scorers = [accuracy_score]

# Instantiate a dataset collection, each collection will have a different class following strategy pattern, e.g., collection of datasets that were used in TSC Bakeoff paper, returns a list of datasets loaded as sktime dataset containers
datasets = DatasetCollection("TSCBakeOff")

# If we want an additional dataset to extend the study, let's assume ArrowHead was not present in the initial study but we want to add it to our collection for this particular study
from sktime.datasets import load_arrow_head
datasets.add(load_arrow_head)

for dataset in datasets:
    benchmark.add_task(
        dataset,
        cv_splitter,
        scorers,
    )

results_df = benchmark.run("./classification_results.csv")
results_df.T
```
### Proposed Classes

#### `BaseDatasetCollection` Class

```python
class BaseDatasetCollection(BaseObject):
        """Base class for dataset collection strategies using sktime's BaseObject."""
    
    def __init__(self):
        super().__init__()
    
    def get_datasets(self):
        """Return a list of dataset loading functions."""
        raise NotImplementedError("Subclasses must implement get_datasets()")
    
    def get_collection_name(self):
        """Return the name of the collection."""
        raise NotImplementedError("Subclasses must implement get_collection_name()")
```

#### `TSCBakeOffCollection` Example
```python
class TSCBakeOffCollection(BaseDatasetCollection):
    """Collection of TSC Bake-off datasets."""
    
    def __init__(self, subset=None):
        """
        Initialize TSC Bake-off collection.
        """
        self.subset = subset
        super().__init__()
    
    def get_collection_name(self):
        return "TSC Bake-off Collection"
    
    def get_datasets(self):
        """Return a list of dataset loaders from the TSC Bake-off paper."""
        from sktime.datasets import load_UEA_UCR_dataset

        dataset_names = [] # Contributor writing the collection will have to manually add the datasets used in the experiment
        all_datasets = []
        for name in dataset_names:
            # If UEA UCR archive has all the datasets then load from there or use additional loaders, even custom ones
            dataset = load_UEA_UCR_dataset(name=name)
            all_datasets.append(dataset)
        
        return all_datasets
```

```python

class DatasetCollection(BaseObject):
    """
    A collection of datasets following the strategy pattern.
    """

    _collections = {
        "TSCBakeOff": TSCBakeOffCollection,
        # We need to add the dataset collections here
    }

    def __init__(self, collection_type):
        super().__init__()
        self.collection_type = collection_type
        self._additional_datasets = []
        self._initialize_collection()

    def _initialize_collection(self):
        if self.collection_type not in self._collections:
            raise ValueError(f"Unknown collection type: {self.collection_type}. "
                             f"Available: {list(self._collections.keys())}")
        collection_cls = self._collections[self.collection_type]
        self._collection = collection_cls()

    def add(self, dataset_loader):
        try:
            dataset = dataset_loader()
            self._additional_datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Failed to add additional dataset: {e}")
        return self

    def __len__(self):
        return len(self._collection.get_datasets()) + len(self._additional_datasets)

    def get_collection_info(self):
        return {
            "collection_name": self._collection.get_collection_name(),
            "total_datasets": len(self),
        }
```