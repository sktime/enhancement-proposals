# Collection of Datasets

Contributors: ["jgyasu"]

## Introduction

For preliminary discussions of the proposal presented here, see issue:

* sktime/sktime#8388

## Problem statement

Currently if a user needs to reproduce a benchmarking experiment or extend it, they need to import all the dataset loaders by themselves, it is not very convenient for the end user of `sktime`. We intend to provide a simple and efficient interface, so this proposal proposes dataset collections. These collections can be created once and can be used by the user efficiently.

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