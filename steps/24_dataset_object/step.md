# Dataset object

Contributors: ["felipeangelimvieira", "fkiraly"]

## Introduction

This STEP is an enhacement proposal concerning the creation of Dataset objects.

For preliminary discussions of the proposal presented here, see issues and PRs: 

* sktime/sktime#4332
* sktime/sktime#4333
* sktime/sktime#7398

## Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Description of Proposed Solution](#description-of-proposed-solution)  
   - [__init__ and load](#__init__-and-load)  
   - [Train and Test Splits](#train-and-test-splits)  
   - [Base Classes](#base-classes)  
   - [Tags](#tags)  
     - [Base Dataset Tags](#base-dataset-tags)  
     - [Classification Tags](#classification-tags)  
     - [Regression Tags](#regression-tags)  
     - [Forecasting Tags](#forecasting-tags)  
4. [Motivation](#motivation)
5. [Discussion and Comparison of Alternative Solutions](#discussion-and-comparison-of-alternative-solutions)
6. [Detailed Description of Design and Implementation](#detailed-description-of-design-and-implementation-of-proposed-solution)  

## Problem statement

Currently, datasets in sktime are loaded by the usage of functions, and having a dataset object would provide many benefits for documentation purposes. For example, in the case of estimators, there's an estimator overview page that allows users to filter and find estimators with certain characteristics for classification, forecasting and regression.
Having dataset objects would allow the retrieval of problems with characteristics that fit the user needs by the usage of the tag system.

In addition to this, an unified interface that also identifies the characteristics of each dataset can be useful for benchmarking purposes. If a researcher is investigating in which circunstances a model performs well, the metadata (tags) of the datasets could be useful to analize and understand these factors.

The dataset object interface could also be used for other scenarious beyond timeseries forecasting, classification and regression. For example, the forecasting dataset object could be used for Marketing Mix Modeling datasets (see felipeangelimvieira/prophetverse#101)

## Description of proposed solution

### `__init__` and `load`

Dataset objects should have all the parameters that identify the content of it defined in `__init__`, similar to what hyperparameters represent for estimators. The `load(*args)` method is used to obtain the sets. This method receives strings as positional args, such as `load("X", "y", "X_train", "y_train", "X_test", "y_test")` and returns the respective sets. Hence, users can define which set should be retrieved by passing its name to `load`. 

### Train and test splits

Many public datasets have default train-tests splits, and could also potentially have default cross-validation splits. Others, however, do not provide such sets. The following behaviour will be used to handle this heterogeneity:

* Calling load with "X_train", "X_test", "y_train" and "y_test" should raise `InvalidSetException` if the dataset does not have a pre-defined split. The sktime interface will not provide a custom split if the original dataset does not define one.
* In the case of existence of a cross validation split, "cv" keyword can be used, and a generator with the CV folds will be returned. This generator will have length equal to the number of folds in the CV split, and contain the "X_train", "X_test", "y_train" and "y_test" data for each split. Datasets with a single train-test split should return generators of length one.
* If the dataset has a CV split that is not a simple train-test split, then calling `"X_train"`, `"X_test"`, `"y_train"` and `"y_test"` should also raise `InvalidSetException`, and an informative error message recommending the user to call `"cv"`.

### Base classes

One BaseDataset class, plus specific classes for each time series problem (e.g., BaseForecastingDataset, BaseClassificationDataset).

### Tags

#### Base dataset tags

- **name**: str
  This is a shorthand and unique ID, similar as in the huggingface datasets namespace. Must be lower snake case.
- **n_splits**: int, default=0
  The number of CV splits. Zero represents no default cross-validation or simple train-test split. One represents
  a simple train-test split.

#### Classification Tags

- **is_univariate**: bool, default=True  
  Indicates whether the dataset is univariate. In the case of a classification dataset, this refers to the dimensionality of the X dataframe (i.e., how many series are associated with each class label).
- **n_instances**: int, default=None  
  The number of instances in the dataset. Should be equal to `y.shape[0]`.
- **n_instances_train**: int, default=None  
  The number of instances in the training set. Should be equal to `y_train.shape[0]`.
- **n_instances_test**: int, default=None  
  The number of instances in the test set. Should be equal to `y_test.shape[0]`.
- **n_classes**: int, default=2  
  The number of classes in the dataset.

#### Regression Tags

- **is_univariate**: bool, default=True  
  Indicates whether the dataset is univariate. In the case of a regression dataset, this refers to the dimensionality of the X dataframe (i.e., how many series are associated with each label).
- **n_instances**: int, default=None  
  The number of instances in the dataset.
- **n_instances_train**: int, default=None  
  The number of instances in the training set.
- **n_instances_test**: int, default=None  
  The number of instances in the test set.

#### Forecasting Tags

- **is_univariate**: bool, default=True  
  Indicates whether the dataset is univariate. In the case of a forecasting dataset, this refers to the dimensionality of the `y` dataframe.
- **is_equally_spaced**: bool, default=True  
  Indicates whether all observations in the dataset are equally spaced.
- **has_nans**: bool, default=False  
  True if the dataset contains NaN values, False otherwise.
- **has_exogenous**: bool, default=False  
  True if the dataset contains exogenous variables, False otherwise.
- **n_instances**: int, default=None  
  The number of instances in the dataset. Should be equal to `y.shape[0]`.
- **n_instances_train**: int, default=None  
  The number of instances in the training set (None if the dataset does not have a train/test split). Should be equal to `y_train.shape[0]`.
- **n_instances_test**: int, default=None  
  The number of instances in the test set (None if the dataset does not have a train/test split). Should be equal to `y_test.shape[0]`.
- **n_timepoints**: int, default=None  
  The number of time points in the dataset, per series. If the dataset contains series of different lengths, this should be equal to the maximum length found in the dataset.
- **n_timepoints_train**: int, default=None  
  The number of time points in the training set, per series. If the dataset contains series of different lengths, this should be equal to the maximum length found in the training set.
- **n_timepoints_test**: int, default=None  
  The number of time points in the test set, per series. If the dataset contains series of different lengths, this should be equal to the maximum length found in the test set.
- **frequency**: str, default=None  
  The frequency of the time series in the dataset. Can be an integer if the frequency is not associated with a time unit.
- **n_dimensions**: int, default=1  
  The number of dimensions in the dataset (i.e., the number of columns in the `y` dataframe).
- **n_panels**: int, default=1  
  The number of panels in the dataset (i.e., the number of unique time series in the dataset).
- **n_hierarchy_levels**: int, default=0  
  The number of hierarchy levels in the dataset (i.e., the number of index levels in the `y` dataframe, excluding the time index).

## Motivation

## Discussion and comparison of alternative solutions

Many libraries provide interfaces for retrieving datasets. Here are some examples:

* HuggingFace's datasets: the dataset objects are usually retrieved by `load_dataset("dataset_name", split="validation")`. They are objects that, in addition to containing the content of the set, also can perform preprocessing steps, such as removing columns and casting columns to a specific type. Methods such as `set_format` convert the dataset to other formats such as Pytorch. A DatasetInfo object contains the metadata, such as citation and name of the set. They also provide `cache_files` attribute and `cleanup_cache_files` to help cleaning up files that were loaded.
* Sklearn's loaders: in sklearn, the dataset interface is composed of loading functions that return a Bunch object, which is similar to dictionaries but allows access it contents as they were attributes. Information of the dataset is stored in the docstring, that appears on the documentation.


## Detailed description of design and implementation of proposed solution 

### Base class

```python
class BaseDataset(BaseObject):
    """Base class for datasets."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "dataset",  # type of object
        "name" :  None, # The dataset unique name
        "python_dependencies" : None, # python dependencies required to load the dataset
        "python_version" : None, # python version required to load the dataset
        "n_splits" : 0, # Number of cross-validation splits, if any.
    }

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

    def load(self, *args): 
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            available/valid strings are provided by the concrete classes
            the expectation is that this docstring is replaced with the details

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """

        if len(args) == 0:
            args = ("X", "y")
        self._check_args(*args)

        return self._load(*args)

    def _check_args(self, *args):
        for arg in args:
            if arg not in self.available_sets:
                raise InvalidSetError(arg, self.available_sets)

    @property
    def available_sets(self):
        """
        Return a list of available sets.
        
        Returns
        -------
        list of str
            List of available sets.
        """
        sets = ["X", "y"]
        n_splits = self.get_tag("n_splits")
        if n_splits == 1:
            sets.extend(["X_train", "y_train", "X_test", "y_test"])
        elif n_splits > 1:
            sets.append("cv")
        return sets
    
    def cache_files_directory(self):
        """
        Get the directory where cache files are stored.
        
        Returns
        -------
        Path
            Directory where cache files are stored
        """
        
        dataset_name = self.get_tag("name")
        return Path(__file__).parent.parent / Path("data") / dataset_name
    
    def cleanup_cache_files(self):
        """Cleanup cache files from the cache directory."""
        
        cache_directory = self.cache_files_directory()
        if cache_directory.exists():
            shutil.rmtree(cache_directory)


class InvalidSetError(Exception):
    """Exception raised for invalid set names."""

    def __init__(self, set_name, valid_set_names):
        self.set_name = set_name
        self.valid_set_names = valid_set_names

    def __str__(self):
        return (
            f"Invalid set name: {self.set_name}. "
            f"Valid set names are: {self.valid_set_names}."
        )

```

### Mixin to wrap current loaders

To handle current loader funcs, the following mixin is implemented:

```python
class _DatasetFromLoaderMixin:
    loader_func = None

    def _encode_args(self, code):
        kwargs = {}
        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TEST"
        else:
            split = None

        # Check if loader_func has split and return_type parameters
        # else set kwargs = {}
        loader = self.get_loader_func()
        loader_func_params = signature(loader).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        if (
            "test" in code.lower() or "train" in code.lower()
        ) and "split" not in loader_func_params:
            raise ValueError(
                "This dataset loader does not have a train/test split"
                + "Load the full dataset instead."
            )

        if "split" in loader_func_params:
            kwargs["split"] = split

        for init_param_name, init_param_value in init_param_values.items():
            if init_param_name in loader_func_params:
                kwargs[init_param_name] = init_param_value

        return kwargs

    def get_loader_func(self):
        # calls class variable loader_func, if available, or dynamic (object) variable
        # we need to call type since we store func as a class attribute
        if hasattr(type(self), "loader_func") and isfunction(type(self).loader_func):
            loader = type(self).loader_func
        else:
            loader = self.loader_func
        return loader

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            "X": full panel data set of instnaces to classify
            "y": full set of class labels
            "X_train": training instances only, for fixed single split
            "y_train": training labels only, for fixed single split
            "X_test": test instances only, for fixed single split
            "y_test": test labels only, for fixed single split

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        if len(args) == 0:
            args = ("X", "y")

        cache = {}

        if "X" in args or "y" in args:
            X, y = self._load_dataset(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y
        if "X_train" in args or "y_train" in args:
            X, y = self._load_dataset(**self._encode_args("X_train"))
            cache["X_train"] = X
            cache["y_train"] = y
        if "X_test" in args or "y_test" in args:
            X, y = self._load_dataset(**self._encode_args("X_test"))
            cache["X_test"] = X
            cache["y_test"] = y

        res = [cache[key] for key in args]

        # Returns a single element if there is only one element in the list
        if len(res) == 1:
            res = res[0]
        # Else, returns a tuple
        else:
            res = tuple(res)

        return res

    def _load_dataset(self, **kwargs):
        """
        Call loader function and return dataset dataframes.

        This method is intended to be overridden by child classes if the order of `X`
        and `y` in the loader output is different from the default order (`X` first, `y`
        second).
        """
        loader_func = self.get_loader_func()
        return loader_func(**kwargs)
```

