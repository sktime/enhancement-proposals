# STEP02: Adding support for 3d np.array/awkward-array as main data containers

Contributors: @fkiraly, @matteogales, @mloning, @big-o, @prockenschaub

For preliminary discussions, see issue [#15](https://github.com/alan-turing-institute/sktime/issues/15).

For experimental/prototyping work, see the [data-container](https://github.com/alan-turing-institute/sktime/tree/data-container) branch.

## Contents
[TOC]

## Summary

### Problem statement
Our current data container for the series-as-features setting is a nested pd.DataFrame. The series-as-features setting includes time series classification, regression and eventually clustering and requires panel data. While very flexible, the nested pd.DataFrame is slow and continues to cause a lot of bugs and confusion on the side of users and contributors. 

### Proposal
In this enhancement proposal, I propose the following:
1. Add support for 3d numpy arrays with shape = (n_instances, n_variables, n_timepoints) for rectangular/time-homogeneous data,
2. Make 3d numpy arrays the default data container, using the nested pd.DataFrame to handle few cases of time-heterogeneous data,
3. Replace nested pd.DataFrame with awkward-array for time-heterogeneous data; deprecate and remove the nested pd.DataFrame. 

## Background
### Generative time series setting
Time series data can arise in different settings and comes in many different forms and shapes. For machine learning with time series, it is important to understand the different forms such data may take. The data can come in the form of a single (or univariate) time series; but in many applications, multiple time series are observed. It is crucial to distinguish the two fundamentally different ways in which this may happen:
* Multivariate time series data, where two or more variables are observed repeatedly over time, with variables representing different kinds of measurements, for a single instance (or experimental unit);
* Panel data, sometimes also called longitudinal data, where multiple independent instances of the same variables are observed repeatedly over tiem. 

In the multivariate case, it is implausible to assume the different univariate component time series to be i.i.d. By contrast, in the panel data case, the i.i.d. assumption with respect to the different instances is plausible. However, time series observations within a given instance are still likely to depend on previous observations. 

In addition, panel data may be multivariate, which corresponds to i.i.d. instances of multivariate time series. In this case, the different instances are i.i.d., but the univariate component series within an instance are not.

For more details, see our [paper](https://arxiv.org/pdf/1909.07872.pdf).

### Representing time series data
This richness of generative scenarios is mirrored in a richness of data representations. 

Common representations include the following: 

| Representation | Description | 
|---|---|
| Long | 2d table, rows represent time points, additional columns for instance, variable, value (e.g. a table in a relational database) |
| Wide | 2d table, rows represent instances, columns respresent time points |
| Nested | json-like dictionary for nesting instances, variables and time points |

Also see this [overview](https://github.com/MaxBenChrist/awesome_time_series_in_python/blob/master/standardize_time_series_formats.md) by one of the tsfresh developers. 

### No consensus data container

There are various data containers in Python, with Pandas and NumPy being the most common ones. However, most of them have important shortcomings for the purpose of modelling and evaluating machine learning workflows with time series. 

There seems to be no consensus or established standard for representing time series. 

This enhancement proposal focuses on in-memory data representations. But the same seems to apply to on-disk representations and file formats. 

### Importance of data containers
Data containers (in-memory representation) are fundamental in any toolbox: they determine how data can be stored, accessed and manipulated. 

## Requirements 
For the purpose of specifying and applying machine learning workflows to panel data, key requirements for data containers are as follows:

### Representation
* **Univariate time series**
* **Multivariate time series**
* **Panel data** (multiple i.i.d. instances of time series), 
* **Date/time indexing** (time series vs sequences)
* **Time-heterogeneous indices** where time points vary across instances and/or variables (e.g. unequal length series, unequally spaced time points, variables with different sampling frequencies)
* **Type-heterogeneous data** (e.g. time series data combined with scalar data, float, integer and string values)
* **Meta-data** for variable names (having a data container with meta-data allows to update the meta-data when applying data transformations, e.g. in pipelines),

### Efficiency
* **Computational efficiency** to be practically useful

### Usability
* **Intuitive API** for indexing, slicing, and data manipulations
* **Date/time index operations** up-sampling/down-sampling with aggregation/interpolation, sliding windows, sorting, conditional subsetting using date/times, slicing, extraction of calendar information (weekdays, holidays, etc)
* **Ecosystem compatibility** for re-using much of scikit-learn's functionality
* **Notational alignment** Ideally, the data container should also align with common mathematical notation and be intuitive to use (e.g. following standard mathematical notation with X being feature matrix with rows representing i.i.d. instances and columns representing variables and y being the target vector to predict).

To our knowledge, there is no library that is flexible enough to meet all of the requirements. 

## Current data container
While pandas data containers are designed to store only primitive data types in their cells, technically cells can store arbitrary types. This gives us a very flexible representation of time series data. It enables us to represent time heterogenous series, i.e. series which do not share a common time index across instances and/or variables. At the same time, we can still make use of most of pandas' functionality and familiar interface. 

This is inspired by [xpandas](https://github.com/alan-turing-institute/xpandas). 

However, there are a number of disadvantages:
* unintuitive to most users who are familiar with the intended usage of pandas,
* not very efficient,
* cannot slice/operate on temporal dimension

## Potential solutions
For a more complete list of data containers we considered, please see our [wiki entry on related software](https://github.com/alan-turing-institute/sktime/wiki/Related-software#time-series-data-containers).

#### Data loader abstraction
* similar to deep-learning libraries 
* create a common layer of abstractions for different representations 
* create a common interface point for different kinds of data (e.g. temporal and cross-sectional data, categoricals)

#### Native Python object-oriented solution
* similar to Java-based solution in TSML
* maintenance burden
* naive implementations are too slow

#### Pandas DataFrame with column multi-index 
* only 2d, so wide format requires column multi-index to handle multivariate panel data
* gives limited ability to slice temporal dimension, but essentially still nested
* assumes shared time index across instances, but not across columns

#### 3d NumPy arrays
For the series-as-features setting, we could use 3d NumPy arrays with shape = `[n_instances, n_variables, n_timepoints]`.

##### Advantages
* Familiarity: Used throughout the scientific computing community in Python, familiar to most, and most intuitive direct extension of tabular scikit-learn setting, 
* Efficiency: highly optimised in terms of CPU and memory usage (vectorisation, threading, etc.) mostly due to restricted computation domain (machine typed arrays rather than arbitrary objects),
* Reliability: well maintained, thoroughly unit and field tested,
* Usability: very powerful and intuitive interface for arrays (slicing in the temporal dimension, etc.),
* Maintenance: no extra maintenance burden on our side,
* Compatibility: used by scikit-learn and tslearn, ensure easy compatibility with PyData ecosystem and perhaps later convergence of different time series related toolboxes,

##### Disadvantages
* only equal-length data (across instances and columns/variables)
* does not handle time indexes/timestamps

##### Comments
* all of our functionality assumes equal-length series at the moment, so the disadvantage isn't really a disadvantage, at least for now ...
* none of our series-as-features functionality makes use of time indices at the moment, generally time indices don't seem very relevant in the prediction algorithms, only in preprocessing, but I believe we have been and should continue to focus on algorithms, model composition and model evaluation
* perhaps it's better to adopt Numpy arrays, and only rely on additional data container for unequal length cases
* for open-source toolboxes, it's important to focus on easy problems to solve, implementing or extending a data container seems a lot harder than building a scikit-learn-like toolbox for ML with time series
* potential compromise: allow multiple data containers, using common input validation function for conversion, use 3d NumPy array as standard data container where possible, use extended Pandas data frame (or other ragged array) for unequal-length cases

#### xarray
* depends on xarray developing support for ragged arrays
* potential collaboration between awkward array and xarray 

#### Awkward Array
* ragged arrays
* support for keeping track of time index, however no support for uo
* how could we support date/time indexing in awkward array ?
* what does pandas add when we use awkward array inside an ExtensionArray in a pandas DataFrame?

#### Extending Pandas
Initially, we decided to use [pandas](https://pandas.pydata.org), because it 
* one of the most developed data containers in Python,
* familiar to most,
* handles time indexes,  
* keeps track of meta-data, and
* handles type-heterogenous data.

However, in its intended usage, pandas can store only primitive (or scalar) values. Consequently, panel data with multiple i.i.d. instances are usually represented in either the wide or the long format, both having important drawbacks as discussed above. There are a number of additional reasons to extend pandas:
* Typing columns with new types for time series,
* Typing of columns based on scitypes to provide counterpart to task objects and estimator compatibility checks (for more details, see e.g. [MLJ's scitype module](https://github.com/alan-turing-institute/ScientificTypes.jl)),
* Single input checks which create flags in the data container, rather than running the checks every time the data is passed to an estimator (e.g. checking data against column types, presence of missing values, unequal length series, shared time index, univariate or multivariate, presence of additional time-constant variables),
* Separation of additional meta-data, e.g. flags from input checks and scitypes,
* More efficient slicing/selecting in time index/dimension.

##### ExtensionArrays
pandas comes with its own extension class for dataframe columns, called [ExtensionArrays](https://pandas.pydata.org/pandas-docs/stable/development/extending.html). ExtensionArrays allow to store data in arbitraries ways, subject to only some requirements so that it still is compatible with the dataframe structure. 

The idea is to represent time series as a new column type, which internally stores time series as 2d numpy arrays for equal length series. For unequal length series, we will consider [awkward-array](https://github.com/scikit-hep/awkward-array). 

##### Implementation

The preliminary ExtensionArray solution primarily consists of three classes:

1. *TimeArray*: A subclass of ExtensionArray which defines how data and indices are stored under the hood. This class also provides all compatibility with pandas. 
2. *TimeSeries*: A subclass of pandas Series that implements additional functionality if the underlying values are a TimeArray (e.g. by providing a function to tabularise or slice along the time axis). Allows for the storage and easy access to meta information.
3. *TimeFrame*: A subclass of pandas DataFrame that implements additional functionality if any of its columns contains a TimeArray. Allows for the storage and easy access to meta information.

The naming so far was more or less chosen by simply putting Time in front of the superclass names. This might become confusing, especially for the TimeSeries which does not contain a single time series but:
* if the underlying value object is a TimeArray, an entire list of time series
* if the undelrying value object is **not** a TimeArray, a list of scalar values just like a vanilla pandas Series

It might make sense to make this distinction clearer in the naming. 

##### Types of data/index combinations to consider
1. *All stored series are equal length*
    1. Sequence: No explicit index is needed, data of the form NxT is simply indexed by its position in each row, i.e. the implicit index ranges from 0, …, T-1
    2. Series with equal index: Each data point has an explicit index (e.g. integer - discrete time, floating point number - continuous time, date time object – continuous time); the index is shared across different series (i.e. rows) and can be represented by a single vector with T elements
    3. Series with unequal indices: Each data point has an explicit index; the index may differ from row to row and must be represented by a matrix of the form NxT

Data in each of these cases can be represented within the ExtensionArray as numpy arrays. The time index could be stored in the same way, i.e. an array of the same size as data (this is the current implementation). Scenarios i. and ii., however, also allow for a more memory efficient storage of the index by omitting the index completely (i.) or by just storing one row of the index (ii.). If we want to allow for this, we will have to make sure that the representation of the index stays consistent when changes to the TimeArray result in the move from one scenario to the other. In addition, care must be taken when slicing with unequal indices (point iii.), since it might result in an unequal length of the data (see next section).

2. *Series can be of arbitrary/unequal length*
    1. Sequence: As in 1.i. no explicit index is needed, ragged data of the form Nx? is simply indexed by its position in each row, i.e. the implicit index of row n ranges from 0, …, length(n)
    2.	Series with (unequal) row indices: Each data point has an explicit index; the index may differ from row to row and must be represented by a ragged matrix of the same form as data

Data and index can not be represented by numpy arrays, and a different solution such as awkwardarray must be used to represent the information within the ExtensionArray.

##### Time indices
Time index are represented by numpy arrays (or awkwardarrays in the case of unequal lengths) and can therefore in theory take any data type that can be stored in a numpy array. However, the current implementation of many internal functions assumes for simplicity that the index is integer or float. Making this assumptin is mostly relevant for missing data operations, where checking for missing data differs for different data types. In order to truly allow for other index types, all functions must be generalized to indices of other types.

##### Representing a single element
ExtensionArrays require a base object that represents an atomic element in the ExtensionArray. In our case, this would be a single row represented by a one-dimensional data vector plus a one-dimensional index vector. This corresponds to a single pandas Series in the current nested implemented in the master branch. Unfortunately, pandas Series can not be used as a base object in ExtensionArray because the interface does not allow base objects to have a `shape` property. 
A TimeBase class was therefore introduced to represent this base object. TimeBase is in its essence a thin wrapper around two pandas arrays (data and index). It primarily acts as a data storage but also implements a small number of functions dealing with comparison, print formatting, and converting to pandas Series and numpy arrays.

##### Missing data
Consistent and intuitive representation of missing data represents one of the larger challenges in creating a time series ExtensionArray. As opposed to one-dimensional data, it is not immediately clear with time series objects what should and should not count as “being missing”. Pandas requires the following definitions:

1. An atomic representation of a single row being missing. Must be able to be compared via python’s `is` operator. 
2. A function `isna(self)` that returns a Boolean vector of all items in the ExtensionArray that are missing.

The implication of allowing for `is` in 1. is that we cannot simply us a `TimeBase(None, None)` object to represent atomic missingness, since two objects are not identical unless they point to the same underlying object, which isn’t true in general. 

The following definitions of missingness are currently used to allow compatibility with pandas. These might be subject to change if they are found to pose unnecessary restrictions or if a better definition can be found:

* A row in ExtensionArray is missing as required for `isna(self)` if both all data elements and all time indices in the row are missing.
* A single missing row in ExtensionArray is represented by a missing row in both the data and the time index representation of the underlying numpy arrays. If a missing row is taken out of the context of its ExtensionArray (e.g. by `__getitem__` or `__array__`), it is represented by None.

A main problem that arises with this definition is how to deal with ExtensionArrays with only missing data. For example, if we a) have a ExtensionArray with both non-missing and missing rows, the underlying numpy arrays have a width that is determined by the non-missing rows. When we slice this array for only missing rows, each row is considered missing (i.e. no data and no index) but the underlying numpy arrays still have the width of the non-missing rows. On the other hand, if we b) create a ExtensionArray object with the same number of rows from a list of None types, we do not have any information about any width and the underlying np.ndarrays consequently have width 0. We now have two equivalent ExtensionArray that are represented differently internally. 

The current way in which we deal with this case is by throwing away the dimensions in (a) when selecting only missing rows. In this way, both will have the same underlying representation (= ExtensionArray with underlying numpy arrays of dimension Nx0).

##### Definition of (common) pandas functions for time series
Basic pandas types (e.g. numbers) allow for a number of convenient functions out of the box. Examples are counts/histograms of values in the series, mathematical and logical functions, sorting, unique values, factorization, shifting, etc. We need to decide which of these functions are relevant/defined for time series and which do not make sense and should raise a NotImplementedError. For examples of methods see the [ExtensionArray unit test suite provided by pandas](https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/base/__init__.py).

#### Other alternatives

| Data container | Comments |
|---|---|
| [numpy structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html) |  |
|[modin.pandas](https://github.com/modin-project/modin) | extends pandas, no additional time series features, but more performant |
| [datatable](https://www.h2o.ai/blog/speed-up-your-data-analysis-with-pythons-datatable-package/) | |
| sparse numpy/scipy arrays | with missing values to pad unequal length series |
| TensorFlow [ragged tensors](https://www.tensorflow.org/guide/ragged_tensor) | |
| [artic](https://github.com/man-group/arctic) | Arctic is a high performance datastore for numeric data. It supports Pandas, numpy arrays and pickled objects out-of-the-box, with pluggable support for other data types and optional versioning. |
| [Featuretools](https://github.com/Featuretools/featuretools) | Time series feature extraction, with possible conditionality on other variables with a pandas compatible relational-database-like data container |
| [xpandas](https://github.com/alan-turing-institute/xpandas) | Labelled 1D and 2D data container for storing type-heterogeneous tabular data of any type, including time series, and encapsulates feature extraction and transformation modelling in an sklearn-compatible transformer interface, work in progress. |
| [pysf](https://github.com/alan-turing-institute/pysf) | A scikit-learn compatible library for supervised forecasting with its own data container |
| [pystore](https://github.com/ranaroussi/pystore) | PyStore is a simple (yet powerful) datastore for Pandas dataframes, and while it can store any Pandas object, it was designed with storing timeseries data in mind. It's built on top of Pandas, Numpy, Dask, and Parquet (via Fastparquet) | 
| [Dask bag](https://docs.dask.org/en/latest/bag.html) | A generic Dask container of Python objects with basic parallelised operations for selecting and manipulating data |
