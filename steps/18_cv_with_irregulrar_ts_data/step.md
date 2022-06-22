# Cross-validation with irregular time series data

Contributors: @khrapovs

## Problem statement

Currently, all splitters in `sktime/forecasting/model_selection/_split.py` assume that the data passed to `.split(y)` method is "regular". By "regular" we mean here that $(y_1,y_2,\ldots,y_t)$ are all equally spaced, there are no missing values (all indices $1,\ldots,t$ are present), and there is only one observation for each time index. This works fine for a typical forecasting pipeline:
- get raw irregular data
- process data (aggregate, impute, align, etc.)
- forecast $y$

This also works well for a typical cross-validation pipeline:
- get raw irregular data
- process data (aggregate, impute, align, etc.)
- cross-validate (repeat the following many times for different splits):
  - split
  - forecast $y$
  - record forecasts, residuals, etc
- aggregate cross-validation results

But the existing implementation fails when some data processing is required after each split operation:
- cross-validate (repeat the following many times for different splits):
  - split
  - **process data**
  - forecast $y$

One example use case is the following. Suppose that besides the main forecasting model, there is another forecasting model, let's call it auxiliary model, that generates additional exogenous variables to be used in the main model. So the pipeline in this case looks as follows:
- cross-validate (repeat the following many times for different splits):
  - split
  - **process data**
  - **forecast $Z$ from the auxiliary model**
  - **forecast $y$ from the main model using both $X$ and $Z$**

Data processing after splitting (not before) is essential here in order to preserve cross-validation fairness in the sense of avoiding the usage of future data. If the data was processed (e.g. aggregated and imputed) before splitting, this would potentially leak future information into the past.

For preliminary discussions of the proposal presented here, see issue:
- https://github.com/alan-turing-institute/sktime/issues/1737

## Description of proposed solution

Suppose we have the following time series: $(y_1,y_2^a,y_2^b,y_4)$. For this ordering the corresponding enumeration is $(1,2,3,4)$ . Here we have two observations for the time index $2$ and no observation for time index $3$.

**Current state**

Below is the list of cutoffs and train/test splits that take into account only the order of observations:

| cutoff | train               | test                |
|--------|---------------------|---------------------|
| 1      | $(y_1)$             | $(y_2^a,y_2^b,y_4)$ |
| 2      | $(y_1,y_2^a)$       | $(y_2^b,y_4)$       |
| 3      | $(y_1,y_2^a,y_2^b)$ | $(y_4)$             |

or, in terms of enumeration indices:

| cutoff | train             | test      |
|--------|-------------------|-----------|
| 1      | $(1)$             | $(2,3,4)$ |
| 2      | $(1,2)$           | $(3,4)$   |
| 3      | $(1,2,3)$         | $(4)$     |

**Expected state**

Below is the list of cutoffs and train/test splits that take into account original time indices:

| cutoff | train               | test                |
|--------|---------------------|---------------------|
| 1      | $(y_1)$             | $(y_2^a,y_2^b,y_4)$ |
| 2      | $(y_1,y_2^a,y_2^b)$ | $(y_4)$             |
| 3      | $(y_1,y_2^a,y_2^b)$ | $(y_4)$             |

or, in terms of enumeration indices:

| cutoff | train               | test      |
|--------|---------------------|-----------|
| 1      | $(1)$               | $(2,3,4)$ |
| 2      | $(1,2,3)$           | $(4)$     |
| 3      | $(1,2,3)$           | $(4)$     |

This example hints for the correct implementation to achieve the goal. Splitters internally should work with original time indices and ignore the order in which the data is given. This would also solve the case when the data is not sorted over time.
