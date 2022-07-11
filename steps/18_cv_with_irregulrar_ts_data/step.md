# Cross-validation with irregular time series data

Contributors: [@khrapovs](https://github.com/khrapovs)

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

### Numerical example

Suppose we have the following time series: $\left(y_1,y_2^a,y_2^b,y_4\right)$. 

For this ordering the corresponding enumeration is $(1,2,3,4)$. Here we have two observations for the time index $2$ and no observation for time index $3$.

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

### Formal definition and proposal

Denote a time series index as $T=\left\{t(1),\ldots,t(k)\right\}$. Assume that it is sorted, that is $t(i+1)\geq t(i)$. Also assume that $t(i)$ can be either an integer or a date/time value.

**Definition.** A time series is regular if $t(i+1)-t(i)=t(j+1)-t(j)$ for any $i,j\in\{2,\ldots,k\}$. Conversely, a time series is irregular, if there exists $i\neq j$ such that $t(i+1)-t(i)\neq t(j+1)-t(j)$.

**Definition.** A cutoff is a reference to the index $t(s)$ such that $t(1)\leq t(s)\leq t(k)$. It separates train and test windows, $F=\left\{t(m_1),\ldots,t(m_f)\right\}$ and $P=\left\{t(h_1),\ldots,t(h_p)\right\}$, respectively. Exact definition of a train/test window depends on a specific splitter. Regardless of a splitter, $t(s)\geq t\in F$ and $t(s)< t\in P$. 

For a regular time series it is guaranteed that any cutoff $t(s)\in T$. Conversely, for irregular time series there exists $s$ such that $t(s)\notin T$.

The current state of `sktime` supports only regular time series. At the core the implementation relied on constructing train/test windows using `np.arange`, which was sufficient given the knowledge of window left and right endpoints. For example,
```python
np.arange(train_start, train_end + 1)
```
gave us `iloc` references to the train window.

After a series of refactoring PRs this implementation was generalized using `pandas.Index.get_loc` and `numpy.argwhere` methods. The first one is used to obtain `iloc` reference $s$ in $t(s)$, while the second is used to get `iloc` references $\{m_1,\ldots,\m_f\}$ and $\{h_1,\ldots,h_p\}$. For example,
```python
train_end = y.get_loc(cutoff)
```
gives us the `iloc` reference to the end of the training window, while
```python
np.argwhere((y >= train_start) & (y <= train_end))
```
gives us `iloc` references to the train window. The advantage here is that we may pass an irregular time series and still get correct `iloc` indices.

Going deeper into the implementation it turns out that such a refactoring is still not sufficient to treat all currently existing splitters. In particular, `y.get_loc(cutoff)` raises `KeyError` if `cutoff` does not belong to the index `y`. We propose to treat this as follows. For an irregular index $T=\left\{t(1),\ldots,t(k)\right\}$ we can construct a corresponding regular index $T^\prime=\left\{t^\prime(1),\ldots,t^\prime(l)\right\}$ such that $t(1)=t^\prime(1)$ and $t(k)=t^\prime(l)$. For such an index 
```python
y_regular.get_loc(cutoff)
```
always returns a meaningful `iloc` reference in the context of a regular time index `y_regular`. Same for
```python
np.argwhere((y_regular >= train_start) & (y_regular <= train_end))
```
After obtaining a train and/or test windows one has to convert them back to the context of original irregular index `y`. This can be achieved by using, for example,
```python
y.get_indexer(y_regular[train])
```
which returns `iloc` references to `y` for only those elements of `y_regular[train]` that exist in `y`.

Constructing `y_regular` for integer valued `y` is trivial:
```python
np.arange(y[0], y[-1] + 1)
```
For date/time `y` one needs to know the frequency of a time series after aggregation/imputation. Then, for example,
```python
pd.period_range(y.min(), y.max(), freq=freq)
```
produces the desired result. Currently, if one passes an irregular time index to any splitter in `sktime`, there is no robust way to guess a desired frequency since aggregation/imputation may be performed for any time unit. Hence, it is required to implement one more optional argument in splitter constructor, namely `freq`:
```python
def __init__(
    self,
    fh = DEFAULT_FH,
    window_length = DEFAULT_WINDOW_LENGTH,
    freq: str = None,
) -> None:
    self.window_length = window_length
    self.fh = fh
    self.freq = freq
```
