# Multivariate forecasting

Contributors: @davidbp, @mloning

This proposal describes a design for a multivariate (or vector) forecasting module. It specifies a template class signature for multivariate forecasters.

## Background

### Multivariate time series

A Multivariate time series can be represented as a matrix from $\mathbb{R}^{P\times T}$ where $T$ is the "length" of the timeseries and $P$ is the number of features at each time step. In this type of time series we asume that  each variable depends not only on its past values but also has some dependency on other variables. This dependency is used for forecasting future values. 

### Multivariate forecasting

Multivariate Forecasting describes the learning task in which we want to make temporal forward predictions of a given multivariate time series. Whereas in classical (or univariate) forecasting we predict only a single series, in multivariate forecasting we predict multiple series. This task is sometimes also called vector forecasting. 

Note that there are two closely related but distinct learning tasks: 
* panel forecasting (like multivariate forecasting but with i.i.d. assumption on series)
* supervised forecasting (PR [#1](https://github.com/sktime/enhancement-proposals/pull/1)) (observed future for some series)

### Example: Vector Auto Regression 

Vector auto regression (or VAR) is one of the most straight forward techniques to do multivariate time series forecasting.

In VAR each variable is a linear combination of the past values of all the other variables. 

In order words,  a vector of values   $\textbf{y}^t = (y_1^t,\dots,y_p^t) \in  Y_1^t \times \dots \times  Y_p^t$  is  predicted from the previous values of the same variables  as follows:  $\textbf{y}^t \approx \textbf{W} \textbf{y}^{t-1} +\textbf{b} $ where $ \textbf{W}$ is a matrix of weights and $\textbf{b}$ is a vector. 

This type of models can be generalized as VAR(K) where the model takes into account the past $K$ time steps of  the timeseries
$$
\textbf{y}^t \approx \textbf{W}^1 \textbf{y}^{t-1} +\textbf{W}^2 \textbf{y}^{t-2} + \dots, +\textbf{W}^K \textbf{y}^{t-K}+ \textbf{b}
$$


## API design

sktime currently has the capabilities for forecasting in univariate time series. The multivariate (or vector) forecasting design should be concordant with the previous API for univariate time series. In particular:

- "apply forecaster by row" should be close to the forecasting interface
- "forecast one time point" should be  close to the supervised time series regression interface

### Module
We propose to create a new module `sktime/multivariate_forecasting`. 

### Data container
There are some issues with multivariate data, the univariate API uses `pd.Series` objects, but those are meant to represent one dimensional ndarray objects (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html). 

We propose to use a `pd.DataFrame` with rows representing time points and columns representing variables for the multivariate target variable. This is how we currently represent multivariate series (e.g. in the `SeriesToSeriesTransformer`). 

Adding another representation will further complicate interoperability of our existing functionality and increase complexity for users.

**Example:**

A sensible way to store multiple features is to use a dataframe, as the following example suggests

```
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)
multivariate_values  = np.vstack( (y_train.values,np.ones(len(y_train.values)))).T
df_multivariate_values       = pd.DataFrame(multivariate_values, columns=["f1","f2"])
df_multivariate_values.index = y_train.index
print(df_multivariate_values0
```

```
            f1   f2
Period             
1949-01  112.0  1.0
1949-02  118.0  1.0
1949-03  132.0  1.0
1949-04  129.0  1.0
1949-05  121.0  1.0
...        ...  ...
1957-08  467.0  1.0
1957-09  404.0  1.0
1957-10  347.0  1.0
1957-11  305.0  1.0
1957-12  336.0  1.0
```

### Estimator interface
The interface should be the same as the interface for univariate forecaster with the only difference that the target variable in `fit` and `update` now is a `pd.DataFrame` (or multivariate series) rather than `pd.Series` (univariate series).

To re-use univariate forecasters, we can provide a reduction estimator that applies separate univariate forecasters to each series of the multivariate target series. This is a simple solution strategy to the multivariate forecasting problem. Note that it ignores any dependency between variables and forecasts them separately. 

We may also be able to refactor other code, however having two types (`Series` and `DataFrames`) may complicate a few things.  

### Example: reduction from multivariate to univariate forecasting

```python
def _check_forecaster(forecaster):
    # check forecaster is univariate forecaster
    pass

class Reducer:
    """multivariate forecasting -> univariate forecasting"""
    
    def __init__(self, forecaster):
        _check_forecaster(forecaster)
        self.forecaster = forecaster
        
    def fit(self, Y, X=None, fh=None):
        self.n_variables = Y.shape[1]
        
        self.forecasters_ = []
        
        for i, y in enumerate(Y):
            
            f = clone(self.forecaster)
            f.fit(y)
            forecasters_.append(f)
            
        return self
            
    def predict(self, fh=None, X=None):
        y = np.empty((len(fh), n_variables))
        
        for i, f in enumerate(self.forecasters_):
            y[:, i] = f.predict(fh)
            
        return pd.DataFrame(y, index=fh.to_absolute().to_pandas()) 
    
    def update(self, Y, X=None):
        # online learning
        for f in enumerate(self.forecasters_):
            f.update(y, X=None)
            
        return self

uf = NaiveForecaster()
mf = Reducer(uf)
mf.fit(y)
Y_pred = mf.predict(fh)
```

## Design principles

We wish to adhere to multiple principles:

- consistency with existing template interfaces in sktime, maybe it would be resonable to respresent multivariate data with a dataframe and methods that are passed a dataframe should understand that they deal with multivariate data.
- avoiding user frustration - natural user expectations on interface behaviour should be met (here separating multivariate and univariate data should be very easy to understand)


## Alternative designs

### Generalize univariate forecasting API

Instead of creating a new API, we can generalize the existing API for univariate forecasting. We can re-use the univariate forecasters currently implemented in `sktime/forecasting` and dispatch on the input type if necessary (`pd.Series` for univariate and `pd.DataFrame` for multivariate).

Given then tight coupling of estimator code with the expected data type of the target series (`pd.Series`), this alternative will involve a lot of refactoring. Having a reduction compositor may be the easier option.

### Data container: pd.Series with tuples

The following example shows one possible solution to represent multivariate data

```
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)
multivariate_values  = np.vstack( (y_train.values,np.ones(len(y_train.values)))).T
y_train_multivariate = pd.Series(list(multivariate_values))
y_train_multivariate.index = y_train.index
y_train_multivariate
```

```
Period
1949-01    [112.0, 1.0]
1949-02    [118.0, 1.0]
1949-03    [132.0, 1.0]
1949-04    [129.0, 1.0]
1949-05    [121.0, 1.0]
               ...     
1957-08    [467.0, 1.0]
1957-09    [404.0, 1.0]
1957-10    [347.0, 1.0]
1957-11    [305.0, 1.0]
1957-12    [336.0, 1.0]
Freq: M, Length: 108, dtype: object
```

Note that the code in sktime is not prepared for this a simple object NaiveForecaster fails at predict time. A simple example:

```
forecaster = NaiveForecaster(strategy="last", sp=12)
forecaster.fit(y_train_multivariate)      # this does not break current code
forecaster.predict(fh)                    # this fails
```

Outputs:

```
TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```
