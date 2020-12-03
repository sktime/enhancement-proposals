# Multivariate forecasting

Contributors: @davidbp

This proposal describes a design for an updated supervised forecasting module, or an additional module `multivariate`.  It specifies a template class signature for multivariate forecasters.

### Multivariate time series

A Multivariate time series can be represented as a matrix from $\mathbb{R}^{P\times T}$ where $T$ is the "length" of the timeseries and $P$ is the number of features at each time step. In this type of time series we asume that  each variable depends not only on its past values but also has some dependency on other variables. This dependency is used for forecasting future values. 

### Multivariate (or vector) Forecasting

Multivariate Forecasting describes the learning task in which we want to make temporal forward predictions of a given multivariate time series.

### Example of vector forecasting: Vector Auto Regression 

Vector auto regression (or VAR) is one of the most straight forward techniques to do multivariate time series forecasting.

In VAR each variable is a linear combination of the past values of all the other variables. 

In order words,  a vector of values   $\textbf{y}^t = (y_1^t,\dots,y_p^t) \in  Y_1^t \times \dots \times  Y_p^t$  is  predicted from the previous values of the same variables  as follows:  $\textbf{y}^t \approx \textbf{W} \textbf{y}^{t-1} +\textbf{b} $ where $ \textbf{W}$ is a matrix of weights and $\textbf{b}$ is a vector. 

This type of models can be generalized as VAR(K) where the model takes into account the past $K$ time steps of  the timeseries
$$
\textbf{y}^t \approx \textbf{W}^1 \textbf{y}^{t-1} +\textbf{W}^2 \textbf{y}^{t-2} + \dots, +\textbf{W}^K \textbf{y}^{t-K}+ \textbf{b}
$$


## API design

Sktime currently has the capabilities for forecasting in univariate timeseries. The multivariate (or vector) forecasting design should be concordant with the previous API for univariate time series. In particular:

- "apply forecaster by row" should be close to the forecasting interface
- "forecast one time point" should be  close to the supervised time series regression interface

There are some issues with multivariate data, the univariate API uses pd.Series objects, but those are meant to represent one dimensional ndarray objects (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html). Therefore there are 2 options

- a) Use pandas dataframes where columns indicate features

- b) Use pandas series created with tuples



#### a) pandas dataframes where columns indicate features

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

This type of data container though is not accepted by sktime

```
forecaster = vector.NaiveVectorForecaster(strategy="last", sp=12)
forecaster.fit(df_multivariate_values)
```

```
AttributeError: 'NaiveVectorForecaster' object has no attribute '_set_y_X'
```

#### b) pd.Series with tuples

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



## Design principles

We wish to adhere to multiple principles:

- consistency with existing template interfaces in sktime, maybe it would be resonable to respresent multivariate data with a dataframe and methods that are passed a dataframe should understand that they deal with multivariate data.
- avoiding user frustration - natural user expectations on interface behaviour should be met (here separating multivariate and univariate data should be very easy to understand)



