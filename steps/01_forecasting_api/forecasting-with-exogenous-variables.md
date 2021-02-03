# Multivariate Time Series: Transformers, Forecasting, Pipelining, Reduction.

Contributors: @fkiraly, @mloning 

[TOC]


## Scityping of Time Series Transformers 

:::warning
one general question seems to be whether `transform(X, y)` should only return `X` as in scikit-learn, or return both `X` and `y`, in the latter case we would have to provide our own pipeline class but could fix the error they made at the beginning
:::

### Problem statement
1. Find a taxonomy of time series transformations using scientific types based on input and output types
2. Develop transformer class structure/hierarchies in Python based on scitype taxonomy

### Transformer scitypes

#### Series -> primitives (non-fittable)
* encapsulated in functions, not classes, because no parameters have to be stored (not stateful, no fitted state)
* may have tuneable arguments (e.g. which quantile to extract), which can be tuned via kwargs loopthrough in compositors like `FunctionTransformer`

:::info
disagree that series->primitives should be functions - these can still have parameters that one could tune in a composite! How would you tune these parameters without exposing them?
ML: via FunctionTransformer, no?

I would advocate: these are still callables with specific signatures, but still inherit from BaseEstimator. That way, they can be constructed with parameters, and behave like a function via th `__call__` method.
:::


##### Interface
```python
def series_to_primitive(*args, **kwargs):
    # non fittable
    pass

class FunctionTransformer:
    pass
```

##### Examples
* "arguments of scikit-learn's `FunctionTransformer`" (e.g. mean, min, std)

#### Series -> series (fittable/non-fittable)

| input type | output type | same data type |
|---|---|---|
| single series | single series | yes |

* may be fittable or non-fittable (empty `fit`)
* if not fittable, generally encapsulated as functions (they are not stateful, there is no fitted state), only wrapped in transformers for convenience. 
* Need to introduce sktime's own `series -> series` `FunctionTransformer` 
* input and output of `transform` may have different number of time points and different index
* input and output have different domain (no longer on the same time line)
* in a forecasting pipeline, `fit_transform` is called before fitting the final forecaster, and `inverse_transform` is called after calling predict of the final forecaster

##### Interface:
```python
def series_to_series(*args, **kwargs):
    # non-fittable
    pass

class SeriesToSeriesFunctionTransformer:
    pass

class SeriesToSeriesTransformer:

    def fit(self, y, X=None): 
        return self
        
    def transform(self, y, X=None):
        # y may be empty or index-only 
        # (without any values), e.g. during
        # prediction in forecasting
        return yt, X  # series 
        
    def inverse_transform(self, y, X=None):
        return yt, X  # series
```

:::warning
* also accept multivariate input?
:::


##### Examples:
* **Normalisers**: in same domain, always return the transformed input series, usually do not require X
    * Box-Cox transformer
    * Logarithm
    * Smoothing, filters
    * Detrender
    * Deseasonalisation
* **Annotators**: 
    * Change point annotator (e.g. returns sequence of change point waiting times)

:::warning
* **Featurizers**: add features to input series and return input series and features, in `transform` has to accept empty series with time index only so that it can work in forecasting via extrapolating results from fit for given index (forecasting horizon), usually do not require X. In a forecasting pipeline, `fit_transform` is called before calling `fit` of the final forecaster and `transform` is called *before* calling `predict` of the final forecaster, hence they need to be aware of the forecasting horizon and need to work even when `y` is not given
    * Adding holiday dummies 
    * Adding Fourier terms

A few questions/comments:
* The Featurizer seems to break sklearn's pipeline/transformer API when no exogenous variables are present, as sklearn Transformers do not pass on y and X, but here we need to pass in y and return y and X (the generated features), so we need to pre-initialise X=None 
* Could the extrapolation in prediction be generalised so that the Featurizer could be a wrapper around other Forecasters? It seems strange to have a FourierTransformer and a FourierFeaturizer
* Is the Featurizer different from the Annotator?
:::


##### Examples


#### Series-as-features -> series-as-features

| input type | output type | same data type |
|---|---|---|
| series-as-features | series-as-features | not necessarily due to unequal length

* fittable (at least some parameters, e.g. number of segments)
* input of `fit` and `transform` may have different number of instances, but same number of columns and time points
* input and output of `transform` have same number of instances
* input and output of `transform` may have different numbers of columns and time points (unequal length)

##### Interface:
```python
class SeriesAsFeaturesTransformer:

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return Xt  # panel data

    def inverse_transform(self, X, y=None):
        return Xt  # panel data
```

:::warning
* Any non-trivial examples of `inverse_transform`?
FK: FFT vs IFFT? sequence PCA/SVD projection (& backprojection)?
* TimeSeriesConcatenator could simply remember where it concatenated the series and hence could inverse transform them
:::

##### Examples:
* dictionary-based transforms 
* time series segmentation: `IntervalSegmenter` (e.g. splitting a time series columns into multiple ones)
* time series concatenation: `ColumnConcatenator` (merging multiple time series columns into a single one)
* composite transformers:
    * e.g. `RowTransformer` with `series -> series` (see below) applying series-to-series or series-to-primitive transforms iteratively over rows

### Series-as-features -> tabular

| input type | output type | same data type |
|---|---|---|
| series-as-features | tabular | no |

* fittable
* input of `fit` and `transform` may have different number of instances, but same number of columns and time points
* input and output of `transform` have same number of instances
* input and output of `transform` may have different numbers of columns and time points (meaning of "column" changes, time points are now represented as columns)

##### Interface:
```python
class SeriesAsFeaturesToTabularTransformer:

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return Xt  # tabular

    def inverse_transform(self, X, y=None):
        return Xt  # tabular
```

##### Examples:
* Shapelet transform (returns minimum distance of each instance to found shapelets as tabular matrix)
* Tabularizer/TimeBinner
* composite transformers:
    * `RowTransformer` with `series -> primitives` (see below)
    * `RandomIntervalFeatureExtractor` series-as-features to primitive feature extractors  (e.g. mean)

:::warning
Perhaps we should rename `RowTransformer` to `InstanceTransformer` inline with our agreed glossary
:::

#### Meta-transformers/adaptors 


##### Chaining/Concatenators - "pipeline"
* [series -> series] x [series -> primitives] -> [series -> primitives]
* [series-as-features -> series-as-features] x [series-as-features -> tabular] -> [series-as-features -> tabular]


##### Appliers

* [series -> series] -> [series-as-features -> series-as-features] (e.g. `RowTransformer` with `series -> series` transformers (i.e. fitting and transforming of series -> series transformers on each instance in both `fit` and `transform`, no fitted parameters of the component transformers are kept), works with both fittable and non-fittable `series -> series` transforms)
* [series -> primitives] -> [series-as-features -> tabular] (e.g. `RowTransformer` with `series -> primitives` transforms)


##### Time-as-df-index-considerer

##### Singleton
* t: [series -> tabular] "time index forgetter"
* u: [tabular -> series], where time indices are remembered "time index back adder"


##### [tabular -> tabular] -> [series -> series]

* [f mapsto uft]
* `SingleSeriesAdaptor`: apply scikit-learn like transformers to single series

##### [tabular -> tabular] -> [series-as-features -> series-as-features]

##### Variable selectors and unions
* ColumnTransformer
* FeatureUnion

:::warning
should we have special transformer types for reduction meta-estimators, e.g. ReducedRegressionForecaster has a transform method which we could expose with the scitype [series -> [tabular, series]]
:::

## Forecasting: Pipelining

### Problem statement
In fit, we 
1. apply transformations, possibly on both the endogeneous and exogeneous variables,
2. fit final forecaster.

In predict, by contrast, we 
1. apply transformations 
    * to exogeneous variables (there is no endogeneous variable, that's what we want to predict), 
    * extrapolate generated features for the given forecasting horizon (e.g. holiday dummies)
3. make a prediction using the fitted forecaster,
4. apply inverse-transformations of the transformations applied to the endogenous series during fitting on the predictions.

It is not obvious: 
* how to specify the sequence of transformation
* how to specify the series to which we apply transformations (endogeneous and/or exogeneous)

We need a way to distinguish whether transforms are applied to y, X or both. For example, when passing a Box-Cox transformer, it is not clear whether to apply it only to y (case 1), only to X (case 2) or to both (case 3). 

| Case | Input | Output | Example
|---|---|---|---|
| 1 | y | yt | Box-Cox, log | 
| 1 | y  | y, Xt | Holiday dummies from y, they essentially look up additional information about the index of y, hence that information will also be available when predicting, as we can look up the required information for the given forecasting horizon without knowing the predicted values yet |
| 2 | y, X | y, Xt | Fourier features from y and X |
| 3 | y, X | yt, Xt | Box-Cox, log for y and X |

### Options
* pipeline input argument flags, e.g. `Pipeline([(name1, transformer1, ["X"]), (name2, transformer2, ["y"]), (name3, transformer3, ["y", "X"])])`
* pipeline constructor kwargs, e.g. `Pipeline(forecaster=None, transformers=None, X_transformers=None, y_transformers=None)`
* pipeline input argument flags, e.g. `Pipeline([(name1, transformer1, ["X"]), (name2, transformer2, ["y"]), (name3, transformer3, ["y", "X"])])`
* transformer types, e.g. `SingleSeriesTransformer`, `SingleSeriesFeaturizer`, `Featurizer`, `SeriesTransformer` corresponding to the four cases in the table
* transformer input argument, e.g. `MyTransformer(**kwargs, apply_to=["X"])`
* composition/wrapper classes (pipeline helper classes like `FeatureUnion`) that pass arguments appropriately, e.g. `TargetTransformer(*transformers)`, `ExogenousTransformer(*transformers)`, `JointTransformer(*transformers)` or even simpler: a `SeriesSelector([(name1, transformer1, ["X"]), (name2, transformer2, ["y"])])` similar to the ColumnTransformer
* different pipeline classes
* sequential API
* single pd.DataFrame with annotation for target, exogeneous (similar to task object) 


#### Pipeline input argument flags
* requires to write a new `ForecastingPipeline`
```python
forecaster = ForecastingPipeline([
    ("standardize", StandardScaler(), ["X"])
    ("fourier", FourierFeaturizer(sp=12, k=4), ["y"]),
    ("holidays", HolidaysFeaturizer(calendar=calendar), ["y"]),
    ("log", LogTransformer(), ["y"]),
    ("arima", AutoARIMA(suppress_warnings=True))
])
```

#### Composition classes
* like scikit-learn
* requires to write a new `ForecastingPipeline`
* FourierTransform would still be different from FourierFeaturizer

```python
forecaster = ForecastingPipeline([
    ("standardize", StandardScaler()),  # X
    ("fourier" FourierTransform()),  # X
    ("forecaster", TransformedTargetForecaster([  # y
        ("fourier", FourierFeaturizer(sp=12, k=4)),
        ("holidays", HolidaysFeaturizer(calendar=calendar)),
        ("log", LogTransformer()),
        ("arima", AutoARIMA(suppress_warnings=True))]))
])
```

## Forecasting: Reduction

### Problem statement
Reduction seems to be a special case of a pipeline. We transform the input series into a matrix of lagged variables using a sliding window tabularisation. 

While reduction is relatively clear in the univariate forecasting case with no exogeneous variables, the case with exogeneous variables adds some complications. In some cases, exogeneous variables should be transformed using the sliding window tabularisation similar to the endogeneous series. In other cases, exogeneous variables should be passed without transformation to the wrapped regressor, as they represent extracted features like Fourier transform coefficients

It is not obvious: 
* how to pass exogeneous variables through the reduction step
* forecasting: information transfer from past to future
* regression: information transfer from X to y 
* reduction with exogeneous features seems combination of both

```python
pipe = ForecastingPipeline(
    Detrender(...),  # endogeneous
    FourierTransform(...) # exogenous
    Reducer(DecisionTree(...), window_length=2)
)

pipe = ReducerPipeline(
    Detrender(...),  # endogeneous
    FourierTransform(...), # exogenous
    SlidingWindowTransformer(window_length=2),
    DecisionTree(...),
)
```

### Simple cases
* passing holiday dummies to the wrapped regressor sidestepping sliding window tabularisation
```python
forecaster = TransformedTargetForecaster([
    ("holidays", HolidayFeaturizer()),
    ("regress", Reducer(RandomForestRegressor()))
])
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
```


### Options
* include kwarg for including/excluding columns in constructor
```python
class Reducer:

    def __init__(self, regressor, window_length=None, include_X_cols=None):
        self.include_cols = include_cols
        self.regressor = regressor
        self.window_length = window_length

    def fit(self, y, fh=None, X=None):
        
        X_train, y_train = self.transform(y, X)
        self.regressor.fit(X_train, y_train)
        
        return self

    # special scitype: [y, X] -> [tab(y), tab(X)]
    def transform(self, y=None, X=None):
        # select columns
        X_to_transform = X.drop(columns=self.include_cols)
        
        # trim, also needs to take account of step size (using reindex/loc)
        X_to_skip = X.iloc[self.window_length:, self.include_cols]
        
        # apply sliding-window-transform
        y_train, X_train = sliding_window_transform(y, X_to_transform)
        X_train = pd.concat([X_train, X_to_skip], axis=1)
        return X_train, y_train
        
    def predict(self, fh, X):
        X_test = self.transform(X)
        return self.regressor.predict(X_test)

```
