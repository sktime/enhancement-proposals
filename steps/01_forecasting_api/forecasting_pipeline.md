# ForecastingPipeline and ForecastingColumnTransformer

Contributors: @aiwalter

## Problem statement

The current `TransformedTargetForecaster` can only apply tranfsormers to `y` and not to `X`. Therefore, we need a new class as descibed in the STEP [here](https://github.com/sktime/enhancement-proposals/blob/8200d8d5ec409bf76a2f8af6d12b17fe201afa0a/steps/01_forecasting_api/forecasting-with-exogenous-variables.md)

## Description of proposed solution
To apply transformers to `X`, we might need two classes. First class is a `ForecastingPipeline` and second class sth like a `ForecastingColumnTransformer`.

### ForecastingPipeline
In a first step, having and intuitive solution like this as described in the existing STEP:
```python
forecaster = ForecastingPipeline([
    ("standardize", TabularToSeriesAdaptor(StandardScaler()), ["X"])
    ("outlier", HampelFilter(), ["y"]),
    ("imputer", Imputer(), ["y", "X"]),
    ("arima", AutoARIMA())
])
```
### ForecastingColumnTransformer
In a second step (separate PR), we might want to also specify the exact columns of `X` like in `sklearn.ColumnTransformer`. This means we would need sth equivalent like a `ForecastingColumnTransformer` or `ExogTransformer`. This would look as follows all together:
```python
column_transformer = ForecastingColumnTransformer(
    transformers=[
        ("imputer", Imputer(), [0,4,7]),
        ("standardize", TabularToSeriesAdaptor(StandardScaler())), [3, 5])

forecaster = ForecastingPipeline([
    ("outlier", HampelFilter(), ["y"]),
    ("columntransformer", column_transformer, ["y", "X"]),
    ("arima", AutoARIMA())
])
```
`ForecastingColumnTransformer` would also be able to receive `y`, in case `["y", "X"]` is given.

## Related issue
We might have to implement the `X` transformation of some existing transformers which currently only supprt `pd.Series` transformations. Otherwise, this could be handled by applying a series transformers n times to given n columns of `X` inside `ForecastingPipeline`. So in `ForecastingPipeline` we could do sth like...
```python
try:
    y, X = transformer.fit_transform(y, X)
except:
    y = transformer.fit_transform(y)
    for col in X.columns:
        X.loc[:, col] = transformer.fit_transform(X[col])
```

