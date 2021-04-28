# ForecastingPipeline and ForecastingColumnTransformer

Contributors: @aiwalter

## Problem statement

The current `TransformedTargetForecaster` can only apply tranfsormers to `y` and not to `X`. Therefore, we need a new class as descibed in the STEP [here](https://github.com/sktime/enhancement-proposals/blob/8200d8d5ec409bf76a2f8af6d12b17fe201afa0a/steps/01_forecasting_api/forecasting-with-exogenous-variables.md)

## Description of proposed solution
To apply transformers to `X`, we might need two classes. First class is a `ForecastingPipeline` and second class sth like a `ForecastingColumnTransformer`.

## ForecastingPipeline
### 1. Option

In a first step, having and intuitive solution like this as described in the existing STEP:
```python
pipe = ForecastingPipeline(steps=[
    ("standardize", TabularToSeriesAdaptor(StandardScaler()), ["X"])
    ("outlier", HampelFilter(), ["y"]),
    ("imputer", Imputer(), ["y", "X"]),
    ("arima", AutoARIMA())
])
```
Iternally, there would be a touple `(name, transformer_y, transformer_X, identifier)` to separate fitting of `y` and `X`. Both transformers would have the same hyperparameters.

*PROs:*
- transformers of `y` could receive the transformerd `X` synchronously with transforming `y` and `X`

*CONs:*
- breaking the `sklearn` convention of pipelining and definition of a `step` as `(name, estimator/transformer)`


### 2. Option

Sticking to the `sklearn` tuples of `(name, transformer)` by having two separate `steps` like `steps_y` and `steps_X` as arguments for `ForecastingPipeline`. Both would be separate pipelines inside the `ForecastingPipeline`. The would be a separate arguement `forecaster`, so the foreacster would not be inside `steps` as in `TransformedTargetForecaster` but separate. Example:
```python
pipe = ForecastingPipeline(
    steps_y=[
        ("outlier", HampelFilter()),
        ("imputer", Imputer())
        ],
    steps_X=[
        ("standardize", TabularToSeriesAdaptor(StandardScaler())),
        ("imputer", Imputer()),
        ],
    forecaster=AutoARIMA()
)
```
In case steps are identical, we could just accept `steps` and copy this internally to have `steps_y` and `steps_X`.

*PROs:*
- writing steps as `(name, estimator/transformer)` is conform with exsting convention of `sklearn`
- transformers of `y` and `X` could have different hyperparameters

*CONs:*
- not obvious how to access the params when doing grid search. Names of steps must be given unique like `imputer_y` and `imputer_X`. If two names are same, we would raise an error.
- having hyperparameters separate means also a bigger grid and slower grid search.

## Related issue
We might have to implement the `X` transformation of some existing transformers which currently only supprt `pd.Series` transformations. Otherwise the application of a `ForecastingColumnTransformer` would be needed.

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


