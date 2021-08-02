# Pipelining for forecasting

Contributors: @aiwalter, @fkiraly, @mloning, @satya-pattnaik 

Date of discussion: 29.04.2021

### Problem statement

#### What do we mean by a "pipeline for forecasting"?
A pipeline for forecasting can be defined as follows. 

##### Fitting
During fitting, we want to:
1. Transform data for the training horizon, 
    * transform the given endogeneous variable,
    * transform any given exogeneous variables,
    * generate new exogenous variables based on given endogeneous or exogenous variables (e.g. holiday dummies),
1. Fit the final forecaster based on the transformed training data.

FK: should this not include joint transformation of multiple endo/exogeneous variables, plus possibly joint transformation of endo/exogeneous variables? E.g., temporal/on-line PCA projection or feature extraction

##### Prediction
During prediction, we want to:
1. Transform data for the forecasting horizon,
    * transform given exogeneous variables, 
    * generate new exogenous variables based on given forecasting horizon or exogenous variables (e.g. holiday dummies)
1. Generate prediction using the fitted forecaster based on forecasting horizon and any transformed exogenous data,
1. Apply inverse-transformations to the predictions in reverse order of the transformations applied to the endogenous series during fitting.
1. Return inverse-transformed predictions.

##### API design 
It is not obvious: 
* how to specify the sequence of transformation
* how to specify the series to which we apply transformations (endogeneous and/or exogeneous)
* how to handle generation of exogenous variables (transformers currently only return input series, not exogenous variables)

#### Transformer case distinctions
We need a way to distinguish whether transforms are applied to y, X or both. For example, when passing a Box-Cox transformer, it is not clear whether to apply it only to y (case 1), only to X (case 2) or to both (case 3). 

|covered? | Case | Input | Output | Example
|---|---|---|---|---|
| yes | 1 | y | yt | Box-Cox, log | 
| yes | 2 | y, X | yt| Detrending conditional on X, `ConditionalImputer` | 
|| 3 | y  | y, Xt | Holiday dummies from y, they essentially look up additional information about the index of y, hence that information will also be available when predicting, as we can look up the required information for the given forecasting horizon without knowing the predicted values yet |
| possible with ColumnTransformer | 4 | y, X | y, Xt | Fourier features from y and X |
| possible with ColumnTransformer | 5 | y, X | yt, Xt | Box-Cox, log for y and X (separate transform) |
|| 6 | y, X | yt, Xt | subsampling (joint transform) |

### Current transformer interface

```python
def fit(self, Z: Union([pd.Series, pd.DataFrame]), X: pd.DataFrame): -> self
    return self
    
def transform(self, Z, X): -> Union([pd.Series, pd.DataFrame])
    return Zt
```

#### problem of featurizer
```
t = Featurizer()
y = pd.Series(...)

y, X = t.fit_transform(y, X)
isinstance(yt, pd.DataFrame) 
```


```
t = Featurizer()
t.fit(...)
t.predict(fh)
```

#### General questions
* Should pipelines work only with transformers and no final forecaster?
* How do we handle those transformers for which no inverse-transform should be applied during prediction?
* VK: what is the input to the steps, the original X and y or the transformed X and y from the previous steps?



### API design proposals
* pipeline input argument flags, e.g. `Pipeline([(name1, transformer1, ["X"]), (name2, transformer2, ["y"]), (name3, transformer3, ["y", "X"])])`
* pipeline constructor kwargs, e.g. `Pipeline(forecaster=None, transformers=None, X_transformers=None, y_transformers=None)`
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
## Alternative API design proposals (ForecastingPipeline)

### VK: Network
```python
forecaster = ForecastingPipeline([
    ("standardize", StandardScaler(), {"X": ["X"]})
    ("fourier", FourierFeaturizer(sp=12, k=4), {"y": ["y"]}),
    ("holidays", HolidaysFeaturizer(calendar=calendar), {"y": ["y"]}),
    ("log", LogTransformer(), {"y": ["fourier"]}),
    ("build_x", BuildX(), {"inputs": ["standardize", "holidays"]}), 
    ("arima", AutoARIMA(suppress_warnings=True), {"y": "log" ,"X":"build_x")
])
```
* `BuildX` is like `FeatureUnion` in scikit-learn

### Martin Walter:

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
- having hyperparameters separate means also a bigger grid and slower grid search., however could be avoided in case only `steps` is given instead of `steps_y` and `steps_X`.

### Markus: following scikit-learn: simple ForecastingPipeline for exogenous variables only
```python
ForecastingPipeline(
    # transform X
    ("standardize", TabularToSeriesAdaptor(StandardScaler())),
    ("imputer", Imputer()),
    # transform y
    ("forecaster", TransformedTargetForecaster(
        ("outlier", HampelFilter()),
        ("imputer", Imputer()),
        ("forecaster", AutoARIMA()))
    ))
```

What about ...

```python 
TransformedTargetForecaster.fit(, X = Pipeline())
```

###  Franz: Alternative interface based on "variable role annotations"

benefit: no artificial split in input signature; annotation of "X-like" and "Z-like" variables and "excluded" variables is explicitly passed, not implicitly (by prior split into `X` and `Z`)

```python
def fit(self, X: pd.DataFrame, X_cols : List(string), Z_cols : List(string)): -> self
    return self

# assumes that cols in X are same as in fit - raises error if not
def transform(self, X : pd.DataFrame): -> pd.DataFrame
    return Xt 
```

or better in constructor?

```python
def __init__(etc etc, X_cols, Z_cols):
    blabla

def fit(self, X: pd.DataFrame): -> self
    return self

# assumes that cols in X are same as in fit - raises error if not
def transform(self, X : pd.DataFrame): -> pd.DataFrame
    return Xt 
```

"advanced" version of this merges annotation in the data container directly

```python
def fit(self, X: ColAnnotatedDataFrame): -> self
    return self

# assumes that cols in X are same as in fit - raises error if not
def transform(self, X : ColAnnotatedDataFrame): -> ColAnnotatedDataFrame
    return Xt 
```

e.g., using the `attrs` (attributes) of `pd.DataFrame`

the pipeline specification is similar to VK proposal, could be sth like

```python
forecaster = ForecastingPipeline([
    ("standardize", StandardScaler(), 'X')
    ("fourier", FourierFeaturizer(sp=12, k=4), 'Xy'),
    ("boxcox", BoxCoxTransformer()), ['my_variable_name']),
    ("holidays", HolidayFeaturizer(calendar=calendar)),
    ("residuals", ExogeneousResiduals()),
    ("arima", AutoARIMA(suppress_warnings=True))
])
```

various conventions:
* the third argument specifies which "variable block" methods are applied to, e.g., `X` means exogeneous, `Xy` is indication that feature is applied to `X` and `y` like features together
*  if not provided, arguments are mapped to sensible defaults - e.g., `HolidayFeaturizer` is applied to indices of `y` by default; multivariate transformers are applied to `X` by default
* instead of "block name", list of variable names can also be provided, e.g., `[my_variable_name]`
*  `ExogenouesResiduals` is automatically applied to any variables created by previous steps
*  transformers on exogeneous variables create new exogeneous variables, transformers on endogeneous variables create exogeneous variables (unless specified otherwise)