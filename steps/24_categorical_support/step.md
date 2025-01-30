# Adding categorical support to sktime

Contributors: @Abhay-Lejith

## Introduction

### Objectives 
Allow users to pass exog data with categorical features to any forecaster. Outcome will depend on whether the forecaster natively supports categorical or not. In any case, ambiguous errors must not be raised, and all cases must be handled appropriately.

### Current status:
* Non-numeric dtypes are completely blocked off in datatype checks.
* https://github.com/sktime/sktime/pull/5886 - This pr removes these checks but has not been merged yet as most estimators do not support categorical and will break without informative error messages.
* https://github.com/sktime/sktime/pull/6490 - pr adding feature_kind metadata


### Conceptual model

We will need to cover situations in which:

1. data may contain one or both of categorical variables, numerical variables
    * example: promotion type 1, 2, or no promotion in sales data
2. specific estimators may support, or not natively support categorical variables.
  We always assume estimators can support categorical variables
    * example: `ARIMA` does not natively support categorical variables. Direct reduction to `HistGradBoost` does support
      categoricals in endogenous and exogenous variables.
3. specific estimators in upstream packages (3rd party estimators)
  may have parameters that directly refer to the categorical
  variables or subsets thereof. This may be something we want to expose to the user or not.
    * example: `skforecast`, `categorical_features` variable
4. categoricals, and support for, may appear in endogeneous, exogeneous data, or both.
  It is more common that models support exogeneous categoricals than endogeneous ones.

Challenge: a special situation arises if we build composites such as `make_reduction`.
The component may or may not support categoricals, but that information may
not be easily obtained from the interal estimator, e.g., `sklearn`.


### Sample data used in examples below
```python
sample_data = {
    "target": [10,20,30,20],
    "cat_feature":["a","b","c","b"],
    "numeric_feature": [1,2,3,4],
}
sample_dataset = pd.DataFrame(data=sample_data)
y = sample_dataset["target"]
X = sample_dataset[["cat_feature","numeric_feature"]]
X_train,X_test,y_train,y_test = temporal_train_test_split(y,X,0.5)
```

## Requirements

1. Need to know whether categorical data is present in X_train and which columns.
   * This is because point 1 in conceptual model.
2. Need to know if forecaster has native categorical support.
   * From point 2 in conceptual model
3. In reduction, we need to know if model used has native categorical support.
4. In reduction, we need to pass parameters(like `enable_categorical=True` in case of xgboost) to the model used if natively supported according to the requirements of the model.
5. Must handle all combinations of cases
    - estimator natively supports categorical: yes/no
    - categorical feature specified by user: yes/no
    - categorical present in data: yes/no

    Important case is when categorical data is passed but is not natively supported.

## About reduction case:

### Models which support categorical natively that have been identified so far:
- xgboost
- lightgbm
- catboost

- Note that all these will have to be used via reduction.
- I suggest creating a wrapper class for these models on top of make_reduction. This is only for regressors which support natively and have to be used via reduction. The discussion below and designs discussed apply to forecasters but right now, there is no forecasting algorithm which has native support. So these regression models in wrapper classes can serve as forecasters with native support.
- Otherwise as of now, though we are discussing cases like forecaster has native support, categorical passed/not passed etc, there is no forecaster(with native support) in sktime which will use any of this functionality right now.

Instead of this, the other option of modifying the existing make_reduction to accommodate the native categorical capability of these 3 models will require some form of hard coding because these are from three different libraries(not sklearn regressors) and have different requirements and expectations.(discussed below in req3 and req4)


## Solutions for some of the requirements:
    
### Requirement 1
- https://github.com/sktime/sktime/pull/6490 - pr adding feature_kind metadata
    
    This pr adds a new field to metadata called feature_kind. It is a list from which we can infer the dtype of the columns present in the data. Currently a broad classification into 'float' and 'categorical' is used.

    #### Sub-challenge: User may want to treat a column with numerical values as a categorical feature(Example - a column which contains ratings from 1-5)
    
    This information cannot be detected using the feature_kind metadata. It has to be passed by the user in some manner.

    - Possible solutions:
        - 'categorical_features' arg in model where user can specify such columns.
        - passing this info via `set_config`.
        - not deal with this case and expect user to convert from numerical to categorical. We can make a transformer for this purpose.

### Req 2
We will use a tag `categorical_support`(=True/False) or similar to identify whether a forecaster has native support.

### Req 3
- Set tag depending on model used in reduction.
    - Since there are few models(3) that have been identified with native support, a hard coded method like an if/else statement to check the model could be used.
        - Would be more difficult to extend if new estimators with native support are found.
    - Maintain some sort of list internally in sktime, of estimators with native support and check with this.

### Req 4
- Should this be expected from the user itself since they have to initialize the model anyway?
- Or, if we choose to pass arguments internally, then we will have to do something like this:
```
if model used is xgboost:
    xgboost specific categorical specification
else if model used is catboost:
    catboost specific cat specification 
else if ..... and so on
```

### Req 5 

## Major case which influences the rest of the design is 
- Categorical data is passed but forecaster does not support natively.

There are two broad ways to deal with this. Rest of the design will depend on this decision.

1) No internal encoding 
    - a) return error saying categorical is not supported by this forecaster.
    - b) drop categorical columns with a warning and proceed to fit.
2) Perform internal encoding
    - a) use fixed default encoder
    - b) provide option to user on encoders to use and on which columns.


1) raise error if mismatch

2) hard coded handling
  * drop
  * one-hot
  * other specific handling

3) user choice in handling
  * sub-decision: default or not
     * sub-decision: if default, which, same as 2



for the above cases there are even more combinations based on which solution we take for the subchallenge mentioned under requirement 1.(cat_features arg, no arg, set_config)
There are too many designs due to the large combinations of solutions possible and they vary greatly depending on the above broad decision we take.

So I suggest taking a call on whether we want to perform internal encoding or not first and then we can explore the corresponding possible designs in detail.

I have listed the specifics for a few combinations below so we have an idea of what is to be expected from the two broad outcomes.

---
---

### Design 1a) . Allow categorical features ONLY in those forecasters which support natively. (no encoding internally)

- `categorical_features` parameter will be present only in forecasters that support natively.
    
    #### How tag is going to be used
    - in datatype checks :
    ```
    if categorical support tag is false:
        if categorical feature is present:
            error - cat feature is not supported by this estimator
    ```       

- In this case `categorical feature` will not be a parameter in models that do not support. So (NO,YES,_) is not possible.    

| **Estimator supports categorical natively** | **Cat feature specified by user in fit** | **Cat feature present in data** | **Outcome** | 
|:-----------------:|:-------------------------:|:--------------:|:---------------:|
| NO | NO | NO | normal case, fits without error. |
| NO | NO | YES| informative error message to user that categorical feature is not supported by the forecaster*(see below table) |
| NO | YES| NO | X |
| NO | YES| YES| X |
| YES| NO | NO | normal case, fits without error. |
| YES| NO | YES| use categorical columns as detected by feature_kind and proceed to fit. |
| YES| YES| NO | informative error message that specified column was not present in data**|
| YES| YES| YES| use categorical columns as detected by feature_kind along with the user specified columns(union) and proceed to fit. |

- *for design 1b) we will instead drop the categorical column and proceed to fit.
- **in case where specified column was not found, we can also choose to ignore that column and continue to fit with warning that column was not found.

#### PROS:
- No need to add `categorical_feature` parameter to all models, only the few which support natively.

#### CONS:
- Less number of estimators support categorical natively.

#### Example:
Suppose XYZforecaster has native support (this could also be the previously discussed wrapper class for regressors using reduction like say isntead of XYZforecaster we have XGBReductionRegressor) - the following runs successfully
```python
from sktime import XYZforecaster

forecaster = XYZForecaster(cat_features=['cat_feature'])
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```
Internally the union of categorical features from feature_kind and those specified by user are used by the forecaster to fit.

Suppose XYZforecaster does not have native support (therefore it does not have `cat_features` arg)
```python
from sktime import XYZforecaster

forecaster = XYZForecaster()
forecaster.fit(y_train,X_train)
```
Output:
```
Error: This forecaster does not have native categorical support.
```
*In design 1b) we will instead drop the categorical columns and proceed to fit/predict.

---
---

### Design 2a). Perform some encoding method as default if not natively supported.

- `categorical_feature`  param will be present in all models.

    #### How tag is going to be used:
    ```
     if cat support tag is false:
         encode cat features and continue
     else:
         continue
    ```

| **Estimator supports categorical natively** | **Cat feature specified by user in fit** | **Cat feature present in data** | **Outcome** | 
|:-----------------:|:-------------------------:|:--------------:|:---------------:|
| NO | NO | NO | normal case, fits without error. |
| NO | NO | YES| encode features internally as detected by feature_kind and continue.
| NO | YES| NO | error that specified column was not present in data |
| NO | YES| YES| use categorical features detected by feature kind and those specified by user(union) and proceed to encode and then fit |
| YES| NO | NO | normal case, fits without error.|
| YES| NO | YES| use categorical features detected by feature_kind and fit|
| YES| YES| NO | error that specified column was not present in data|
| YES| YES| YES| use categorical features detected by feature kind and those specified by user(union) and proceed|

- in case where specified column was not found, we can also choose to ignore that column and continue to fit with warning that column was not found.

#### PROS:
- All forecasters will be able to accept categorical inputs.

#### CONS:
- adding `cat_features` parameter to all estimators is not desired
- Not flexible in choice of encoding (label,one-hot,etc) if we fix a default. If user wants different types, will have to resort to externallly doing it anyway.

#### Example:
- XYZforecaster has native support - the following runs successfully
- does not have native support -  categorical features are encoded and used in fit/predict.
        
```python
from sktime import XYZforecaster

forecaster = XYZForecaster(cat_features=['cat_feature'])
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```

---
---

### Design 2a). default encoding but NO `cat_features` parameter in model
Instead, user is required to convert the required numerical features to categorical. To ease the process, we can make a `transformer` to make this conversion.

- no `cat_features` parameter in models.

| **Estimator supports categorical natively** | **Cat feature present in data** |  **Outcome**  |
|:-------------------------------------------:|:-------------------------------:|:---------------:|
| NO |  NO | normal case, fits without error. |
| NO |  YES| encode features internally as detected by feature_kind and continue to fit.
| YES|  NO | normal case, fits without error.|
| YES| YES| use categorical features detected by feature_kind and fit|

#### PROS:
- All forecasters will be able to accept categorical inputs.

#### CONS:
- Not flexible in choice of encoding (label,one-hot,etc) if we fix a default. If user wants different types, will have to resort to externallly doing it anyway.
- Removing parameter comes at cost of requiring user to convert numerical to categorical themselves. 
    - If this is the case then user can just convert categorical to numerical and use the data. Providing internal encoding is kind of redundant.

#### Example:
- XYZforecaster has native support - the following runs successfully
- does not have native support -  categorical features are encoded with default encoder and used in fit/predict.
        
```python
from sktime import XYZforecaster

forecaster = XYZForecaster()
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```  

In case the user wants to treat a numerical columns as categorical, they have to convert it to categorical themselves before calling fit/predict (since no `cat_features` arg). We can make a transformer say 'Categorizer' to do this. So the workflow would look like this:
```python
from sktime import XYZforecaster, Categorizer

new_X_train = Categorizer(X_train) #This is NOT the design of categorizer, just a dummy
forecaster = XYZForecaster()
forecaster.fit(y_train, new_X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```  


---
---
### Design 2b).
- Same as Design 2a)(without `cat_features` parameter) but without fixing a default encoder, instead encoders as specified by the user are used when forecaster does not support natively.
- suggested methods to let user specify the encoders and columns
    - set_config 
    - As @yarnabrina suggested - Users can pass configuration for the forecaster to specify encoding choices, with a specific encoding method (say one hot encoding being default). It'll look like 
`forecast.set_config(encoding_configuration=...)`

 - The values can be of 2 types:

    1) single encoder instance -> use feature kind to identify categorical features
    2) dictionary with column names as keys and value as encoder instance

#### Example

Using set_config:
```python
from sktime import XYZforecaster

forecaster = XYZForecaster()
config_dict={'cat_feature1':LabelEncoder(), 'cat_feature2':OneHotEncoder()}
forecaster.set_config(enc_config=config_dict)
forecaster.fit(y_train, X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```  

#### PROS:
- Provides full freedom to user to select encoders and columns

#### CONS
- Not tunable params because of set_config
- Is implementing existing functionality(similar to `ColumnTransformer`) under the hood.

---
---


