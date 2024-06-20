# Adding categorical support to sktime

Contributors: @Abhay-Lejith

## Introduction

### Objectives 
Allow users to pass exog data with categorical features to any forecaster. Outcome will depend on whether the forecaster natively supports categorical or not.
Two broadly possible outcomes:
- error in fit saying not natively supported.
- internally encode and fit successfully.

### Current status:
* Non-numeric dtypes are completely blocked off in datatype checks.
* https://github.com/sktime/sktime/pull/5886 - This pr removes these checks but has not been merged yet as most estimators do not support categorical and will break without informative error messages.
* https://github.com/sktime/sktime/pull/6490 - pr adding feature_kind metadata


### Models which support categorical natively that have been identified so far:
- xgboost
- lightgbm
- catboost

Note that all these will have to be used via reduction.

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

### Example 1: (Reduction case)
```python
from sktime.forecasting.compose import make_reduction
from catboost import CatBoostRegressor

regressor = CatBoostRegressor()
forecaster = make_reduction(regressor)
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```

### Example 2:
```python
from sktime import XYZforecaster

forecaster = XYZForecaster()
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```

## Requirements
1. Need to know whether categorical data is present in X_train and which columns.
2. Need to know if forecaster has native categorical support.
3. In reduction, we need to know if model used has native categorical support.
4. In reduction, we need to pass parameters(like `enable_categorical=True` in case of xgboost) to the model used if natively supported according to the requirements of the model.
5. Must handle all combinations of cases
    - estimator natively supports categorical: yes/no
    - categorical feature specified by user: yes/no
    - categorical present in data: yes/no

    Important case is when categorical data is passed but is not natively supported.


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
Discussed according to the different design options below.

---
---

### Design 1. Allow categorical features ONLY in those forecasters which support natively. (no encoding internally)

- `categorical_features` parameter will be present only in forecasters that support natively.
- For forecasters that do not support natively, users can make a pipeline with an encoder of their choice and use categorical variables.
    
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
| NO | NO | YES| informative error message to user that categorical feature is not supported by the forecaster |
| NO | YES| NO | X |
| NO | YES| YES| X |
| YES| NO | NO | normal case, fits without error. |
| YES| NO | YES| use categorical columns as detected by feature_kind and proceed to fit. |
| YES| YES| NO | informative error message that specified column was not present in data.|
| YES| YES| YES| use categorical columns as detected by feature_kind along with the user specified columns(union) and proceed to fit. |

- in case where specified column was not found, we can also choose to ignore that column and continue to fit with warning that column was not found.

#### PROS:
- No need to add `categorical_feature` parameter to all models.

#### CONS:
- Less number of estimators support categorical natively.


---
---

### Design 2. Perform some encoding method as default if not natively supported. 

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

---
---

### Design 3. Same as design 2 but NO `cat_features` parameter in model
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

---
---
### Design 4.
- Same as Design 2 or 3 but without fixing a default encoder.
    - How much choice do we want to give user?
        - Which encoders to use?
        - Which columns to encode?
        - Which encoder to use on which column?
    - This is basically trying to provide the entire encoding ecosystem internally.
    - Will be very difficult to implement. Where will the user specify all this data? 

- I don't think this is a good idea.

---
---


