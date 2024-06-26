# Adding categorical support to sktime

Contributors: @Abhay-Lejith

## Introduction

### Objectives 
Allow users to pass exog data with categorical features to any forecaster. Outcome will depend on whether the forecaster natively supports categorical or not. In any case, ambiguous errors must not be raised, and all cases must be handled appropriately.

### Current status:
* Non-numeric dtypes are completely blocked off in datatype checks.
* https://github.com/sktime/sktime/pull/5886 - This pr removes these checks but has not been merged yet as most estimators do not support categorical and will break without informative error messages.
* https://github.com/sktime/sktime/pull/6490 - pr adding feature_kind metadata

### Conceptual model:

Note: For this proposal, I am limiting the scope to categorical in `exogeneous` features in forecasters.
1. Data passed to a forecaster in fit/predict may include exogeneous(X)data along with endogeneous y.
2. The exogeneous data may contain categorical variables.(i.e non-numeric dtype). There may also be numeric columns to be treated as categorical as per the user.
3. Manner of specification of such features may vary depending on the design.
4. If the forecaster natively supports categorical, proceed to fit as usual. If not supported, it needs to be handled accordingly so that categorical data is not fed into the model causing uninformative errors internally.

5. In reduction cases, categorical support is determined by the model used in reduction.
6. Further, different libraries (xgboost, catboost etc) have different expectations/specifications and these need to be followed in reduction.



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
2. Need to know if forecaster has native categorical support.
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
- HistGradientBoostingRegressor (sklearn)

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
We will use a tag `capability:categorical_input`(=True/False) or similar to identify whether a forecaster has native support.

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

Rest of the design will depend on this decision.

1) Raise error that categorical is not natively supported
2) Default handling of categorical:
    - a) drop categorical columns with warning(that they were dropped because of lack of native support)
    - b) encode with fixed default encoder
3) Handling based on user choice
i.e.user can specify choice of encoder and on which columns to apply

for the above cases there are even more combinations based on which solution we take for the subchallenge mentioned under requirement 1.(cat_features arg, no arg, set_config)
There are too many designs due to the large combinations of solutions possible and they vary greatly depending on the above broad decision we take.

So I suggest taking a call on which of the above designs we want to go ahead with and then go into the finer details.

I have listed the specifics with pros/cons and examples for a few combinations below so we have an idea of what is to be expected from the design options.

---
---

### Design 1) Raise error that categorical is not natively supported. (no encoding internally)

- `categorical_features` parameter will be present ONLY in forecasters that support natively.
    
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

- *for design 2a) we will instead drop the categorical column and proceed to fit, everything else remains the same.
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
*In design 2a) we will instead drop the categorical columns and proceed to fit/predict.

---
---

### Design 2b). Perform some encoding method as default if not natively supported.

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
- does not have native support - internally, categorical features are encoded and used in fit/predict.
        
```python
from sktime import XYZforecaster

forecaster = XYZForecaster(cat_features=['cat_feature'])
forecaster.fit(y_train,X_train)
y_pred = forecaster.predict(X_test,fh=[1,2])
```

---
---

### Design 2b). default encoding but NO `cat_features` parameter in model
- similar to 2a) but without the `cat_features` arg.
- Instead, user is required to convert the required numerical features to categorical. To ease the process, we can make a `transformer` to make this conversion.

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
### Design 3). Handling based on user choice

- There will still be a default encoding method which is applied on categorical columns detected by feature_kind, but this can be overridden by user. If nothing is specified by the user, will use default.
- suggested methods to let user specify the encoders and columns
    - set_config 
    - As @yarnabrina suggested - Users can pass configuration for the forecaster to specify encoding choices, with a specific encoding method (say one hot encoding being default). It'll look like 
`forecast.set_config(encoding_configuration=...)`

 - The values can be of 2 types:

    1) single encoder instance -> use feature kind to identify categorical features
    2) dictionary with column names as keys and value as encoder instance

| **Estimator supports categorical natively** |**User has specified handling** |**Cat feature present in data** |  **Outcome**  |
|:-------------------------------------------:|:--:|:-------------------------------:|:---------------:|
| NO | NO | NO | normal case, fits without error. |
| NO | NO | YES| encode features by default handling and continue.
| NO | YES| NO | error that specified column was not present in data/ignore with warning |
| NO | YES| YES| encode features as per user specification|
| YES| NO | NO | normal case, fits without error.|
| YES| NO | YES| use categorical features detected by feature_kind and fit|
| YES| YES| NO | error that specified column was not present in data/ignore column with warning|
| YES| YES| YES| encode features as per user specification|

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
The user specified handling is performed and then continue to fit. If user did not specify encoder to use for some categorical columns say 'cat_feature3', then we use the default on it.

#### PROS:
- Provides full freedom to user to select encoders and columns
- no changing of signature of forecasters

#### CONS
- Not tunable params because of set_config
- Is implementing existing functionality(similar to `ColumnTransformer`) under the hood.

Here we can also remove one con but at a cost:
- instead of set_config pass the dict as an argument to the forecaster.
- now it is tunable
- but all forecaster signatures will get modified.

Note: In this design, the user is allowed to specify encoders for the columns, but if there is no restriction, they can specify transformers as well(like minmaxscaler etc).
This would be an additional ability we are adding to all estimators.


---
---


