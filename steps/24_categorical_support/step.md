# Adding categorical support to sktime

Contributors: @Abhay-Lejith

## Introduction

### Objectives 
Allow users to pass data with categorical features to any estimator. If not supported, return an informative error message. Otherwise it should run without errors.

### Current status:
* Non-numeric dtypes are completely blocked off in datatype checks.
* https://github.com/sktime/sktime/pull/5886 - This pr removes these checks but has not been merged yet as most estimators do not support categorical and will break without informative error messages.
* https://github.com/sktime/sktime/pull/6490 - pr adding feature_kind metadata


## Description of proposed designs


For each design we will have to discuss 8 combinations of cases:
1. estimator natively supports categorical: yes/no
2. categorical feature specified by user: yes/no
3. categorical present in data: yes/no
    
## Specifying categorical features:
- The reason I proposed this is in case the user wants to treat a column with numerical values as categorical. This will not be detected by feature_kind as it relies only on the dtype.
- More on how this is tackled in each design below.
---
---


### Design 1. Allow categorical features ONLY in those estimators which support natively. (no encoding internally)

- `categorical_features` parameter will be present only in estimators that support natively.
- For estimators that do not support natively, users can make a pipeline with an encoder of their choice and use categorical variables. This is the general ml workflow, like how it's done in sklearn too.
    
    #### How tag is going to be used
    - in datatype checks :
    ```
    if categorical support tag is false:
        if categorical feature is present:
            error - cat feature is not supported by this estimator
    ```       

- In this case `categorical feature` will not be a parameter in models that do not support. So (NO,YES,_) is not possible.    

| **Estimator supports categorical natively** | **Cat feature specified by user** | **Cat feature present in data** | **Outcome** | 
|:-----------------:|:-------------------------:|:--------------:|:---------------:|
| NO | NO | NO | normal case, proceeds as usual without error. |
| NO | NO | YES| informative error message to user that categorical feature is not supported by the estimator |
| NO | YES| NO | X |
| NO | YES| YES| X |
| YES| NO | NO | normal case, proceeds as usual without error. |
| YES| NO | YES| use categorical columns as detected by feature_kind and proceed. |
| YES| YES| NO | informative error message that specified column was not present in data.|
| YES| YES| YES| use categorical columns as detected by feature_kind along with the user specified columns(union) and proceed. |

- in case where specified column was not found, we can also choose to ignore and continue with warning that column was not found.

#### estimators that can be added:
- xgboost, lightgbm and catboost are some models that have native support for categorical. But method of passing categorical features varies quite a lot between them (I've commented on this in umbrella issue - https://github.com/sktime/sktime/issues/6109). Making a general interface might be challenging.
- So it might be better to interface them in separate classes inheriting from reduction forecaster class. (similar to darts lightgbm, see - https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html)

#### PROS:
- No internal encoding, comparatively easier to implement.
- No major changes to core logic.
- Follows general ML workflow i.e if estimator does not support, then user must preprocess their data(including encoding) and then use it.
#### CONS:
- Less number of estimators support categorical natively.


---
---

### Design 2. Perform one hot encoding if not natively supported. 

- `categorical_feature`  param will be present in all models.

    #### How tag is going to be used:
    ```
     if cat support tag is false:
         encode cat features and continue
     else:
         continue
    ```

| **Estimator supports categorical natively** | **Cat feature specified by user** | **Cat feature present in data** | **Outcome** | 
|:-----------------:|:-------------------------:|:--------------:|:---------------:|
| NO | NO | NO | normal case, proceeds as usual without error. |
| NO | NO | YES| encode features internally as detected by feature_kind and continue.
| NO | YES| NO | error that specified column was not present in data |
| NO | YES| YES| use categorical features detected by feature kind and those specified by user(union) and proceed to encode |
| YES| NO | NO | normal case, proceeds without error.|
| YES| NO | YES| use categorical features detected by feature_kind and proceed|
| YES| YES| NO | error that specified column was not present in data|
| YES| YES| YES| use categorical features detected by feature kind and those specified by user(union) and proceed|

#### Estimators that will be affected:
- all estimators that take exog input

#### PROS:
- Most estimators will be able to accept categorical inputs.
- If user only wants to one-hot encode, this design is easier as user does not have to do anything extra.(encoding is done internally)

#### CONS:
- Will be more complex to implement, may require significant changes to core logic (for internal encoding).
- adding `cat_features` parameter to all estimators is not desired
- Not flexible in choice of encoding (label,one-hot,etc). If user wants different types, will have to resort to externallly doing it anyway.

---
---

### Design 3. Same as design 2 but NO `cat_features` parameter in model
Instead, user is required to convert the required numerical features to categorical. To ease the process, we can make a `transformer` to make this conversion.

- no `cat_features` parameter in models.

| **Estimator supports categorical natively** | **Cat feature present in data** |  **Outcome**  |
|:-------------------------------------------:|:-------------------------------:|:---------------:|
| NO |  NO | normal case, proceeds as usual without error. |
| NO |  YES| encode features internally as detected by feature_kind and continue.
| YES|  NO | normal case, proceeds without error.|
| YES| YES| use categorical features detected by feature_kind and proceed|

#### PROS:
- Most estimators will be able to accept categorical inputs.
- If user only wants to one-hot encode, this design is easier as user does not have to do anything extra.(encoding is done internally)
- Removes need for adding extra parameter to all models.

#### CONS:
- Will be more complex to implement, may require significant changes to core logic (for internal encoding).
- Not flexible in choice of encoding (label,one-hot,etc). If user wants different types, will have to resort to externally doing it anyway.
- Removing parameter comes at cost of requiring user to convert numerical to categorical themselves. 
    - If this is the case then user can just convert categorical to numerical and use the data. Providing internal encoding is kind of redundant.

---
---

## Personal preference - 
- Design 1: 
- I see no need for the encoding to be done internally. Preprocessing data is a regular part of the ml workflow and encoding categorical features is part of it. If users want to use cat features in estimators that do not support natively, then they can encode it themselves externally or in pipelines and then pass it to the estimator. This is how it is everywhere else, including sklearn.
- Additionally, even if we are providing internal encoding, it is not flexible and provides only a single option to the user. If user wants to use different encoding on diff columns, will have to resort to doing it externally anyway. 


