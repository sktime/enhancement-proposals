# Design Time Series Classification/Regression Module

Contributors: @RavenRudi

## Introduction
Regression and Classification are two very similiar tasks. Many already implemented time series classifiers can be used as regressors by solely changing a few lines of code. 

## Contents
[table of contents]

## Problem statement
Extending the time series regression module without a clear design specification will likely lead to boilplate code and inconvenient extension possibilites.

### Example
The ROCKETClassifier can be easily modified such that it can be applied to regression tasks. A straight forward extension of the regression module with a ROCKETRegressor can be achieved in two steps at the moment:

**STEP 1**:
Copy the RocketClassifier class.
```python
class ROCKETClassifier(BaseClassifier):

    # This is just an excerpt from the class

    def _fit(self, X, y):
        
        # This is just an excerpt from the method

        self.classifier = rocket_pipeline = make_pipeline(
            Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
        rocket_pipeline.fit(X, y)

        return self
```
**STEP 2**:
Create `ROCEKTRegressor`. Mainly requires to change `RidgeClassifierCV` to `RidgeRegressorCV`.

```python
class ROCKETRegressor(BaseRegressor):

    # This is just an excerpt from the class

    def fit(self, X, y):
        
        # This is just an excerpt from the method

        self.regressor = rocket_pipeline = make_pipeline(
            Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            RidgeRegressorCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
        rocket_pipeline.fit(X, y)

        return self
```
### Questions
* In terms of ROCKET, the difference between classifier and regressor is very small. Does this also apply for other models? 
* Is it possible to group models according to their classifier-regressor relationship?

### Current work on the module
The classifier module has been refactored recently. Work was mainly done in the `_BaseClassifier` class, which is basically a template for all classifiers. Many classifieres haven't adapted the new structure yet, further information on this can be found in [#1146](https://github.com/alan-turing-institute/sktime/issues/1146). The `_BaseRegressor` has not yet been refactored, hence the structure is of the regression module is not consistent with the one of the classification module.

## Goal
As the classification module is still being refactored and classification as well as forecasting module follow a similar design (see [#1146](https://github.com/alan-turing-institute/sktime/issues/1146) and [#912](https://github.com/alan-turing-institute/sktime/pull/912)), the goal is to come up with a proposal that sticks to that design, however applies it to the regression/classification module as a whole



## Description of proposed solution
## Motivation

## Discussion and comparison of alternative solutions

## Detailed description of design and implementation of proposed solution 
[prototypical implementation if applicable]
