# Design Time Series Classification/Regression Module

Contributors: @RavenRudi

## Introduction
Regression and Classification are two very similiar tasks. Many already implemented time series classifiers can be used as regressors by solely changing a few lines of code. 

## Contents
[table of contents]

## Problem statement
Extending the time series regression module without a clear design specification will likely lead to boilplate code and inconsistent code design especially in the regression module.

### Example
The ROCKETClassifier can be easily modified such that it can be applied to regression tasks. A straight forward extension of the regression module with a ROCKETRegressor can be currently achieved in two steps:

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
The classifier module was refactored recently. Work was mainly done in the `BaseClassifier` class, which is basically a template for all classifiers. Many classifieres haven't adapted the new structure yet, further information on this can be found in [#1146](https://github.com/alan-turing-institute/sktime/issues/1146). The `BaseRegressor` has not yet been refactored, hence the structure of the regression module is not consistent with the classification module.

## Motivation
As the classification module is still being refactored and classification as well as forecasting module follow a similar design (see [#1146](https://github.com/alan-turing-institute/sktime/issues/1146) and [#912](https://github.com/alan-turing-institute/sktime/pull/912)), the motivation is to come up with a proposal that sticks to that design and applies it to the regression/classification module as a whole. Thereby boilerplate code can be avoided and a clear step by step description for the extension of the regression module could be provided.

## Description of proposed solution
The classification module is still under refactoring, hence the following design could be implemented in parallel and each time a classifier is extended to a regressor, the code could be refactored into this structure.
### Level 1
The `BaseTimeSeriesSupervisedLearner` serves as a basis for the entire classification/regression module. It is an abstract class which specifies the core API. 

```python
# Template for XXXBaseEstimator
class BaseTimeSeriesSupervisedLearner(BaseEstimator, ABC):
    
    def fit(self, X, y):

        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)

        X, y = check_X_y(X, y, coerce_to_numpy=coerce_to_numpy)

        # Implementation of _fit in XXXBaseEstimator
        self._fit(X, y)

        # this should happen last
        self._is_fitted = True

        return self

    @abstractmethod
    def _fit(self,X,y):
        pass

    def predict(self, X):

        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)

        X = check_X(X, coerce_to_numpy=coerce_to_numpy)
        self.check_is_fitted()

        # Implementation of _predict in XXXBaseEstimator
        y = self._predict(X)

        return y

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod 
    def score(self, X, y):
        pass
```

### Level 2
The XXXBaseEstimator inherits from `BaseSupervisedLearner`. These classes contain the core logic of a model (e.g. ROCKET), i.e they implement the missing fit and predict functionality. Regression and classification require slightly different implementations here and there (e.g see example above). Hence it should be possible to identify whether the object is a regressor or a classifier (sklearn seems to do that in a similar way, see [here](https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/tree/_classes.py#L88)), such that _fit and _predict can provide the correct functionality.  

```python
class XXXBaseEstimator(BaseTimeSeriesSupervisedLearner, metaclass=ABC):

    def _fit(self, X, y):
        # Concrete Implementation here
        # also somehow check if instantiated by regressor or classifier
    
    def _predict(self, X):
        # Concrete Implementation here, also check if regressor or classifier
```

Moreover the classes `BaseRegressor` and `BaseClassifier` specify further functions the respective model class should offer, e.g classifiers have to have `predict_proba()`.

```python
class BaseRegressor(ABC):
    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))

    # Any other functions for this class? 
```

```python
class BaseClassifier(ABC):

    def predict_proba(self, X):
        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)

        X = check_X(X, coerce_to_numpy=coerce_to_numpy)
        self.check_is_fitted()
        return self._predict_proba(X)

    @abstractmethod
    def _predict_proba(self, X):
        pass

    
    def score(self, X, y):

        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), normalize=True)
```

### Level 3
All classes up to Level 2 are abstract. The concrete classes which are exposed to the user and therefore not abstract are `XXXRegressor` and `XXXClassifier`.
```python
class XXXRegressor(BaseRegressor, XXXBaseEstimator):
    ## RegressorMixin is sklearn implementation and provides score function
    def __init__(*args):
        # only __init__ should be necessary in the most of the cases


class XXXClassifier(BaseClassifier, XXXBaseEstimator):
    ## ClassifierMixin is sklearn implementation and provides score function
    def __init__(*args):
        # only __init__ should be necessary in the most of the cases

    def predict_proba(self, X):
        # Implement here
```

## Discussion and comparison of alternative solutions
While `BaseClassifier` is somewhat necessary due to `predict_proba()`, `BaseRegressor` is not really. Also these two classes are not the real "Base" of the model, they can be rather considered as some sort of Mixin. The real base is the `BaseSupervisedLearner` class. 

Moreover one might argue that this suggests refactoring a module that was just refactored. The refactor [#1146](https://github.com/alan-turing-institute/sktime/issues/1146) is an excellent foundation to implement this proposal, as it already uses `fit`, `_fit` and `predict`, `_predict`. Creating a regressor from a classifier will require the creation of a base class for the specific model to avoid boilerplate code anyway. Hence I would suggest to use this structure, whenever a new regressor is added.