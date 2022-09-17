# BaseDeepNetwork

Contributors: ['AurumnPegasus']

## Introduction

Each DL Estimator primarly have two parts: Network, and Estimator. The network is the class containing the core keras code for the DL Estimator, where we build our main keras network. Estimator class is simply a layer of abstraction for the user for easy interaction with the keras network.
For different DL estimators, the only thing that mainly changes is the `network` (since most abstractions are structurally the same). Hence, having a common Base Class for DL models would make sense, as it reduces redundancy of code, and allows for a common testing interface.

For preliminary discussions of the proposal presented here, see issue: [#3190](https://github.com/alan-turing-institute/sktime/issues/3190)

## Contents

[TOC]

## Problem statement

The current implementation considers `Network` and `Estimator` to be two different entities, which interact with each other via objects. For example, as shown in the figure below: If I were to create a `CNNClassifier`, it creates an object of `CNNNetwork` within it, from which I call the `build_network` function to get my keras network to be used in `CNNClassifier`.
Within the estimator, if I want to create a `CNNClassifier`, I inherit from `BaseDeepClassifier` (which contains lots of common functions for all DL Classifiers). The issue with this is that there will exist a `BaseDeepRegressor` and `BaseDeepForecastor` as well, which will be specific to Regressors and Forecastors respectively. These will lead to lot of redundant code, since there are lot of functionalities which are the same across all DL Estimators.
Hence, we need to design a `BaseDeepClass`, which will contain all code and structure which needs to be inherited by every DL estimator.

### Current Implementation

![](https://i.imgur.com/1x2IjJv.png)

- Legend:
    - Green: `sktime` Base Class
    - Blue: `sktime` child classes
    - Red: `sklearn` Base Class

`BaseDeepNetwork` is a base class for creating keras networks. Each specific neural network is child of the `BaseDeepNetwork`. For example, `CNNNetwork`, `CNTCNetwork`, `LSTMNetwork` etc would be classes inheriting from the `BaseDeepNetwork` having a single function called build_model (which builds and returns the created keras network)

```python
class BaseDeepNetwork(BaseObject, ABC):

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        # Creates keras networks and returns input and output layers
```

For estimators, there exist specific `BaseDeepClassifier` and `BaseDeepRegressor`, inheriting from `BaseClassifier` and `BaseRegressor` respectively. 
Specific estimators like `CNNClassifier` inherit from `BaseDeepClassifier`, and within the `init` method create an object of the class `CNNNetwork`. Then, in the fit method, it gets the respective keras neural network by calling `build_network` method from the `CNNNetwork` object.

```python
class CNNClassifier(BaseDeepClassifier):
    def __init__(self):
        # creates object of network class
        self._network = CNNNetwork()

    def build_model(self):
        # gets network from created object of the class
        input_layer, output_layer = self._network.build_network()

        # additional dense layer on top of output layer
        output_layer = keras.layers.Dense()(output_layer)

        model.compile()
        return model

    def _fit(self, X, y):
        # gets the compiled model here
        self.model_ = self.build_model()
```

### Problems

1. Repeated code across `BaseDeepClassifier` and `BaseDeepRegressor`
    - Since there is no common `BaseDeepClass` (and `BaseDeepNetwork` is just related to the networks), there are functions which are repeated across everything
        - `build_model`: Is the same across all classifiers and regressors (only difference is `n_classes` for classifiers, the value of which is 1 for regressors, but since it is a parameter it can be considered the same)
        - `fit`: Is almost the same across all classifiers and across all regressors
        - `save`: function when merged with main [#3128](https://github.com/alan-turing-institute/sktime/pull/3128) will be common as well.
2. No dedicated testing suite for Deep Learning networks:
    - One of the ideas discussed was to introduce tests specific to DL networks
    - Test to see if saved and loaded model give the same answer
    - Writing pytest with different parameters (currently afaik DL models arent tested with different parameters at all)
    - Test to see if loss is reducing over epochs

## Alternative Solution 1

Lets say I want to create `CNNClassifier`. So the first step here would be to define `CNNNetwork` and `BaseDeepNetwork` with there structure.

![](https://i.imgur.com/4uRJrYr.png)

---

In this solution, we use `BaseDeepNetwork` as a base class in which we define all functionalities similar across all keras network (for eg: save, load etc)

So, our `BaseDeepNetwork` would end up looking like (The proper defined code can be seen in `Solution_1/basedeepnetwork.py`):

```python
class BaseDeepNetwork(BaseObject, ABC):

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        ...

    def build_model(self, input_shape, n_classes, **kwargs):
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )
        return model
```

Here, I am defining `BaseDeepNetwork` in a way which will give basic structure for any specific network down the line. For example, now if I want to create `CNNNetwork`, I just need to overwrite the `build_network` function of `BaseDeepNetwork` (since other functions relevant to keras network have already been defined).  (The proper defined code can be seen in `Solution_1/cnnnetwork.py`)

```python
class CNNNetwork(BaseDeepNetwork):
    def __init__(
        self,
    ):
        pass

    def build_network(self, input_shape, **kwargs):
        conv = keras.layers.Conv1D(
            filters=self.filter_sizes[0],
            kernel_size=self.kernel_size,
            padding=padding,
            activation=self.activation,
        )(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
```
---

Now, once we have created the structure for `Network`, we need to integrate it with the `Estimators`. Let's say I want to create `CNNClassifier` and `CNNRegressor`:

The structure of `BaseDeepClassifier` (`BaseDeepRegressor` will be similar) is slightly different now. Earlier, `BaseDeepClassifier` used to have all the functions and code related to DL Classifiers and DL Networks in it, but now I have kept code related to `network` in `BaseDeepNetwork`, so the `BaseDeepClassifier` will only have code specific to DL Classifiers. (The complete code is there in `Solution_1/basedeepclass.py`)

```python
class BaseDeepClassifier(BaseClassifier):

    def __init__(self, batch_size=40, random_state=None):
        super(BaseDeepClassifier, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def summary(self):
        return self.history.history

    def _predict(self, X, **kwargs):
        pass

    def _predict_proba(self, X, **kwargs):
        pass
```

Previously, we would have put the functions related to `save` or `load` in `BaseDeepClassifier`, but since they are specific to keras network, we leave it out from `BaseDeepClassifier` now.

To see how we integrate `CNNNetwork` with `CNNClassifier`, we define:

```python
class CNNClassifier(BaseDeepClassifier, CNNNetwork):

    def __init__(
        self,
        args
    ):
        _check_dl_dependencies(severity="error")
        super(CNNClassifier, self).__init__()

    def _fit(self, X, y):
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        self.history = self.model_.fit(
            X,
            y_onehot,
            args
        )
        return self
```

The complete version of this code is written in `Solution_1/cnnclass.py`. The difference from the original structure to this is that here, we inherit from `BaseDeepClassifier` (to preserve the structure required from Classifiers, as well as have some specific functions related to DL Classifiers) and `CNNNetwork` (to preserve structure required from all DL Networks)

---


Pros:
1. `CNNNetwork` becomes deeply integrated to the appropriate classifiers
2. There are 2 different base classes for DL estimators
    a. `BaseDeepNetwork`: (parent of `CNNNetwork`) which is more of a base class for the networks themselves, than the estimators. A lot of code which will be similar for all networks can be kept here, for example: summary, save. 
    b. `BaseDeepClassifier`: Here, code which is specific to all DL classifiers can be kept


## Alternative Solution 2

Lets say I want to create `CNNClassifier`. In this solution, I do not propose any significant changes in the structure of `BaseDeepNetwork` or `CNNNetwork`, they largly remain the same.

---

To go around the problem of writing functions specific to `networks` in all `BaseDeep` classes, we create a parent `BaseDeepEstimator` (a complete idea is there in `Solution_2/baseest.py`)

```python
class BaseDeepEstimator(BaseEstimator):

    def __init__(self, batch_size=40, random_state=None):

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def summary(self):
        return self.history.history

    def convert_y_to_keras(self, y):
        pass

    def save(self, path):
        pass
```

The `BaseDeepEstimator` will have all network specific code which has to be present in all DL Estimators. The whole idea is that to use `BaseDeepNetwork` as a seperate entity just for the creation of `CNNNetwork`, and take care of all redundancies of code in `BaseDeepEstimator`

Based on this, we re-define our `BaseDeepClassifier` to inherit functions/structure of Classifier from `BaseClassifier` and functions/structure of DL Network from `BaseDeepEstimators`. The complete idea is there in `Solution_2/basedeepclass.py`:

```python
class BaseDeepClassifier(BaseClassifier, ABC, BaseDeepEstimator):

    def __init__(self, batch_size=40, random_state=None):
        super(BaseDeepClassifier, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, n_classes, **kwargs):
        ...


    def _predict(self, X, **kwargs):
        pass

    def _predict_proba(self, X, **kwargs):
        pass

```

Here, the `CNNClassifier` largly remains unchanged as well, since the structure of it doesnt go through a huge change. 

---

Pros:
1. Do not have to change anything for any of the Regressors or Classifiers.
2. Minimal changes required imply it is the easiest to implement, with no real complications
3. Existing issue of re-writing code across `BaseDeepClassifier`, `BaseDeepRegressor` (and in the futere, `BaseDeepForecastor`) will be solved using a common parent `BaseDeepEstimator`.

