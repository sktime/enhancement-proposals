# Generalized Pipeline and Graphpipeline

Contributors: benheid

## Contents

[TOC]


## Overview

### Problem Statement

We would like to introduce a new type of pipeline in sktime.

1. We would like to have a general pipeline instead of having pipelines for different tasks.
I.e. one pipeline that can contain classifier but also works with forecaster.
2. Based on that, we would like to introduce or enable the graphpipelineing concept as proposed in pyWATTS.

Initially proposed in sktime issues
* https://github.com/sktime/sktime/issues/4281

Relies on existing Step PR: 
* https://github.com/sktime/sktime/pull/4341

### The proposed solution for the Generalized Pipeline

The solution below introduces a new pipeline class that 
* inherits from BaseEstimator. 
* uses ducktyping for behaving like a specific class (e.g. Forecaster, Classifier, ...)
* uses inspection for determining the correct arguments for call, fit, etc. methods.

A prototype that describes the general idea is provided in the following PR:

### The proposed solution for the Graphpipeline
The proposed solution relies on the PR [sktime 4321](https://github.com/sktime/sktime/pull/4341). It may look as follows

```python
class Pipeline(BaseEstimator):

    def __init__(self, steps):
        super().__init__()
        self.steps =  steps

    @staticmethod
    def _check_validity(step, method_name, **kwargs):
        use_kwargs = {}
        if not hasattr(step, method_name):
            raise Exception(f"Method {method_name} does not exist for {step.__name__}")
        method = getattr(step, method_name)
        method_signature = inspect.signature(method).parameters

        for name, param in method_signature.items():
            if name == "self":
                continue
            if name not in kwargs and param.default is inspect._empty and param.kind != _ParameterKind.VAR_KEYWORD:
                raise Exception(f"Necesssary parameter {name} of method {method_name} is not provided")
            if name in kwargs:
                use_kwargs[name] = kwargs[name]
        return use_kwargs

    def fit(self, **kwargs):
        kwargs = deepcopy(kwargs)
        for transformer in self.steps[:-1]:
            required_kwargs = self._check_validity(transformer, "fit_transform", **kwargs)
            X = transformer.fit_transform(**required_kwargs)
            kwargs["X"] = X
        # fit forecaster
        required_kwargs = self._check_validity(self.steps[-1], "_fit", **kwargs)
        f = self.steps[-1]
        f.fit(**required_kwargs)
        return self

    def transform(self, *args, **kwargs):
        kwargs = deepcopy(kwargs)
        for transformer in self.steps[:-1]:
            required_kwargs = self._check_validity(transformer, "transform", **kwargs)
            X = transformer.transform(**required_kwargs)
            kwargs["X"] = X
        required_kwargs = self._check_validity(self.steps[-1], "transform", **kwargs)
        f = self.steps[-1]
        return f.transform(**required_kwargs)


    def predict(self, *args, **kwargs):
        kwargs = deepcopy(kwargs)
        for transformer in self.steps[:-1]:
            required_kwargs = self._check_validity(transformer, "transform", **kwargs)
            X = transformer.transform(**required_kwargs)
            kwargs["X"] = X
        required_kwargs = self._check_validity(self.steps[-1], "_predict", **kwargs)
        f = self.steps[-1]
        return f.predict(**required_kwargs)

    # All methods needs to be implemented.
```
#### Explanations of the code:
* The `__init__` gets the steps as argument, which are just assigned to the parameter `self.steps`.
  * Note for the GraphPipeline Solution, we probably need a additional add_step method, which can vary `self.steps`
* The fit/transform/predict/... methods have a similar design.
  * To take all possible arguments they have the `*args, **kwargs` parameters. These parameters are cloned to avoid side effects.
  * All transformers of the pipeline are executed (all steps before the last).
    Thereby, first the validity is checked. I.e., does the current step implement the transform method and are all required parameters available.
  * Call the transform method with the required arguments.
  * Afterwards, it is checked if all required arguments for the desired method of the last step are available. 
  * Finally, the desired method of the last step is called
* The `_check_validity` method checks for a specific estimator/transformer if the method is available and if all required parameters are in kwargs.
  Afterwards, it returns a dict containing all required parameters.
## Problem to solve

### Requirements

* the new pipeline should be compatible to all sktime estimators
* it should behave like an classifier, if a classifier is added in the pipeline, behave like a forecaster 
if a forecaster is added ...
* Code duplications should be reduced. I.e., we need to figure out if we can share code accross the different transform/predict/.. methods.


### Current implementation and workflow

Currently, for each estimator a separate pipeline implementation exist.

### Problems of the current soultion

TODO 

## Proposed solution

### User journey design

The user has only to import the pipeline regardless of the type of the used estimator.


#### User journey: general Pipeline Design

Taken from [Hackmd](https://hackmd.io/6PMsV6DLRIyCvwBMfKHuhw)
```python
p = Pipeline(steps, stuff)
p.fit(data, params)
p.method(more_data)
```

The general pipeline needs to be constructed with the steps and additional stuff as input.
Afterwards the fit method can be called with data and neeeded params.
Finally, a method can be called. The allowed methods are determined by the last step of the pipeline.


### Code design: General Pipeline

The general pipeline implements each fit/predict/transform method that is available in sktime. See example from above.

### Code design: Ducktyping if a method is allowed

If a method is called on the pipeline, the pipeline checks if the method is available. If not it will aise
an NotImplementException or something similar. 


### Code design: Use inspection for indentifying parameters.
If the method is available then pipeline uses inspection to determine which arguments has to be passed to the method of
the pipeline step. Afterwards it is checked if these parameters are available and then the method on the step is called.

## Graph Pipeline Code Design

