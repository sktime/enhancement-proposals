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


## Requirements

### Minimal Functional Requirements

* The graphpipeline should cover the functionality of:
  * TransformedTargetForecaster linear pipeline
  * ForecastingPipeline linear pipeline
  * TransformerPipeline
  * FeatureUnion and variable subsetting
* Additionally, a simple “triangle” graphical situation should be realised in the minimal version
  * This requires the management of the execution order. May use the resolution order from pyWATTS.
* The minimal graphpipeline has only one output step, which can be a forecaster.
* The minimal graphpipeline supports only one forecaster.

These three requirements cover the following aspects:
* The implementation needs to be polymorph with regard to transformer and forecaster
* simple truly graphical situations beyond current sktime functionality are covered (triangle)


#### General requirements


* the new pipeline should be compatible to all sktime estimators
* it should behave like an classifier, if a classifier is added in the pipeline, behave like a forecaster 
if a forecaster is added ...
* Code duplications should be reduced. I.e., we need to figure out if we can share code accross the different transform/predict/.. methods.


## Proposed solution

### User journey design
The user has only to import the pipeline regardless of the type of the used estimator.


Taken from [Hackmd](https://hackmd.io/6PMsV6DLRIyCvwBMfKHuhw)
```python
p = Pipeline(stuff)
p.add_step(...)
...
p.add_step(...)
p.fit(data, params)
p.method(more_data)
```
* **Constructor of the Pipeline (`__init__`)** 
  * The constructor creates the pipeline object it accepts the following arguments
    * A list of steps/step_informations of the pipeline that can fully describe a pipeline. 
      However, probably **should not** this. This is mainly required to be compatibe to sklearn.
      Since the pipeline steps are pipeline parameters. 
    * Additional parameters as store paths for intermediate results (at least pyWATTS uses such information.)
* **add_step** method.
  * The `add_step` method adds a transformer/forecaster to the pipeline. Thereby, it mutates the state of the pipeline.
    Under the hood, the graph structure is maintained by having a Step/StepInformation object that contains the added
    transformer/forecaster together with the links to its' predecessors.
    It requires the following arguments:
    * The transformer/forecaster which should be add to the pipeline.
    * The name that this step should have within the pipeline.
    * A list of predessors, identified by the keys. 
    * Further kwargs, that can specify the behaviour of the transformer/forecaster in the pipeline.
* **fit** method
  * Fits everything within the pipeline.
* **method** stands for every method that is possible on the pipeline. The possible methods of the pipeline a determined by its last step.

The pipeline object is created, and then with `add_step` the transformer/forecaster/.. are added to the pipeline.
Afterwards, fit can be called, and then for example `predict` if the last step in the pipeline has a `predict` method.


#### Comparison to existing solutions
##### Compare to make_pipeline
##### Compare to dunders

## Code design

The solution below introduces a new pipeline class that 
* inherits from BaseEstimator. 
* uses ducktyping for behaving like a specific class (e.g. Forecaster, Classifier, ...)
* uses inspection for determining the correct arguments for call, fit, etc. methods.

A prototype that describes the general idea is provided in the following PR:

#### TODOS
* [ ] Exemplarily flowchart, how the pipeline works
  * [ ] Adding steps
  * [ ] Fitting/Transforming/Predicting the pipeline (Resolution Order)
    * How to determine whether transformers are pre or post. 
    * How to determine transformer outputs “skipping” the forecaster, e.g., a lagger
* [ ] Type inference (is the pipeline a transformer or a forecaster?)

The general pipeline implements each fit/predict/transform method that is available in sktime. See example from above.

### Code Design: Graphpipeline
The proposed solution relies on the PR [sktime 4321](https://github.com/sktime/sktime/pull/4341). It may look as follows
* [ ] Structure of the proposed pipeline class (public methods, parameters, covered functionality)
* [ ] Add a add_step method in the code below. Change to more pseudo-code instead of correct python code.
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
    
    def add_step(self, skobject, name, edges, **kwargs):
        # TODO Fill out
        pass

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

#### Further Supportive Classes/Methods and their code
* [ ] List and Structure of supportive classes as Steps, StepInformation, etc.


* **Step** class


#### How is determined if a method is allowed to be called on the pipeline. I.e. what is the type of the pipeline.
If a method is called on the pipeline, the pipeline checks if the method is available. If not it will raise
an NotImplementException or something similar. 
The allowed methods are specified by the type of the last element of the pipeline. I.e., if the last element is a transformer,
`transform` is allowed. If the last element is a forecaster, then `predict`, `predict_quantile`, ... are allowed.  

Under the hood, this is determined by using ducktyping as in the following code snippet.

````python
if not hasattr(last_step, method_name):
    raise Exception(f"Method {method_name} does not exist for {last_step.__name__}")
method = getattr(last_step, method_name)
````
hereby, `last_step` is the last element in the pipeline, since it determines the allowed methods. 
Furthermore, `method_name` is the name of the method that should be called, e.g. `predict`. 
The result of the `getattr` method is the requested method.


To identify the allowed arguments of the method, we use inspection. 
More specifically, first, we determine which parameters the method needs via inspection.
Afterwards, we check if these parameter are provided to the method that is called.
The code for this might look as follows:

```python
method_signature = inspect.signature(method).parameters

for name, param in method_signature.items():
    if name == "self":
        continue
    if name not in kwargs and param.default is inspect._empty and param.kind != _ParameterKind.VAR_KEYWORD:
        raise Exception(f"Necesssary parameter {name} of method {method_name} is not provided")
    if name in kwargs:
        use_kwargs[name] = kwargs[name]
```
First, with inspect we get a dict of all arguments which the method has.
Second, we iterate through this dict and make the following checks:
* ignore the `self` argument
* raise an exception if an required argument is not provided. 
The argument is required if it's default value is empty 
and if it is not  key word argument.
* store the argument in a dict of the arguments `use_kwargs` that should be used with the name of the parameter as key. 

The dict `use_kwargs` is then passed to the method that should be called using the double star (`**use_kwargs`).

#### How is the graph represented within in the pipeline
The pipeline has a Directed Acyclic Graph (DAG) structure. DAGs are storable by backward links. 
Thus, we have an intermediate layer (class `step`), such a step is created with each call of `add_step`.
The step stores the transformer/estimator added by `add_step`, together with links to all predecessors.
The pipeline itself needs to dissolve the graph structure the information which steps have no sucessors.


### Code Design: What will be reused
* skBASE
* pyWATTS resolution order

#### Interface Compatibility
* [ ] How does the interfaces comply to sklearn/sktime
* [ ] How can the pipeline be tested efficiently

## Comparison to current linear pipelines

##### TODOs 
* [ ] 1:1 comparison of the same example. Perhaps the example from the pyData global.
* [ ] Architectural Differences. 

### Current implementation and workflow

### Problems of the current soultion
* Currently, for each estimator a separate pipeline implementation exist.
* Furthermore, graph pipelines cannot be realised

## Implementation plan
* logical units
* can be worked on separately?
* tests of sub-units
  * interface compatibility
* re-use or import of existing functionality
* TDD or partial TDD using the simple examples

## Examples that are used for partial tdd
* [ ] List of examples 
