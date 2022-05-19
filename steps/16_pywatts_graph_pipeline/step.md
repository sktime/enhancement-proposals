# Integrating the pyWATTS' Graph Pipeline

Contributors: @kalebphipps, @fkiraly, @smeisen, @aiwalter, @benheid

## Introduction

Workflows in ML projects are often non-sequential. E.g., a regressor gets as input statistical features, calendar 
features, and historical values. All of these information are extracted from the same series or dataarray.

To realise such mappings via a pipeline, the pipeline has to be non-sequential.

[concise introduction to problem and overview of proposed solution]

For preliminary discussions of the proposal presented here, see issue: [links to issues/pull requests]

## Contents
[table of contents]

## Problem statement
[concise problem statement]
Workflows in ML projects are often non-sequential. E.g., a regressor gets as input statistical features, calendar 
features, and historical values. All of these information are extracted from the same series or dataarray.

To realise such mappings via a pipeline, the pipeline has to be non-sequential.


## Brief Description of the current solution in pyWATTS

pyWATTS solves this problem by proposing a graph pipeline. Based on this solution, we can aim to enable the usage of 
such pipelines in pyWATTS.

* Pipeline
  * Tasks
* Step
  * Tasks:
* Module
  * Tasks:
* StepInformation
  * Task

* How are the modules added to the pipeline
  * Functional API

* How is the data flow be realised
  * Step fetches the data and maintains a buffer

## Description of proposed solution

### Adding transformers, etc. to a graph pipeline
#### Requirements:
1. It has to be possible to add a transformer with multiple inputs
2. It has to be possible to use a transformer as input for multiple other transformers
3. Passing additional parameters to control the execution of the transformers etc.

#### Solution 1: Current Keras-Style API Solution in pyWATTS
The current solution in pyWATTS is as follows:
```python
pipeline = Pipeline()
step_information_1 = Transformer()(x=pipeline["bar"])
step_information_2 = Transformer()(x=step_information_1)
Transformer()(x_1=step_information_1, x_2=step_information_2, **additional_params)
pipeline.train(data)
```
Explanation of the example:
* `pipeline["bar"]` is a placeholder for the column `bar` in the dataset `data`.
* This example requires that the `__call__` dunder is implemented. This enables to call `Transformer()`
* `step_information` is the return value of the the `__call__` dunder. It contains information about the newly created 
step in the pipeline and the pipeline in which `Transformer()` is added.

The `__call__` dunder implementation could like as:

```python
   def __call__(self, pipe: GraphPipeline, stuff):
        
        return StepFactory().create_step(
                self,
                stuff
            )
```

#### Solution 2: Keras like API with GraphPipeline Object instead of StepInformation
Based on the solution above, the `StepInformation` could be replaced by an object which is 
itself a transformer etc.
```python
pipeline = Pipeline()
graph_pipeline1 = Transformer()(x=pipeline["bar"])
graph_pipeline2 = Transformer()(x=step_information_1)
graph_pipeline3 = Transformer()(x_1=step_information_1, x_2=step_information_2, **additional_params)
graph_pipeline3.train(data)
```
To implement this solution the ```__call__``` in base could look like:


Possible Graphpipeline implementations:

##### Possible implementation of `GraphPipeline` types: delegation
```python
class BaseForecaster(BaseObject):
    
    # etc

    def __call__(self, pipe: GraphPipeline, stuff):
        
        return GraphPipeline(
            StepFactory().create_step(
                self,
                steps,
                stuff
            )
        )
```


```python

class GraphPipeline(BaseObject):

    def __init__(self, steps, stuff):

        scitype = self._infer_scitype(steps)
        self._scitype = scitype

        if scitype == "forecaster":
            self.estimator = GraphPipelineForecaster(steps, stuff)
        elif etc


    @property
    def scitype(self):
        return self._scitype

    def _infer_scitype(self, steps):
        # infers based on graph/type structure

    # example method
    def predict(self, **kwargs):

        if _has_predict(self.scitype):
            return self.estimator.predict(**kwargs)
        else:
            raise TypeError(
                f"GraphPipeline has scitype {self.scitype}"
                f"and hence no predict method"
            )
```


##### Possible implementation B of `GraphPipeline` types: dispatch

```python

def make_graph_pipeline(steps, stuff):

    scitype = self._infer_scitype(steps)

    if scitype == "forecaster":
        return GraphPipelineForecaster(steps, stuff)
    elif etc
```

then in the `__call__` dunder

```python

class BaseForecaster(BaseObject):
    
    # etc

    def __call__(self, pipe: GraphPipeline, stuff):
        
        return make_graph_pipeline(
            StepFactory().create_step(
                self,
                steps,
                stuff
            )
        )
```

where `GraphPipeline` is an intermediate base or mixin class for `GraphPipelineForecaster`, `GraphPipelineClassifier`, etc

#### Solution 3: Full Keras Like API
In contrast to the above solutions, the pipeline can be created by specifying the inputs and output transformers. 
This could look as follows:

```python
step_information_1 = Transformer()(x=Column("bar"))
step_information_2 = Transformer()(x=step_information_1)
step_information_3 = Transformer()(x_1=step_information_1, x_2=step_information_2, **additional_params)

pipeline = Pipeline(inputs=[step_information_1], outputs=[step_information_3])
pipeline.train(data)
```
`Column("bar")` determines which column of the input data is used as input for the transformer.

Probably this can be combined with the GraphPipeline approach.


#### Solution 4: Removing the initialisation of the Pipeline
Perhaps, we can even remove the need to initialise a Python object:

```python
graph_pipeline_1 = Transformer()(x=Column("bar"))
graph_pipeline_2 = Transformer()(x=step_information_1)
graph_pipeline_3 = Transformer()(x_1=step_information_1, x_2=step_information_2, **additional_params)
graph_pipeline_3.train(data)

```
This solution is similar to Solution 2

### Data exchange format between transformers, ..


### 

