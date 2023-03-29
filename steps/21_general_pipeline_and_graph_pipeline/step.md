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
*

Relies on existing Step PR: ..

### The proposed solution for the Generalized Pipeline

The solution below introduces a new pipeline class that 
* inherits from BaseEstimator. 
* uses ducktyping for behaving like a specific class (e.g. Forecaster, Classifier, ...)
* uses inspection for determining the correct arguments for call, fit, etc. methods.

A prototype that describes the general idea is provided in the following PR:

### The proposed solution for the Graphpipeline
TODO


## Problem to solve

### Requirements

* the new pipeline should be compatible to all sktime estimators
* it should behave like an classifier, if a classifier is added in the pipeline, behave like a forecaster 
if a forecaster is added ...
* 

### Current implementation and workflow

Currently, for each estimator a separate pipeline implementation exist.

### Problems of the current soultion

TODO 

## Proposed solution

### User journey design

The user has only to import the pipeline regardless of the type of the used estimator.


#### User journey: general Pipeline Design

```python
TODO
```


### Code design: General Pipeline

The general pipeline implements each fit/predict/transform method that is available in sktime.

### Code design: Ducktyping if a method is allowed

If a method is called on the pipeline, the pipeline checks if the method is available. If not it will aise
an NotImplementException or something similar. 


### Code design: Use inspection for indentifying parameters.
If the method is available then pipeline uses inspection to determine which arguments has to be passed to the method of
the pipeline step. Afterwards it is checked if these parameters are available and then the method on the step is called.

## Graph Pipeline Code Design

