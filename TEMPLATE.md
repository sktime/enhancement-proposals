# BayesianEstimator Class for sktime and skpro

## 1. Introduction

This proposal outlines the design for a new `BayesianEstimator` template class to be integrated into `sktime` and `skpro`. The class aims to provide a flexible and robust framework for Bayesian estimation, while ensuring compatibility with existing `sktime` and `skpro` interfaces.

## 2. Problem Statement

Bayesian estimators integrate prior knowledge through prior distributions and update this knowledge based on observed data to form posterior distributions. The proposed `BayesianEstimator` class will accommodate this unique workflow while offering methods for prior specification, model building, posterior inference, and visualization.

## 3. Key Elements
Here are the key elements are that every Bayesian model needs to support:
1. Specification of prior distributions
2. Retrieval and inspection of prior distributions
3. Model fitting (Bayesian inference)
4. Retrieval and inspection of posterior distributions
5. Prediction using the posterior distribution
6. Bayesian update of an already fitted model
7. Visualization and diagnostics

## 3.1. Specification of prior distributions
A key design decision we need to address is how users should specify prior distributions to be passed to the PyMC model context. Below are some possible strategies


| Strategy                                                       | Advantage                                    | Disadvantage                                  |
| -------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------- |
| 
Pre-define the prior distributions, only allow users to specify parameters (e.g., mu and sigma) | <ul><li>Simplifies user input</li><li>Doesn't have to think about how to represent priors as distributions</li></ul> | <ul><li>Limits flexibility</ul> |
| 
Using `skpro` distribution                                                 | <ul><li>Native to the `skpro` project</li><li>Ease of interface with `pandas`</li></ul> | <ul><li>Require convert to PyMC distribution within model stack</li> |
|
`bambi`                                                    | | <ul><li>Inflexibility</li><li>Unfamiliar API (using string to define the whole model) that might be incompatible with sklearn interface</li></ul> |
|
`Prior` class from `pymc-marketing`                                                    | <ul><li>Easy to use API</li><li>Compatibility with pymc</li><li>Comes with pre-defined Bayesian template class that we can refit to our template and repurpose as our Bayesian class going forward</li></ul>| |


## 3.7. Visualization and Diagnostics

Implement methods for:
1. Visualizing prior and posterior distributions
2. Trace plots for MCMC diagnostics
3. Autocorrelation plots
4. Prior vs Posterior plots
5. Posterior predictive check plots
6. Pair plots for parameter relationships
7. Model flow visualization (similar to Graphviz visualizations of PyMC models)


## 4. Considerationss
### 4.0 General principles
1. Proliferation of methods should be avoided, all other things being equal.
2. Unified interfaces that are easy to comply with are better than those that are difficult to follow, and that´s better than non-unified

From this, some slight preferences:
- mapping priors on __init__, and posteriors on `get_fitted_params`
- unification: single prior param argument and `posterior_` fitted param
- that means, we need a `BaseObject` child class for priors/posteriors; this could be a `BaseDistribution` representing multiple, potentially associated random variables?

### 4.1 Model Specification
- Support defining Bayesian models, including specifying likelihood functions and linking them with priors.
- Design for modularity to support various Bayesian paradigms (e.g., Conjugate Bayesian analysis, MCMC, Variational Inference).

### 4.2 Inference Methods
- Implement methods for obtaining posterior distributions using MCMC (e.g., Metropolis-Hastings, Gibbs Sampling) or Variational Inference.
- Consider implementing different inference methods as strategy objects.

### 4.3 Hyperparameter Tuning
- Implement methods for tuning hyperparameters of priors, possibly using cross-validation.

### 4.4 Extensibility
- Design the `BayesianEstimator` classes to be easily subclassed for specific Bayesian paradigms or custom models.
- Allow users to define custom models and priors that can be easily integrated into the framework.

### 4.5 Integration with skpro Distributions
- Consider how to integrate or interoperate with existing `skpro` distribution classes.
- Possibly implement conversion methods between `BaseDistribution` and skpro distributions.

### 4.6 Compatibility with sktime and skpro
- Design the class to be orthogonal to existing model classes in `sktime` and `skpro`.

