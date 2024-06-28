# BayesianEstimator Class for sktime and skpro

## 1. Introduction

This proposal outlines the design for a new `BayesianEstimator` class to be integrated into `sktime` and `skpro`. The class aims to provide a flexible and robust framework for Bayesian estimation in continuous tabular regression, while ensuring compatibility with existing `sktime` and `skpro` interfaces.

## 2. Problem Statement

Bayesian estimators integrate prior knowledge through prior distributions and update this knowledge based on observed data to form posterior distributions. The proposed `BayesianEstimator` class will accommodate this unique workflow while offering methods for prior specification, model building, posterior inference, and visualization.

## 3. Key Workflow Elements

Here are the key elements are that every Bayesian model needs to support:
1. Specification of prior distributions
2. Retrieval and inspection of prior distributions
3. Model fitting (Bayesian inference)
4. Retrieval and inspection of posterior distributions
5. Prediction using the posterior distribution
6. Bayesian update of an already fitted model
7. Visualization and diagnostics


## 4. Design Decisions
1. Proliferation of methods should be avoided, all other things being equal.
2. Unified interfaces that are easy to comply with are better than those that are difficult to follow, and that´s better than non-unified

From this, some slight preferences:
- mapping priors on __init__, and posteriors on `get_fitted_params`
- unification: single prior param argument and `posterior_` fitted param
- that means, we need a `BaseObject` child class for priors/posteriors; this could be a `BaseDistribution` representing multiple, potentially associated random variables?

## 5. Class Structure 
Here is a WIP prototypical implementation that fulfills the above requirements.

```python
from skpro.regression.base import BaseProbaRegressor
from skpro.base import BaseObject

class BaseDistribution(BaseObject):
    """Representation of one or more potentially associated random variables/distributions."""
    
    def __init__(self, distributions):
        self.distributions = distributions
    
    def plot(self):
        # Implementation for plotting the distribution(s)
        pass
    
    def sample(self, n_samples):
        # Implementation for sampling from the distribution(s)
        pass
    
    def summary(self):
        # Implementation for summarizing the distribution(s)
        pass

class BayesianEstimator(BaseProbaRegressor):
    """Bayesian probabilistic supervised regressor."""

    def __init__(self, prior=None):
        self.prior = prior
        super().__init__()

    def _fit(self, X, y):
        self._posterior = self._perform_bayesian_inference(X, y)
        return self

    def _predict_proba(self, X):
        return self._predict_from_posterior(X)

    def get_fitted_params(self, deep=True):
        return {"posterior_": self._posterior}

    def update(self, X, y):
        self._posterior = self._perform_bayesian_update(X, y)
        return self

    # Additional methods for visualization, diagnostics, etc.
```
## 6. Class Structure 
### 6.1 Prior Specification
- Use a single `prior` parameter in `__init__` for passing priors.
- The `prior` should be an instance of `BaseDistribution`.
- Allow specification of common prior distributions (e.g., Normal, Beta, Gamma) and custom user-defined distributions.

### 6.2 Posterior Retrieval
- Use `get_fitted_params` for retrieving posteriors, returning a single `posterior_` parameter.

### 6.3 Bayesian Update
- Implement an `update` method for Bayesian updating of an already fitted model.

### 6.4 Distribution Representation
- Introduce a `BaseDistribution` class to represent both priors and posteriors.

### 6.5 Visualization and Diagnostics
- Implement common diagnostic plots as methods of `BayesianEstimator`.
- Leverage the `plot` method of `BaseDistribution` for distribution-specific visualizations.

## 7. Additional Considerations

### 7.1 Model Specification
- Support defining Bayesian models, including specifying likelihood functions and linking them with priors.
- Design for modularity to support various Bayesian paradigms (e.g., Conjugate Bayesian analysis, MCMC, Variational Inference).

### 7.2 Inference Methods
- Implement methods for obtaining posterior distributions using MCMC (e.g., Metropolis-Hastings, Gibbs Sampling) or Variational Inference.
- Consider implementing different inference methods as strategy objects.

### 7.3 Hyperparameter Tuning
- Implement methods for tuning hyperparameters of priors, possibly using cross-validation.

### 7.4 Extensibility
- Design the `BaseDistribution` and `BayesianEstimator` classes to be easily subclassed for specific Bayesian paradigms or custom models.
- Allow users to define custom models and priors that can be easily integrated into the framework.

### 7.5 Integration with skpro Distributions
- Consider how to integrate or interoperate with existing skpro distribution classes.
- Possibly implement conversion methods between `BaseDistribution` and skpro distributions.

### 7.6 Compatibility with sktime and skpro
- Design the class to be orthogonal to existing model classes in `sktime` and `skpro`.

## 8. Visualization and Diagnostics

Implement methods for:
1. Visualizing prior and posterior distributions
2. Trace plots for MCMC diagnostics
3. Autocorrelation plots
4. Prior vs Posterior plots
5. Posterior predictive check plots
6. Pair plots for parameter relationships
7. Model flow visualization (similar to Graphviz visualizations of PyMC models)

## 9. Next Steps

1. Implement the `BaseDistribution` class with basic functionality.
2. Refine the `BayesianEstimator` class based on this proposal.
3. Implement a concrete example of a simple Bayesian model (e.g., Bayesian linear regression) using this framework.
4. Develop diagnostic and visualization methods.

