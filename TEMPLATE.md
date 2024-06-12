# Enhancement Proposal: Design of a `BayesianEstimator` Class for `sktime` and `skpro`

Contributors: [meraldoantonio]

## Introduction
This proposal outlines the design for a new `BayesianEstimator` class to be integrated into `sktime` and `skpro`. 

## Contents
[table of contents]

## Problem statement
Bayesian estimators are unique because they integrate prior knowledge through prior distributions and update this knowledge based on observed data to form posterior distributions. To accommodate the distinctive workflow of Bayesian estimators, we propose designing a new `BayesianEstimator` class. This class aims to offer robust and flexible methods for prior specification, model specification, and visualization of both prior and posterior distributions, among other unique Bayesian functionalities, while still ensuring compatibility with the existing `sktime` and `skpro` `fit`/`predict` frameworks. 

Initially, the focus will be on continuous tabular regression.

## Special Aspects and Requirements of Bayesian Estimators

### Prior Specification
- **Methods for Prior Specification**: The estimator should allow users to specify priors for each parameter explicitly. This can be done through methods that accept common prior distributions (e.g., Normal, Beta, Gamma) or custom user-defined distributions.
- **Hyperparameter Tuning**: Functionality to tune hyperparameters of priors through cross-validation or other methods.

### Model Specification
- **Model Building**: Support for defining Bayesian models, including specifying likelihood functions and linking them with priors.
- **Modular Design**: The class should be modular to support various Bayesian paradigms (e.g., Conjugate Bayesian analysis, MCMC, Variational Inference).

### Posterior Inference
- **Inference Methods**: Implement methods for obtaining posterior distributions using MCMC (e.g., Metropolis-Hastings, Gibbs Sampling) or Variational Inference.
- **Posterior Summarization**: Methods to summarize posterior distributions (mean, median, credible intervals).

### Visualization
- **Prior and Posterior Visualization**: Methods to visualize prior and posterior distributions, including density plots, trace plots for MCMC diagnostics, and convergence diagnostics.
- **Model Flow**: Methods to visualize the model structure, similar to Graphviz visualizations of PyMC models. This will help users understand the flow of the model and how priors, likelihoods, and posteriors interact within the Bayesian framework.

## Design Considerations

### Compatibility with `sktime` and `skpro`
- **Orthogonal to Model Classes**: The Bayesian Estimator class should be designed to be orthogonal to existing model classes in `sktime` and `skpro`. This means it can be used alongside current models without modification but adds Bayesian inference capabilities.

### Extensibility
- **Extensible Framework**: Design the class to be easily extensible to accommodate future Bayesian paradigms and methodologies.
- **Custom Models and Priors**: Allow users to define custom models and priors that can be easily integrated into the Bayesian Estimator framework.

## Detailed description of design and implementation of proposed solution 

(WIP)
```python
from skpro.regression.base import BaseProbaRegressor
import numpy as np
import matplotlib.pyplot as plt

# todo: change class name and write docstring
class BayesianEstimator(BaseProbaRegressor):
    """Bayesian probabilistic supervised regressor.

    Custom Bayesian regressor that incorporates prior knowledge and updates it with observed data to form posterior distributions.

    Parameters
    ----------
    priors : dict
        A dictionary of prior distributions for the model parameters.
    """


    def __init__(self, priors=None):
        self.priors = priors
        self.posterior = None
        super().__init__()

    def set_priors(self, priors):
        """Set prior distributions for the model parameters."""
        self.priors = priors

    def _fit(self, X, y):
        """Fit the Bayesian model to the data."""
        self.posterior = self._perform_bayesian_inference(X, y)
        return self

    def _predict_proba(self, X):
        """Make predictions using the posterior predictive distribution."""
        return self._predict_from_posterior(X)

    def summarize_posterior(self):
        """Summarize the posterior distributions of the parameters."""
        return self._summarize_posterior()

    def plot_prior(self):
        """Visualize the prior distributions."""
        self._plot_distributions(self.priors, title="Prior Distributions")

    def plot_posterior(self):
        """Visualize the posterior distributions."""
        self._plot_distributions(self.posterior, title="Posterior Distributions")

    def plot_trace(self):
        """Plot trace for MCMC diagnostics."""
        pass

    def _perform_bayesian_inference(self, X, y):
        """Perform Bayesian inference to obtain the posterior distribution."""
        pass

    def _predict_from_posterior(self, X):
        """Generate predictions from the posterior distribution."""
        pass

    def _summarize_posterior(self):
        """Summarize posterior distributions."""
        pass

    def _plot_distributions(self, distributions, title):
        """Plot distributions (prior or posterior)."""
        pass

    def visualize_model_flow(self):
        """Visualize the model flow using Graphviz."""
        pass

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = {"priors": {"param1": np.random.normal(size=1000), "param2": np.random.normal(size=1000)}}
        return params

