Here’s the revised documentation with the changes requested:

# `BayesianLinearRegressor` Class for `skpro`

Contributors: @meraldoantonio

## 1. Introduction

This proposal outlines the design and functionality of the `BayesianLinearRegressor` class in `skpro`. This class provides a flexible framework for performing Bayesian linear regression while ensuring compatibility with the existing `skpro` interfaces. By leveraging `PyMC` for Bayesian inference, this class simplifies the process of specifying priors, fitting models, and performing posterior inference. This class is also intended as a blueprint after which future Bayesian estimators are to be modeled.

## 2. Problem Statement

Bayesian linear regression incorporates domain knowledge through the specification of prior distributions over model parameters. During the fitting process, these priors are updated with observed data to form posterior distributions. The `BayesianLinearRegressor` class supports this workflow by offering methods for:
- Prior specification
- Model fitting using Bayesian inference (via Markov Chain Monte Carlo - MCMC)
- Posterior and posterior sampling
- Inference
- Visualization and diagnostics

## 3. Backend Libraries Used

The logic of the  `BayesianLinearRegressor` class is built on top of the following backend libraries, chosen to ensure the `BayesianLinearRegressor` class is built on a stable foundation.

- **`PyMC`**: 
`PyMC` is the primary backend for Bayesian inference in the `BayesianLinearRegressor` class. This library, widely regarded as industry standard for Bayesian modeling in Python, provides a flexible framework for defining probabilistic models and performing inference using various flavors of MCMC. Its integration with `ArViz` and other libraries makes it a powerful tool for Bayesian analysis in Python.

- **`ArViz`**: 
`ArViz` is a library for managing, visualizing and diagnosing Bayesian models. It uses the `InferenceData` (`idata`) object as a standard container for storing and organizing data related to Bayesian models. This object holds posterior samples, prior samples, observed data, and other metadata, making it easy to analyze model outputs and allowing for seamless integration of various stages of the Bayesian workflow.

- **`pymc-marketing`**: 
Developed by the same team behind `PyMC`, this library provides a convenient `Prior` class for specifying prior distributions. It natively integrates with the rest of `PyMC` ecosystem.

## 4. Overview of Key Elements and Methods
The `BayesianLinearRegressor` class provides a set of methods for managing different stages of the Bayesian regression workflow. These stages, along with the relevant methods, are described below:

### 1. **Prior Specification**

The class allows users to define custom prior distributions or use default weakly informative priors for intercept, slopes, and noise variance.

- `sample_prior(return_type=None)`: Samples from the prior distributions.
- `get_prior_summary(**kwargs)`: Provides summary statistics of the prior distributions.

### 2. **Model Fitting**

The fitting process uses MCMC to estimate the posterior distributions based on the data and specified priors.

- `fit(X, y)`: Fits the Bayesian Linear Regression model to the training data `X` and target `y`.

### 3. **Inference**

After fitting, the model can be used to predict the distribution of the target variable for new input data.

- `predict_proba(X)`: Predicts the probability distribution of the target variable for the input data `X`.

### 4. **Bayesian Updates**

The class allows users to update an already fitted model by sampling from posterior distributions and posterior predictive checks.

- `sample_posterior(return_type=None)`: Samples from the posterior distributions.
- `get_posterior_summary(**kwargs)`: Provides summary statistics of the posterior distributions.
- `sample_in_sample_posterior_predictive()`: Performs in-sample predictions and samples from the posterior predictive distribution.

### 5. **Sampling and Visualization**

The class offers multiple methods to visualize the model and assess its performance.

- `visualize_model(**kwargs)`: Visualizes the Bayesian model using Graphviz.
- `plot_ppc(**kwargs)`: Plots the posterior predictive check using the sampled posterior predictive distribution.

## 5. Prior Specification: Further Details

The class provides flexible options for specifying prior distributions through the `Prior` class from `pymc-marketing`. By default, weakly informative priors are used for intercept, slopes, and noise variance, but users can easily modify these priors.

The `Prior` class integrates smoothly with `PyMC`, allowing for a clean and simple interface to define prior distributions. Each prior is specified by choosing a distribution (e.g., `Normal`, `HalfCauchy`) and setting its parameters (e.g., `mu`, `sigma`). This method is intuitive and aligns with the PyMC model-building process.

### Alternative Considerations:
Before finally settling for the use of the `Prior` class, alternative ways of specifying priors were considered and are briefly discussed below:
- **Predefined Priors with User-Adjustable Parameters**: Simplifies user input but limits flexibility.
- **skpro Distributions**: Would ensure native compatibility with skpro but requires tricky conversion to PyMC distributions

## 6. Model Fitting: Further Details

The `fit` method is responsible for fitting the Bayesian Linear Regressor to the provided dataset, \(X\) (features) and \(y\) (target). This method follows these steps:

1. **Set Up Data**: The method ensures that \(X\) and \(y\) are compatible with the PyMC framework. The target variable \(y\) is reshaped into a 1-dimensional array to meet PyMC's requirements.
   
2. **Model Construction**: The `fit` method constructs the Bayesian linear regression model within a PyMC model context. It uses the prior distributions specified in the `prior_config` and creates the variables for the intercept, slopes, and noise variance.

3. **Posterior Sampling**: Using the PyMC `sample` function, the method performs MCMC sampling to generate posterior distributions for the model parameters. The default MCMC sampler settings can be customized via the `sampler_config` parameter.

4. **Storing Results**: After fitting, the model stores the generated posterior samples in the `idata` attribute, managed by ArViz. The model is then ready for inference or further analysis.

## 7. Inference: Further Details

Once the model is fitted, it can be used to predict the probability distribution of the target variable for new input data using the `predict_proba` method.

1. **New Data Handling**: The method first sets the new input data \(X\) into the model using PyMC's `set_data` functionality. This ensures that predictions are made based on the new data rather than the training data.

2. **Posterior Predictive Sampling**: The `predict_proba` method performs posterior predictive sampling, which generates predictions by sampling from the posterior distribution of the model. This accounts for uncertainty in the model parameters.

3. **Returning Results**: The predictions are returned as a probability distribution using the skpro framework’s `Empirical` distribution. This distribution captures the full range of possible outcomes, reflecting the Bayesian nature of the model.
