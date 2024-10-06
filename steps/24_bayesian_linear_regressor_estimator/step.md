Here’s the revised documentation with the changes requested:

# `BayesianLinearRegressor` Class for skpro

Contributors: @meraldoantonio

## 1. Introduction

This proposal outlines the design and functionality of the `BayesianLinearRegressor` class, to be integrated into `skpro`. This class provides a flexible framework for performing Bayesian linear regression while ensuring compatibility with the existing `skpro` interfaces. By leveraging `PyMC` for Bayesian inference, this class simplifies the process of specifying priors, fitting models, and performing posterior inference.

## 2. Problem Statement

Bayesian linear regression incorporates domain knowledge through the specification of prior distributions over model parameters. During the fitting process, these priors are updated with observed data to form posterior distributions. The `BayesianLinearRegressor` class supports this workflow by offering methods for:
- Prior specification
- Model fitting using Bayesian inference (via Markov Chain Monte Carlo - MCMC)
- Posterior sampling
- Visualization and diagnostics

## 3. Libraries Used

- **PyMC**: The primary backend for Bayesian inference, including MCMC sampling.
- **ArViz**: For visualization and diagnostic checks of Bayesian models.
- **pymc-marketing**: Provides a convenient `Prior` class for specifying prior distributions.

These libraries ensure the `BayesianLinearRegressor` class is built on a stable foundation, facilitating integration with broader Bayesian workflows.

## 4. Overview of Key Elements and Methods

The `BayesianLinearRegressor` class provides a comprehensive set of methods for managing different aspects of the Bayesian regression workflow. Here’s how these methods map to the key steps in the workflow:

### 1. **Prior Specification**

The class allows users to define custom prior distributions or use default weakly informative priors for intercept, slopes, and noise variance.

- `sample_prior(return_type=None)`: Samples from the prior distributions.
- `get_prior_summary(**kwargs)`: Provides summary statistics of the prior distributions.

### 2. **Model Fitting**

The fitting process uses MCMC to estimate the posterior distributions based on the data and specified priors.

- `fit(X, y)`: Fits the Bayesian Linear Regression model to the training data `X` and target `y`.

### 3. **Using the Model for Inference**

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

## 5. Specification of Prior Distributions

The class provides flexible options for specifying prior distributions through the `Prior` class from `pymc-marketing`. By default, weakly informative priors are used for intercept, slopes, and noise variance, but users can easily modify these priors.

The `Prior` class integrates smoothly with `PyMC`, allowing for a clean and simple interface to define prior distributions. Each prior is specified by choosing a distribution (e.g., `Normal`, `HalfCauchy`) and setting its parameters (e.g., `mu`, `sigma`). This method is intuitive and aligns with the PyMC model-building process.

### Alternative Considerations:
- **Predefined Priors with User-Adjustable Parameters**: Simplifies user input but limits flexibility.
- **skpro Distributions**: Would ensure native compatibility with skpro but requires conversion to PyMC distributions.
- **`Prior` Class from `pymc-marketing`**: Provides an easy-to-use API with full compatibility with PyMC. This approach is adopted as the default for the `BayesianLinearRegressor`.

## 6. Model Fitting

The `fit` method is responsible for fitting the Bayesian Linear Regressor to the provided dataset, \(X\) (features) and \(y\) (target). This method follows these steps:

1. **Set Up Data**: The method ensures that \(X\) and \(y\) are compatible with the PyMC framework. The target variable \(y\) is reshaped into a 1-dimensional array to meet PyMC's requirements.
   
2. **Model Construction**: The `fit` method constructs the Bayesian linear regression model within a PyMC model context. It uses the prior distributions specified in the `prior_config` and creates the variables for the intercept, slopes, and noise variance.

3. **Posterior Sampling**: Using the PyMC `sample` function, the method performs MCMC sampling to generate posterior distributions for the model parameters. The default MCMC sampler settings can be customized via the `sampler_config` parameter.

4. **Storing Results**: After fitting, the model stores the generated posterior samples in the `idata` attribute, managed by ArViz. The model is then ready for inference or further analysis.

## 7. Using the Model for Inference

Once the model is fitted, it can be used to predict the probability distribution of the target variable for new input data using the `predict_proba` method.

1. **New Data Handling**: The method first sets the new input data \(X\) into the model using PyMC's `set_data` functionality. This ensures that predictions are made based on the new data rather than the training data.

2. **Posterior Predictive Sampling**: The `predict_proba` method performs posterior predictive sampling, which generates predictions by sampling from the posterior distribution of the model. This accounts for uncertainty in the model parameters.

3. **Returning Results**: The predictions are returned as a probability distribution using the skpro framework’s `Empirical` distribution. This distribution captures the full range of possible outcomes, reflecting the Bayesian nature of the model.
