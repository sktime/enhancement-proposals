# Causality-Based Models for Time Series Analysis in sktime

Contributors: [Spinachboul](https://github.com/Spinachboul)

## Introduction

Time series analysis often focuses on correlation and predictive modeling, but understanding causal relationships is crucial for decision-making, policy evaluation, and robust forecasting. This proposal aims to integrate causality-based models into sktime, enabling users to uncover and leverage causal relationships in time series data. The initial implementation will focus on Granger Causality, with plans to expand to other methods like Structural Causal Models (SCMs) and Causal Impact Analysis.

## Contents

1. Problem Statement
2. Description of proposed solution
3. Motivation
4. Discussion and Comparison of Alternative Solutions
5. Detailed Description of Design and Implementation
6. Prototypical Implementation

## Problem statement

Current time series analysis tools, including sktime, primarily focus on correlation-based methods and predictive modeling. However, these approaches do not explicitly model causal relationships, which are essential for:

- Understanding the drivers of time series dynamics.
- Evaluating the impact of interventions or policy changes.
- Improving forecasting accuracy by accounting for causal effects.

There is a need for a unified framework within sktime to incorporate causality-based models, making it easier for users to analyze and leverage causal relationships in time series data.

## Description of proposed solution

The proposed solution involves integrating causality-based models into sktime, starting with Granger Causality and expanding to other methods such as:

- Structural Causal Models (SCMs): Representing causal relationships using directed acyclic graphs (DAGs).
- Causal Impact Analysis: Estimating the effect of interventions on time series.

## Motivation

**Enhanced Insights**: Causal models provide a deeper understanding of time series dynamics beyond correlation.

**Improved Forecasting**: Incorporating causal relationships can lead to more accurate and interpretable forecasts.

**Policy and Decision-Making**: Causal models are critical for evaluating interventions in domains like economics, healthcare, and climate science.

**Community Demand**: There is growing interest in causal inference within the time series community, and sktime is well-positioned to address this need.

## Discussion and comparison of alternative solutions

1. **Grangular Causality**
    - **Pros**: Simple, widely used, and easy to interpret.
    - **Cons**: Limited to linear relationships and may not capture true causality in complex systems.

2. **Structural Causal Models(SCMs)**
    - **Pros**: Can model complex, non-linear relationships and handle cofounding variables.
    - **Cons**: Requires domain knowledge to specify causal graphs and is computationally intensive.

3. **Causal Impact Analysis**
    - **Pros**: Specifically designed for intervention analysis and easy to use.
    - **Cons**: Limited to scenarios with a clear pre- and post-intervention period.

## Detailed description of design and implementation of proposed solution 

### Design Principles

1) **Consistent API**: Follow sktime's `fit`, `predict`, and `transform` interface.
2) **Modularity**: Allow users to easily switch between different causality models.
3) **Interoperability**: Ensure compatibility with sktime's data structures and pipelines.

### Implementation Plan

1) **Granger Causality**
    - Use `statsmodels` for the core implementation.
    - Add methods for testing and visualizing causal relationships.
2) **Structural Causal Models (SCMs)**
    - Integrate with libraries like `pywhy` or `dowhy`. Official documentation can be found here: [Click Here](https://www.pywhy.org/)
    - Provide tools for specifying and validating causal graphs.
3) **Causal Impact Analysis**
    - Adapt Google's `CausalImpact` library for sktime's interface.
    - Add functionality for visualizing intervention effects.

### Prototypical Implementation

```python
from sktime.base import BaseEstimator
from statsmodels.tsa.stattools import grangercausalitytests

class GrangerCausality(BaseEstimator):
    def __init__(self, maxlag=2, test="ssr_chi2test"):
        self.maxlag = maxlag
        self.test = test

    def fit(self, X, y=None):
        # X is a pandas DataFrame with multiple time series
        self.results_ = {}
        for col in X.columns:
            if col != y.name:
                test_result = grangercausalitytests(X[[y.name, col]], maxlag=self.maxlag, verbose=False)
                self.results_[col] = test_result
        return self

    def get_causal_relationships(self):
        # Summarize causal relationships
        causal_relationships = {}
        for col, result in self.results_.items():
            p_values = [result[lag][0][self.test][1] for lag in range(1, self.maxlag + 1)]
            causal_relationships[col] = min(p_values) < 0.05  # Check significance at 5% level
        return causal_relationships
```
