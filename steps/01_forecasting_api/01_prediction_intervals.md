# Forecasting prediction intervals
https://people.bath.ac.uk/mascc/PIs_paper.pdf

## What are prediction intervals?
* an prediction interval consists of an upper and lower limit between which a future value is expected to lie with a prescribed probability 
* the limits are called forecast limits or prediction bounds while the intervals is referred to as prediction interval
* note that confidence interval usually applies to estimates of a fixed but unknown parameter, whereas prediction interval is an estimate of an unknown future value of a random variable (see https://robjhyndman.com/hyndsight/intervals/)
* prediction intervals can be computed for a single or multiple future time points
* finding the entire probability distribution of a future values is called a density forecast
* typically, but not necessarily, prediction intervals get wider indicating the increasing uncertainty about future values ("fan chart")

## Why compute prediction intervals?
* why compute prediction intervals: point forecasts provide no guidance as to their uncertainty

## How to compute prediction intervals?
* various methods, no generally accepted method of calculating prediction intervals
* general formula for a 100(1 - a)% prediction interval for $X_{N+h}$:

$\hat{x}(h) +- z_{a/2}\sqrt(Var[e_N(h)])$

where $e_N(h)$ is the forecasting error defined as 

$e_N(h) = X_N(h) - \hat{x}_N(h)$

* assumes errors are normally distributed, an assumption that is usually violated in practice
* main problem is to evaluate $Var[e_N(h)]$

### Derive from probability model
* for some models, it is possible to derive them from the assumed probability model (e.g. ARIMA, various regression models), but typically not possible for non-linear models
* assumes true model has been identified

### prediction mean squared error formula (PMSE)

* reasonable if one-step ahead forecast errors show no autocorrelation and no other obvious pattern in the data which needs to be modelled (e.g. trend)


### Approximate formulae
* usually very inaccurate

### Methods based on observed distribution

#### 1. From in-sample residuals
* obtain in-sample residuals at all required steps ahead from all available time origins (cutoffs)
* find the standard deviation $s(h)$ of the residuals for each step ahead h
* use $s(h)$ as an estiamte of $Var[e_N(h)]$
* assumes normality 
* can be unreliable for small $N$ and large $h$

#### 2. From temporal cross-validation errors
* split training data into training and validation data
* train on training data and make prediction for required steps head on validation data and compute out-of-sample errors
* repeat with sliding window 
* compute standard deviation $s(h)$ on out-of-sample errors for each step ahead h
* use $s(h)$ as an estiamte of $Var[e_N(h)]$

#### 3. Monte Carlo simulations
* for assumed probability model
* assumes true model has been correctly identified

#### 4. Bootstrapping
* obtain in-sample residuals at all required steps ahead from all available time origins (cutoffs) as in 1. above
* assuming future errors to be similar to past errors, we draw repeated sample from empirical distribution of residuals
* makes no distributional assumption on the residuals
* assumes residuals to be uncorrelated
* https://otexts.com/fpp3/prediction-intervals.html

## More complications: transformations
* inverse-transformations of forecast values of non-linear transformations (e.g. log or Box-Cox) give biased point forecasts (e.g. median not mean)


## Evaluation of prediction intervals
* Empirical evidence shows that out-of-sample forecast errors tend to be larger than model-fitted residuals, implying that more than 5% of future observations will fall outside a 95% P.I. on average
* The various reasons why P.I.s are too narrow in general include
    1. parameter uncertainty
    2. non-normally distributed innovations
    3. identification of the wrong model
    4. changes in the structure underlying the model
