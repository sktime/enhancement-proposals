# API for pretraining parameters

Contributors: @fkiraly @SimonBlanke @felipeangelimvieira

This STEP centralizes the discussion of edge cases for global forecasters
and pretraining-compatible models with respect to pretrained parameters.

## Related issues
* https://github.com/sktime/sktime/issues/10151
* https://github.com/sktime/skbase/issues/554

## Definitions

### Global models
Models that can be used with tasks different from those used to learn their parameters.

### Local models
Models whose parameters and training are task-specific.

### Time series foundation model (TSFM)
Global models whose weights and parameters are already stored somewhere and can be imported; there is no need for a learning phase on the user's machine to use them for new tasks.

### Global non-foundation model (GNFM)
Global models that need to be trained once before they can be used. There is no weight or parameter storage available. Example: ML models in a reduction framework.

### Pretrained attributes
These refer to attributes of the class changed or created during a `.pretrain` call.

### Pretrained parameters
These refer to the model parameters (e.g. neural network weights, an XGBoost fitted model for a reduction forecaster) that are global and not task-specific.

### Fit attributes
Parameters changed or created during a `.fit` call.

### Fit parameters
Task-specific model parameters. For global models, there might be no fit parameters. For classical statistical models such as ARIMA, which have task-specific implementations, all ARIMA parameters are fit parameters.

### Finetunable models
Models that can have incremental pretraining rounds without needing to be pretrained from scratch. XGBoost with reduction is not fine-tunable.

### Forecasting task
A given time-series forecasting problem; in sktime convention, it is defined by the data passed to `.fit(X=X, y=y)`.

## Use-cases

### Pretrained, fit

State change: `pretrained` -> `fitted`
Classical usage of global models. The pretraining step learns pretrained parameters and sets pretrained attributes, which are later used in `fit` for tasks that might be different from those seen in the pretraining step.

### Pretrained, clone

State change: `pretrained` -> `pretrained`

Common in composition with global models. Pretrained models are cloned for use in pipelines, ensembles, and cross-validation.

### pretrained, pretrain again (finetuning)

Stage change: `pretrained` -> `pretrained`

This use-case triggers different behavior depending on the model.

* Finetuning: for models that support finetuning, this would use new data to change pretrained parameters and, therefore, attributes. One potential challenge here is that we use the multiton pattern, and we should be careful to allow finetuning **without side effects**, i.e., when finetuning this instance, other instances that had the same initial pretrained parameters should not be affected. Therefore, the *memory location* of pretrained attributes should change after this call.
* No support for finetuning: for reduction forecasters, for example, this would lead to undoing the first pretrain step and doing it again from scratch. If the two pretrain steps are called with the same data (and seeds are properly set), both calls would result in the same pretrained parameters.

### pretrained or fitted, then `set_params`

This is a more delicate use-case. This pattern can have two objectives:
* Changing model hyperparameters to prepare for a new finetuning pretrain call. In this case, some hyperparameter changes might be incompatible with the current pretrained state.
* Changing the model with new hyperparameters to obtain an entirely new estimator, without the intention of finetuning.

This use-case is intrinsically related to the behavior of `reset`, which is called during a `set_params` call. Should `reset` also reset pretrained attributes? This answer also defines what `set_params` does for models that can be finetuned.

In terms of user experience, the best option could be for `set_params` to reset pretrained attributes only if explicitly told to do so. This would make current models and behavior similar to what is expected in hyperparameter tuning pipelines and compositions.

One edge case is an ML model (e.g. XGBoost) in a hyperparameter tuning pipeline. The user might try to tune a pretrained Reduction model in a grid search, for example. However, it is not possible to, for example, change the `max_depth` of the tree and keep the pretrained state. In this case, we should decide whether the model fails or resets to the `new` state.

The important aspect here is to notice that calling `set_params` on a model that is `fitted` can lead to different behavior for global and local models. For example, calling `set_params` on a fitted ARIMA model makes it go back to the `new` state. When calling `set_params` on a global neural network to change the learning rate, should it go to the `new` or `pretrained` state?


### Initialization then fit of a global model

After initialization, users might call `fit` without `pretraining`. This step, although valid for TSFM, requires downloading model weights and, if this is not done at initialization, the model weights need to be imported at `pretrain` or `fit`, depending on which one is called first. For GNFM, this `fit` call would require a first pretrain call.


## Problem

### Problem 1: resetting pretrained parameters in `reset`

* State: `pretrained`
* Action: `.set_params` or `reset`

Currently, `set_params` resets all model parameters, including parameters learned
during pretraining. This should be avoided because, when pretrained parameters
are used in pipelines or other compositions, clone operations and `set_params`
should keep those parameters intact, for example, model weights. Therefore,
similarly to the changes in the clone API, where pretrained parameters were
made to remain unchanged, `set_params` should preserve pretrained parameters as
well.

As pointed out by user Deep-Axe
[in this issue](https://github.com/sktime/skbase/issues/554#issuecomment-4364219761),
`reset` currently does not follow the original sktime contract. Its docstring
says:

> "Equivalent to clone, with the exception that reset mutates self instead of returning a new object."

This means that, because clone currently keeps pretrained parameters unchanged,
`reset` should do the same. However, this is not the case, and this
incompatibility is also causing problems in the `set_params` API. Therefore,
`reset` needs to be aware of parameters that belong to the model's pretraining
phase and are not task-specific. At the same time, users might be interested in
resetting all parameters, including pretrained ones. For example, in a
cross-validation setting where each fold has different pretraining data, the
pretraining state should be erased. Similarly, in a hyperparameter tuning
pipeline where we want to find the best hyperparameters for pretraining, the
pretrained parameters should be reset after each iteration. Therefore, we need a
way to control whether pretrained parameters are reset, while keeping the
default behavior of preserving non-task-specific parameters.

There is also ambiguity in terms of the final state after `set_params` in a global model. For a finetunable model, `set_params` can be used to change some hyperparameters, such as the learning rate, while keeping pretrained parameters. In that sense, going back to the `new` state could lead to ambiguity, since the model will be in a state where, originally, its pretrained parameters were not yet defined.

### Problem 2: incompatibility of certain hyperparameters with existing pretrained parameters

Consider the case where a user has a pretrained model with certain
hyperparameters, such as `hidden_dim`, and sets them to new values. Because the
model is a global model, the pretrained parameters keep the old `hidden_dim`
weights. This can cause errors when the user tries to use the model with the new
hyperparameters, because the pretrained parameters are not compatible with the
new hyperparameters.

We can make the user responsible for setting the correct parameters properly,
but this could introduce errors that are hard to debug. Therefore, we should
also make the user aware when they are trying to set hyperparameters that are
incompatible with the existing pretrained parameters, and decide whether we want
to:

1.  raise an error in this case
2.  raise a warning and force the reset of pretrained parameters when
    incompatible hyperparameters are set.

Option 2 can silently change the behavior of the model without the user being
fully aware of it, if they are not checking the logs. Therefore, we consider
option 1 to be more explicit, because it makes the user aware of the exact
behavior of the code while avoiding silent buggy behavior.

This would require models to check whether a certain change in `__init__`
arguments is incompatible with the existing pretrained parameters, and to raise
an error in that case. This might not be linear because, for certain parameters,
the incompatibility of the change might depend on the value of other parameters
and the current state.

### Problem 3: shared memory between cloned objects

When pretrained parameters are not reset during cloning, there is a risk of
shared memory between the original and cloned objects. This can lead to
unintended side effects when one of the objects is modified, as it can affect
the other object as well. Therefore, we need to ensure that when pretrained
parameters are not reset during cloning, they are also properly copied to avoid
shared memory issues.

## Requirements

* by default, `reset` should have the same behavior as `clone`
* pretrained parameters should be cloned (or deep-copied if `.clone()` is not supported) and keep their values unchanged by default in `reset`, `clone`, and `set_params` operations
* users should be able to reset pretrained parameters when needed
* users should be made aware when they are trying to set hyperparameters that are incompatible with the existing pretrained parameters, and an error should be raised in this case.
  
## Proposed solution

### State transition & actions

|State|Action|Is global model|Is TSFM|Is finetunable|Final State| Changes
|-----|-----|-----|-----|-----|-----|-----|
|`new`|`pretrain`|Yes|-|-|`pretrained`|Pretrained parameters are set|
|`new`|`fit`|Yes|Yes|-|`fitted`|Load weights, save to pretrained attributes and bind to task|
|`new`|`fit`|Yes|No|-|`fitted`|Call pretrain and then fit|
|`pretrained` or `fitted`|`reset`|Yes|-|-|`pretrained`|Reset non-pretrained attributes, keep pretrained parameters|
|`pretrained` or `fitted`|`reset(reset_pretrained=False)`|Yes|-|-|`pretrained`|Reset non-pretrained attributes, keep pretrained parameters|
|`pretrained` or `fitted`|`reset(reset_pretrained=True)`|Yes|-|-|`new`|Reset pretrained attributes, reset all parameters|
|`pretrained`|`set_params` on finetunable hyperparameters|Yes|-|Yes|`pretrained`|Call reset with `reset_pretrained=False` and change finetunable hyperparameters|
|`pretrained`|`set_params`|Yes|-|No|`new`|Change hyperparameters, reset all attributes and initialize|
|`pretrained`|`pretrain` again|Yes|-|No|`pretrain`|Call reset(reset_pretrained=True) and then pretrain again, potentially going back to the same initial pretrained parameters|
|`pretrained`|`pretrain` again|Yes|-|Yes|`pretrain`|Copy and incrementally change pretrained parameters, go to `pretrained` state|

### On API level

1. Add a new argument to `reset`, `clone`, and `set_params` to control whether
   pretrained parameters should be reset, with default value `False`, i.e.,
   pretrained parameters are not reset by default. Suggested name:
   `reset_pretrained=False`.
2. Estimators should have a tag `pretrained_attributes` that lists the attributes that
   belong to the model's pretraining phase and are not task-specific.
3. `reset` and `clone` should both inspect that tag, reset the
   listed attributes if `reset_pretrained=True`, and keep them cloned/deep-copied if
   `reset_pretrained=False`. Every other attribute should be reset as usual.
4. In a `set_params` call, `reset()` is called and then `__init__` and `__post_init__` are called. As we have seen, there are use-cases where pretrained state is preserved after a `reset` call. When called with `set_params`,if the pretrained state is preserved and pretrained attributes are present, `__post_init__` should check whether the new hyperparameters are compatible with the existing pretrained parameters and raise an error if they are not. This function should be called before setting the new parameters. See the following skbase changes for more details on how to implement this.

## Alternative solutions

### Use only the function argument `_reset` or `reset_pretrained`

This would require fewer API changes, but it would not solve the problems in
`reset` and would make it hard to control the compatibility of global models in
pipelines and other compositions (see issue
[#554](https://github.com/sktime/skbase/issues/554)).

### Use a class attribute to control whether pretrained parameters should be reset

An alternative solution would be to have a private class attribute
`_PRETRAINED_ATTRIBUTES` to list pretrained parameters instead of a tag.

## Suggestion of tests to add to TestAllForecasters

1. When the pretrained tag is `True`, check whether `_PRETRAINED_ATTRIBUTES` is
   set to a list. We should require users to set this attribute to an empty list
   to explicitly say that there are no pretrained parameters, instead of leaving
   it as `None`.
2. Check whether pretrained parameters are not reset in the following
   circumstances:
    2.1. model instantiation + pretrain + reset
    2.2. model instantiation + pretrain + clone
    2.3. model instantiation + pretrain + set_params
    2.4. model instantiation + pretrain + fit + reset
    2.5. model instantiation + pretrain + fit + clone
    2.6. model instantiation + pretrain + fit + set_params
3. For the same circumstances above, check whether pretrained parameters are
   reset when `force_reset=True` is passed as an argument to `reset`,
   `clone`, and `set_params`.

4. Check the behavior of test 2 when the model is used inside a pipeline or composition with a non-global model.
5. The reserved keyword argument `force_reset` should not be a `__init__` argument.
6. Check whether cloning and calling pretrain on the second instance does not change the pretrained parameters of the first instance.
7. Check whether setting hyperparameters that are incompatible with existing pretrained parameters raises an error.
