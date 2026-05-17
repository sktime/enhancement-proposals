# API for pretraining parameters

Contributors: @fkiraly @SimonBlanke @felipeangelimvieira

This STEP centralizes the discussion of edge cases for global forecasters
and pretraining-compatible models with respect to pretrained parameters.

## Related issues
* https://github.com/sktime/sktime/issues/10151
* https://github.com/sktime/skbase/issues/554

## Concepts

### Pretrained attributes
These refer to attributes of the class changed or created during a `.pretrain` call.

### Pretrained parameters
Refer to the mode parameters (e.g. neural network weights, a XGBoost fitted model for ar eduction forecaster) that are global and not task-specific. 

### Fit attributes
Parameters changed or created during `.fit` call

### Fit parameters
Task-specific model parameters. For global models, there might be no fit parameters. For classical statistical models such as ARIMA, which have task-specific implementations, all ARIMA parameters are fit parameters.

### Fine-tunable models
Models that can have incremental pretraining rounds, without needing to be pretrained from scratch. XGBoost with reduction is not fine-tunable.

## Use-cases

### Pretrained, fit

State change: `pretrained` -> `fitted`
Classical usage of global models. Pretraining step learns pretrained parameters and set pretrained attributes, which are later used in `fit` for tasks that might be different from the ones seen in pretraining step.

### Pretrained, clone

State change: `pretrained` -> `pretrained`

Common in composition with global models. Pretrained models are cloned for usage in pipelines, ensembles and also cross-validation.

### pretrained, pretrain again (finetuning)

Stage change: `pretrained` -> `pretrained`

This use-case triggers different behaviours depending on the model.

* Finetuning: for models that support finetuning, this would use new data to change pretrained parameters and therefore attributes. One potential challenge here is that we use multiton pattern and we should be careful to allow finetuning **without side effects**, i.e., when finetuning this instance, other instances that had the same initial pretrained parameters should not be affected. Therefore, the *memory location* of pretrained attributes should change after this call.
* No support for finetuning: for reduction forecasters, for example, this would lead to undoing the first pretrain step and doing it again from scratch. If the two pretrain steps are called with the same data (and seeds are properly set), both calls would result in the same pretrained parameters.

### pretrained, `set_params`

This is a more delicate use-case. This pattern can have two objectives:
* Changing model hyperparameters to prepare to a finetuning pretrain call. In this case, some changes of hyperparameters might be incompatible with current pretrained state.
* Changing he model with new hyperparameters to obtain an entirely new estimator, without the intention of finetuning.

This use-case is intrinsically related to the behaviour of `reset`, which is called during a `set_params` call. Should reset also reset pretrained attributes? This answer also defines what `set_params` does for models that can be finetuned.

In terms of user experience, the best option could be `set_params` only resetting pretrained attributes if explicitely told to do so. This would make current models and behaviour similar to what is expected in hyperparameter tuning pipelines and compositions.

One edge case is a ML model (e.g XGBoost) in a hyperparameter tuning pipeline. The user might try to tune a pretrained Reduction model in a grid search, for example. However, it is not possible to, for example, change the `max_depth` of the tree and keep the pretrained state. In this case, we should decide if the model fails or it resets to `new` state.

The important aspect here is notice that calling `set_params` in a model that is `fitted` can lead to different behaviours for global and local models. For example, calling `set_params` to a fitted ARIMA make it go back to `new` state. Calling `set_params` to a global Neural Network, to change the learning rate, should make it go to `new` or to `pretrained` state? 


### Initialization, fit


## Problem

### Problem 1: resetting pretrained parameters in `reset`

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
* non-task-specific parameters should be cloned (or deep-copied if `.clone()` is not supported) and keep their values unchanged by default in `reset`, `clone`, and `set_params` operations
* users should be able to reset non-task-specific parameters when needed
* users should be made aware when they are trying to set hyperparameters that are incompatible with the existing pretrained parameters, and an error should be raised in this case.
  
## Proposed solution

### On API level

1. Add a new argument to `reset`, `clone`, and `set_params` to control whether
   pretrained parameters should be reset, with default value `False`, i.e.,
   pretrained parameters are not reset by default. Suggested name:
   `force_reset=False`.
2. Estimators should have a class attribute
   `_PRETRAINED_ATTRIBUTES : list[str] | None` that lists the attributes that
   belong to the model's pretraining phase and are not task-specific.
3. `reset` and `clone` should both inspect that class attribute, reset the
   listed attributes if `force_reset=True`, and keep them cloned/deep-copied if
   `force_reset=False`. Every other attribute should be reset as usual.
4. `set_params` should call a function `_check_set_params_compatibility(self, **params) -> None` that checks whether the new hyperparameters are compatible with the existing pretrained parameters, and raises an error if they are not. This function should be called before setting the new parameters. See the following skbase changes for more details on how to implement this.

### Skbase changes

For handling incompatible hyperparameters with existing pretrained parameters:
* Option 1: Add a `set_params` plugin as in `clone`
* Option 2: explicitly call a function `_check_set_params_compatibility(self, **params) -> None` in the base `set_params` implementation.

For handling non-resettable attributes:

* On the skbase side, we could add a `_NON_RESETTABLE_ATTRIBUTES_ATTR_NAME` class attribute for objects. This attribute specifies the name of the class attribute that lists non-resettable attributes.
* The default is `None`, which maintains the current behavior of not having non-resettable attributes.
* If set to a string, it should be the name of a class attribute that lists non-resettable attributes.
* During reset and clone operations, it should check whether the attribute exists and raise an error if it is not set. It should then use the `force_reset` argument to decide whether to reset the attributes listed in that class attribute. Passing `force_reset` to objects that do not have non-resettable attributes should not cause any issues and should behave as usual.

This would allow us to have a more general API for estimators that have attributes that are not task-specific.



## Alternative solutions

### Use only the function argument `_reset` or `reset_pretrained`

This would require fewer API changes, but it would not solve the problems in
`reset` and would make it hard to control the compatibility of global models in
pipelines and other compositions (see issue
[#554](https://github.com/sktime/skbase/issues/554)).

### Use a tag to control whether pretrained parameters should be reset or not

This would be equivalent to having the private class attribute
`_PRETRAINED_ATTRIBUTES`. However, it introduces a tag for which the set of
values is not clear beforehand. Tags store metadata, and they currently have a
finite set of possible values for every key. Adding a pretrained attributes tag
would expand the scope of tags and make it less clear what values are expected
for such a tag. On the other hand, having a private class attribute to list
pretrained parameters is more explicit and easier to understand, and does not
expand the scope of tags.


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

4. Check behavior of tests 2. when the model is used inside a pipeline or composition with a non-global model.
5. The reserved keyword argument `force_reset` should not be a `__init__` argument.
6. Check whether cloning and calling pretrain on the second instance does not change the pretrained parameters of the first instance.
7. Check whether setting incompatible hyperparameters with existing pretrained parameters raises an error.
