# API for pretraining parameters

Contributors: @fkiraly @SimonBlanke @felipeangelimvieira

This STEP centralizes the discussion of edge cases for global forecasters
and pretraining-compatible models with respect to pretrained parameters.


## Related issues
* https://github.com/sktime/sktime/issues/10151
* https://github.com/sktime/skbase/issues/554


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
