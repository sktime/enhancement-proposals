# API for pretraining parameters

Contributors: @fkiraly @SimonBlanke @felipeangelimvieira

This STEP has the idea of centralizing the discussion on edge-cases of global forecasters
and pretraining-compatible models with respect to pretrained parameters.


## Related issues
* https://github.com/sktime/sktime/issues/10151 
* https://github.com/sktime/skbase/issues/554   


## Problem

Currently, `set_params` reset all model parameters, including the ones learned during pretraining. This should be avoided since, when using pretrained parameters in pipelines or other compositions, the clone operations and `set_params` should keep their pretrained parameters intact (e.g. model weights). Therefore, similarly to changes in clone API -- where we forced pretrained parameters to keep unchanged -- we should make set_params to keep pretrained parameters unchanged as well.

As pointed out by user Deep-Axe [in this issue](https://github.com/sktime/skbase/issues/554#issuecomment-4364219761), `reset` is currently not following the original sktime's contract. Its docstring says:

> "Equivalent to clone, with the exception that reset mutates self instead of returning a new object."

This means that, as clone currently keeps pretrained parameters unchanged, `reset` should do the same. However, this is not the case, and this incompatiblity is causing problems in `set_params` API too. Somehow, reset needs to be aware of parameters that belong to the model's pretraining phase and are not task-specific. At the same time, users might be interest in resetting all parameters including pretrained ones. For example, in a cross-validation setting where for each fold we have different pretraining data and the pretraining state should be erased, or a hyperparameter tuning pipeline where we want to understand the best hyperparameters for pretraining and after each iteration the pretrained parameters should be reset. Therefore, we need a way to control whether pretrained parameters should be reset or not, keeping the default behavior of keeping non-task-specific parameters unchanged.

## Requirements

* `reset` should have the same behavior as `clone` as default
* non-task-specific parameters should be kept unchanged by default in `reset`, `clone` and `set_params` operations
* users should be able to reset non-task-specific parameters when needed

## Proposed solution

1. New argument `reset`, `clone` and in `set_params` to control whether pretrained parameters should be reset or not, with default value `False` (i.e., pretrained parameters are not reset by default). Suggested name: `reset_pretrained=False`.
2. Estimators should have a class attribute `_PRETRAINED_ATTRIBUTES : list[str] | None` to list the attributes that belong to the model's pretraining phase and are not task-specific.
3. `reset` and `clone` should both look at such class attribute and reset them if `reset_pretrained=True`, and keep them unchanged if `reset_pretrained=False`. Every other attribute should be reset as usual.

## Alternative solutions

### Use only the function argument `_reset` or `reset_pretrained`
This would require less changes in the API, but it would not solve the problems in `reset` and would make it hard to control the compatiblity of global models in pipelines and other compositions (see issue [#544](https://github.com/sktime/skbase/issues/554)).

### Use a tag to control whether pretrained parameters should be reset or not
This would be equivalent to having the private class attribute `_PRETRAINED_ATTRIBUTES`. However, it introduces a tag which the set of values is not clear beforehand. Tags store metadata and they currently have, for every key, a finite set of possible values. Adding a pretrained attributes tag would force expand tags' scope and make it less clear what values are expected for such tag. On the other hand, having a private class attribute to list pretrained parameters is more explicit and easier to understand, and does not expand the scope of tags.


## Suggestion of tests to add to TestAllForecasters

1. When the pretrained tag is True, check if `_PRETRAINED_ATTRIBUTES` is set to a list. We should force users to set this attribute to an empty list to explicitly say that there are no pretrained parameters, instead of leaving it as None.
2. Check if the pretrained parameters are not reset in the following circunstances:
    2.1. Model instatiation + pretrain + reset
    2.2. Model instatiation + pretrain + clone
    2.3. Model instatiation + pretrain + set_params
    2.4. Model instatiation + pretrain + fit + reset
    2.5. Model instatiation + pretrain + fit + clone
    2.6. Model instatiation + pretrain + fit + set_params
3. For the same circunstances above, check if pretrained parameters are reset when `reset_pretrained=True` is passed as an argument to `reset`, `clone` and `set_params`.
