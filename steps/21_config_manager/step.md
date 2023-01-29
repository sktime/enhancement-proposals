# Flag manager for configuration flags and tags

Contributors: fkiraly

## Contents

[TOC]

## Overview

### Problem Statement

`sklearn` has recently introduced a configuration manager that allows
users to configure behaviour of estimators, e.g., return type.

This is in contrast to tags, which contain metadata for estimator search,
developers, and internal APIs.

To clarify:

* *configuration* flags are set by the user to modify behaviour of estimator objects.
  Examples could be: setting the output type, or turning off checks.
  Configuration flags can be stateful and may be changed throughout usage of an estimator.
  
* *tags* are set by the developer as metadata for the estimator.
  For instance, they modify boilerplate behaviour or annotate estimators.
  Tags should be fully determined at construction of an estimator and are not stateful.

Status quo in `sktime`, and examples:

* `sktime` currently does not have a configuration flag interface, but
  uses ad-hoc attributes like `_output_covert` in `BaseTransformer` that fulfil the function.
  The `_output_convert` attribute is used as a config, to determine the output type of `transform`.
* Tags are already used in `sktime`, via the `get_tag` and `set_tag` interface.
  Examples are the `X_inner_mtype` tag (controls boilerplate conversion to inner `_fit`),
  or the `capability:unequal_length` tag (tells the user whether the estimator supports unequal length panels).

### The proposed solution

The solution below introduces:

* a configuration interface, via `get_config` and `set_config`
* an abstract `_FlagManager`, which contains logic for both tags and configs.

As tag and config logic are highly similar, the `_FlagManager` contains abstract
flag management functionality, which is called from within `get_config` and `get_tag`,
`set_config` and `set_tag`, etc.

### Requirements

* the end state must support the current tag interface via `get_tag`, `set_tag`, etc, unchanged.
  The internal refactor should not be too complex
* the end state must translate config-like attributes such as `_output_convert` naturally to the config interface
* addition of a config interface similar to `sklearn`'s, via `get_config` and `set_config`

## Proposed solution

### User journey design

The user journey for the tag system should not change.

For the configuration flags, setting and insection constitute the key elements
of the user journey, as below.

#### User journey design: configuration setting

Configuration setting will work as follows:

```python
# 1. create model and set config flags
my_estimator = MyEstimator().set_config(foo="foo", bar="bar", input_checks=False)

# 2. using the model
my_estimator.fit(y_train, fh=fh)
# this now behaves according to the config
# e.g., input checks are turned off
```

#### User journey design: configuration inspection

Each estimator comes with a set of configurable flags.
At any point in time, their values can be read out

```python
# 1. run some code, including potentially configuration
my_estimator = ...

# 2. using the model
config_dict = my_estimator.get_config()
# config_dict is a dictionary of key-value pairs
# keys are configuration flag names, values are their current values in my_estimator
```

### Code design: flag manager

The flag manager is a mixin class `_FlagManager`, which provides abstract
functionality for tags and configs. From this, `BaseObject` can inherit.

`_FlagManager`'s methods are private, as they are used within public methods like `get_tag`,
and all have an argument that provides a reference for the attribute dict
containing the flags, e.g., `_tags` or `_config`.

* `_get_class_flag(key, flagname -> value)` which reads values of a flag (in the class)
  E.g., `_get_flag(my_key, "tags")` reads from the `_tags` dict of the class
* `_get_flag(key, flagname) -> value` for obtaining value of an object flag.
  E.g., `_get_flag(my_key, "tags")` reads from the `_tags_dynamic` dict.
  Object flags, when set, take predence over class flags.
* `_set_flags(flags_dict, flagname)` updates the values of flags.
  e.g., `_set_flag(my_key, "tags")` sets values in the `_tags_dynamic` dict
* potentially further methoes such as `_clone_flags`, etc.

The getter methods also implement inheritance overrides.
Parent classes may set tags, for the getter `_get_flag`.
The override order for values, in decreasing priority order, is:

1. object flag set dynamically, e.g., via `_set_flags`
2. class flag of an object, i.e., in the class rather than the object
3. class flags set in parent classes, in reverse inheritance order

### Alternative designs considered

#### Composition - tag manager as object

The principle "composition over inheritance" suggests to investigate whether
the above could be more elegantly or clearly realized via composition, i.e.,
the flag manager being a component class rather than a mixin.

I have not been able to come up with a simple solution, because
some logic acts on the level of classes, and `_get_class_flag` is a class method,
and would have to be available alreaddy in the class rather than objects.

Whereas common composition patterns work on the level of objects, not classes.

#### Direct addition to `BaseObject` instead of a mixin

Another alternative would be adding the functionality directly to `BaseObject`,
instead of having `BaseObject` inheriting from the `_FlagManager` mixin.

I have a slight (not strong) preference against it, because:

* that would increase the number of methods in `BaseObject` even more than it already hase.
  "an object/function should do one thing and only one thing".
  Arguably that depends on the definition of what thing `BaseObject` is,
  but I find the "flag management" concern is better localized this way.
* The mixin is preferable in a case where one may like to manage tags/flags in an object
  that is not `BaseObject` descendant, although this is more of a hypothetical
* if we find a "composition not inheritance" solution later (see above),
  this might be easier to refactor towards that

### Code design: implementing tag and config manager

The tag and config managers would be refactored and implemented as follows:

* current tag functionality is abstracted into `_FlagManager`, from which
  `BaseObject` will inherit.
* current methods such as `get_class_tag` and `set_tags` use `_get_class_flag` and
  `_set_flags` etc under the hood with the `"tag"` attribute reference,
  resulting in `_tags` and `_tags_dynamic` attribute in classes and objects
* new methods `set_config` and `get_config` use `_set_flags` and `_get_tags`
  under the hood, with the `"config"` attribute reference,
  resulting in `_config` and `_config_dynamic` attribute in classes and objects
