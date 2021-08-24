# Estimator test instance creation by the estimator

Contributors: @fkiraly, 

## High-level summary 

### The Problem

Currently, adding an estimator to the testing framework of `sktime` is not an easy task.

A key issues with the extension burden is the requirement to make modifications in a non-obvious place, in non-obvious manner, namely the necessity to add "default configs" (parameter settings) in the `test/_config` file.

### The Aim

We aim to reduce the extension burden by:

* reducing the number of extension loci to one - the file containing the new estimator
* enabling individual estimator tests that can be run easily from the command line.

### The proposed solution

Our proposed solution involves:

* a change to the `BaseObject` interface that adds a "test instance" initialization class method
* a minor refactor of auxiliary test functions that allows testing directly on a class
* removal of `test/_config` as a locus of extension and dependency (as far as possible)


## Implementation details

### New `BaseObject` interface points

We propose addition of two new interface point to `BaseObject` template:

A class method `get_test_params()`, which returns a `dict`, or iterable of `dict` (preferably a list). 

All `dict` in question are dictionaries `x` such that `MyClass(**x)` is a valid instance of the class itself if it is concrete.

A class method `create_test_instance()`, which returns an object that is an example instance of `MyClass` or one of its concrete descendants.

`get_test_params` is implemented by any concrete descendant of `BaseObject`; `create_test_instance` is implemented by any descendant of `BaseObject`, including base classes and `BaseObject` itself. It has a default where it uses the first (or only) `dict` in `get_test_params`.

### Changes to test logic

The key change to test logic is that example instances are constructed by invoking `create_test_instance`, or `get_test_params` - the former if a single example is needed, the latter if multiple test cases need to be covered.

This is easily achieved by the following modifications:

* all estimator tests that test objects are changed to assume the argument is a freshly constructed object rather than a class. For convenience, if a class is passed, an object is created using `create_test_instance`.
* all estimator tests that test classes are changed to assume the argument is an estimator class.
For convenience, if an object is passed, a class is created using `type`.
* the loop in `check_estimator` is changed to be:
    * over tests for objects vs tests for classes, separately 
    * object tests loop over all estimator instances with `get_test_params` parameters, and all tests.
    * class tests loop over all estimator classes only
* the above replaces `_construct_instance` in all cases.

### Changes to individual estimators and default settings

All estimator default settings are moved into `get_test_params`, from `tests/_config`.

Where estimator default settings refer to template defaults like `FORECASTER`, that default is added as the return of `create_test_instance` of the respective base class, e.g., `BaseForecaster`, which is used in the descendant.

There are some edge cases like `STEPS_y`. It appears most of these are used only once, so can be moved to the respective estimator's `get_test_params`.

### Changes to the extension templates

All extension templates need to be updated with an explanation of how to implement `get_test_params`.

`create_test_instance` does not need to be added to the extension template since it has a default implementation; but it needs to be added to instances of documentation around the base class templates, most importantly in the extension template and base class docstring preambles.

## Suggested refactor sequence

I would suggest to implement in the following sequence, split across multiple PR:

1. Extending `BaseObject` with the new methods. Temporarily, `get_test_params` should default to the current logic in `_construct_instance`. This way, all concrete estimators are endowed with functioning class methods. This could be but need not be a separate PR.
2. making the changes to the test logic, i.e., removing `_construct_instance` from all the tests, and making the changes as described in the "changes to test logic" section above. This should be done in a separate PR, and also contain the changes to the docs and extension templates. From now one, new estimators can use the new interface, while the old ones are still working due to the interface loopthrough. Now only the old ones need to be refactored.
3. Only when the above is done with all tests still running, start making changes one-by-one, moving default parameter settings from `tests/_config` into the individual estimators. This is very formulaic, so I recommend to open an issue once we are here and ask for help from the community, individual contributors taking on an entire module or sub-module at a time, with different sub-modules kept track of on a checklist.
4. Once `test/_config` is entirely empty, celebrate :smiley:
