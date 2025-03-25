# API for pre-training and fine-tuning

Contributors: @fkiraly

## High-level summary 

### The Aim

`sktime` now has a number of estimators that can do pre-training, fine-tuning, global learning, and cross-learning.

However, the current API design for these use cases has a few problems, and repeatedly issues have been opened for a rework.

This STEP is about finalizing a good interface for:

* global forecasting
* pre-training
* fine-tuning of foundation models
* zero-shot use of foundation models

References:

* conceptual design issue: https://github.com/sktime/sktime/issues/6580
* umbrella issue foundation models: https://github.com/sktime/sktime/issues/6177
* newer issue: https://github.com/sktime/sktime/issues/7838

### requirements

* design covers the above use cases with a simple interface
* composability - use of sensible pipelines, tuning, etc should be simple and not require major surgery in current compositors
* downwards and upwards compatibility - design should not impact current extension contracts
* maintainability: maintaining the framework and estimators with the above capabilities should be simple

### The proposed solution

Our proposed solution adds a new state, and a simple switch.

No new public methods are added beyond this, and signatures of methods are not modified.

Estimators get a third state, "pretraining phase", besides unfitted and fitted.

The solution is best illustrated in the basic vignette below.

### Discussion of current solutions

There are multiple current solutions, all have problems:

#### Global forecasting

Forecasters inheriting from `_BaseGlobalForecaster`.

A `y` is added in the `predict` methods. If this is passed, the `fit` is interpreted
as a pretraining pass.

Problems:

* some models need to know at `fit` time whether the data is for pretraining.
Examples: global reduction approaches. Broadcasting.
* as a general design principle, all `predict` methods would need to be changed to
allow for addition of the `fit` arguments. This clashes in cases where arguments
of the same name are present both in `fit` and `predict`, e.g., the `X` in forecasting,
or all arguments for transformations.

#### Pre-training foundation models

Foundation models currently come in two different, contradictory forms:

* those that carry out fine-tuning in `fit`
* those that pass the context in `fit`, e.g., zero-shot models

Problems: This is inconsistent, and it does not seem to be possible - without an `__init__` arg
that acts as a switch, or in different classes, to have the same weights in the same class
be part of a zero-shot or fine-tuning algorithm.


## Design: pretraining vignette

Presenting user facing API. For delineation against current designs:

* no new arguments are added to `predict`-like methods
* a flag is added before or at `fit` to determine whether usage is normal fitting, or pre-training.
    * two vignettes are presented that pass this information on, for discussion.

### basic usage vignette

Illustrated for global forecasting.

```python

y_pretrain, X_pretrain = load_pretrain_data()

f = MyForecasterCls(params)

f.pretrain()

f.fit(y=y_pretrain, X=X_pretrain)

# fh is optional, but some models require this

f.pretrain("off")

# usual vignette starts here
y, X = load_data()

f.fit(y, X, fh=[1, 2, 3])

f.predict()
f.predict_intervals()
```

With optional serialization after pre-training:

```python

# optional: serialize

f.save(checkpoint_name)

# restart

f = load(checkpoint_name)
```


### Alternative vignette 1

An alternative idea would be adding an arg to `fit`:

```python

y_pretrain, X_pretrain = load_pretrain_data()

f = MyForecasterCls(params)

f.fit(y=y_pretrain, X=X_pretrain, pretrain=True)

# usual vignette starts here

y, X = load_data()

f.fit(y, X, fh=[1, 2, 3])

f.predict()
f.predict_intervals()
```

### Alternative vignette 2

An alternative idea would be adding an new method

```python

y_pretrain, X_pretrain = load_pretrain_data()

f = MyForecasterCls(params)

f.pretrain(y=y_pretrain, X=X_pretrain)

# usual vignette starts here

y, X = load_data()

f.fit(y, X, fh=[1, 2, 3])

f.predict()
f.predict_intervals()
```


### Mapping use cases on the vignette

The following map on the "pre-train" phase:

* Training for global forecasting
* fine-tuning
* pre-training of any other kind

Zero-shot models do not have pre-training, but `fit` needs to be called,
it is used only to read in the context (there is no `y` in `predict`).


## Design: concepts and internals

### Conceptual model, state diagram

Estimators get a third state, from two:

* blueprint/pristine
    * definition: directly after `__init__`
    * even if a pretrained neural network is constructed with a checkpoint reference, we consider the `sktime` model a blueprint.
* pretrained (new)
    * definition: pretrained attributes are present, at least one call of `fit` in pretrain mode
* fitted
    * at least one call of `fit` in normal mode.

The definition of pretrained is: pretrained attributes are present, definition as below.

Blueprint transitions to pretrained or directly to fitted.

Fitted cannot transition back to pretrained.

### Pretrained attributes and state attributes

Pretrained attributes, by convention, start with an underscore and end in an underscore.

They should not be present after `__init__`.

A `fit` (or `pretrain`) call may write only to pretrained attributes.

An attribute `_is_pretrained` is added, this tracks whether the model is pretrained.

### Tags

A tag `capability:pretrain` is introduced, and signifies models with non-trivial pretrained state.

The default behaviour is not an error raised, but the empty operation (a `pass`).

### Extension contract

An optional extender method `_pretrain` is added. This method returns `self`.

### Optional: checkpoint serialization for deep learning models

Some neural network models may have a `save_checkpoint` method.

This allows to serialize checkpoints directly for use in `__init__`.

Not all models will have this method.

usage:

```python

f = MyDLmodel(checkpoint=my_ckpt_path)

f.pretrain()

f.fit(y)

f.save_checkpoint(my_new_ckpt_path)

# later, it can be loaded in new kernel:

f = MyDLmodel(checkpoint=my_new_ckpt_path)
```
