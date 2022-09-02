# Save/Load for DL Estimators

Contributors: ['AurumnPegasus']

## Introduction

Initially proposed in sktime issue [#3128](https://github.com/alan-turing-institute/sktime/pull/3128), we need to introduce a `save` and `load` functionality for estimators so as to easily store and load fitted models.

Later, Franz introduced a design for save/load estimators in a more general way in sktime issue [#3336](https://github.com/alan-turing-institute/sktime/pull/3336), and the solution I plan to propose here is built on the same.

## Contents

[TOC]

## Problem Statement


### Current Implementation

The current implementation for `serialisation` and `deserialiation` is based on `__getstate__` and `__setstate__` functions implemented in `sklearn`'s  `BaseEstimator`. It is done using pickle, where the user simply has to:

```python
import pickle
vecm = VECM()
vecm.fit(train, fh=fh)
save_output = pickle.dumps(vecm)
----------------------------------------------
model = pickle.loads(save_output)
model.predict(fh=fh)
```

### Problems

The issue here is that for general DL Estimators, you cannot do that, because of the `optimizer` parameter. The `optimizer` parameter uses lambda function in its inherent implementation, which can not be pickled in a straightforward manner. 

Hence, we need to find a better and more general solution which would allow us to save and load the DL estimators as well.

## Solution

In this case, we want to use the base design proposed by Franz in   [#3336](https://github.com/alan-turing-institute/sktime/pull/3336).

As proposed by him:

In the BaseObject class, we add three functions:

```python
def save(self, path=None):
    import pickle
    if path is None:
        return (type(self), pickle.dumps(self))
    from zipfile import ZipFile
    with ZipFile(path) as zipfile:
        with zipfile.open("metadata", mode="w") as meta_file:
            meta_file.write(type(self))
        with zipfile.open("object", mode="w") as object:
            object.write(pickle.dumps(self))
    return ZipFile(path)
def load_from_serial(cls, serial):
    import pickle
    return pickle.loads(serial)
def load_from_path(cld, serial):
    import pickle
    return pickle.loads(serial)
```

For DL Estimator, we will overwrite this in a base class for all DL Estimators (which is in design phase currently [#26](https://github.com/sktime/enhancement-proposals/pull/26))

```python
class BaseDeepClass():
    def __getstate__(self):
        out = self.__dict__.copy()
        del out['optimizer']
        del out['optimizer_']
        return out
    def save(self, path=None):
        import pickle
        if path is None:
            return (type(self), pickle.dumps(self))
        from zipfile import ZipFile
        with ZipFile(path) as zipfile:
            with zipfile.open("metadata", mode="w") as meta_file:
                meta_file.write(type(self))
            with zipfile.open("object", mode="w") as object:
                object.write(pickle.dumps(self))
            with zipfile.open("model", mode="w") as model:
                model.write(self.model_.save(path))
        return ZipFile(path)
    def load_from_path(cls, serial):
    # supposed to return the keras model directly
        return keras.load(serial)
```


