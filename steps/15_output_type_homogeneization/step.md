# Output type homogeneization

Contributors: @fkiraly, @mloning

## High-level summary 

### The Aim

Currently, `sktime`'s estimators produce outputs that are of same mtype as the input mtype,
to the "corresponding" input. For instance:
* forecaster `predict` will produce `pd.Series` if `fit` is given a `pd.Series`
* forecaster `predict` will produce `pd.DataFrame` if `fit` is given a `pd.DataFrame`
* transformer `transform` will produce `np.ndarray` (if possible) if given an `np.ndarray` as input
* transformer `transform` will produce `pd.DataFrame` if given a `pd.DataFrame` as input

The aim is to move to a situation where the output is always the same for a given estimator class, e.g.,:
* forecaster `predict` will produce `pd.DataFrame` if `fit` is given a `pd.Series`
* forecaster `predict` will produce `pd.DataFrame` if `fit` is given a `pd.DataFrame`
* transformer `transform` will produce `pd.DataFrame` if given an `np.ndarray` as input
* transformer `transform` will produce `pd.DataFrame` if given a `pd.DataFrame` as input

### discussion of alternatives, pros/cons

The current output=input design may lead to the following counterintuitive situations:
* transformer `transform` will produce `pd.DataFrame` if given a `pd.Series` as input, in a case
where the transformer adds columns. If the number of columns added depend on estimator parameters,
this can be confusing.

The alternative "output is always `pd.DataFrame` if possible" design leads to the following counterintuitive situation:
* forecaster `predict` will produce `pd.DataFrame` if `fit` is given a `pd.Series`
* transformer `transform` will produce `pd.DataFrame` if given an `np.ndarray` as input

Pros/cons have been discussed at length among core developers, with a clear vote
for "output always `pd.DataFrame`".

### The proposed solution

The proposed solution will:
* change the output behaviour in forecasters to always return `pd.DataFrame` (and mtype of the same name), in `predict` and similar methods.
* change the output behaviour in transformers to always return `pd.DataFrame`, in `transform` and similar methods.
The mtype will be `pd.DataFrame` if `Series` scitype, and `pd-multiindex` if `Panel` scitype.
* adding constructor tags `X_output_mtype` and/or `y_output_mtype` that allow to override thie behaviour.
This is useful to avoid superfluous back/forth conversions in pipelines and composites (e.g., when chaining
many components with inner mtype not being `pd.DataFrame`).
* adding a global override for the constructor tags

## Solution overview

### Fixed output format

The specification is as above, and will require changes in `BaseForecaster` as well as `BaseTransformer`.

This will be achieved by hard-coding the output mtypes based on output scitype.

### output mtype override

All forecasters and transformers will have an additional optional argument in their constructor, `X_output_mtype` (for transformers) resp `y_output_mtype` (for forecasters).

This can take values that are mtype strings, `"default_output_mtype"` (default), `"same_as_input"`, or `"no_conversion"`.

Behaviour is as follows:
* if `"default_output_mtype"`, the behaviour above in "fixed output format"
* if `"same_as_input"`, the current behaviour
* if `"no_conversion"`, no conversion is done before the output is returned
* if an mtype string, conversion is attempted to that mtype; error is raised if the scitype does not match

Estimator creation would then look like this:

```python
    my_estimator = MyEstimatorClass(42, param2=4242, X_output_mtype="np.ndarray")
```

Most pipelines will likely use overrides using `set_params` to `"no_conversion"`.

### global override

A global override variable `OUTPUT_BEHAVIOUR` will be introduced.
This allows to override the default behaviour in all estimators, with one of
`"default_output_mtype"`, `"same_as_input"`, `"no_conversion"`.

Estimator parameters will override the global override.

## Development steps

### Step 1 - implement the behaviour, set default to `"same_as_input"`

This will introduce the functionality while keeping the interface intact.
Modifications will be localized around the `convert_to` invocations near the outputs.

The constructor will have default settings of `None`, in which case the global override will be read.

### Step 2 - deprecation message

This is a major change to the core interface, so 3 minor cycles notice should be given.
In all release notes during this cycle, this upcoming change is prominently displayed.

### Step 3 - removal

The global default is changed to `"default_output_mtype"`.

In strategic locations, errors and notes will point to the possibility to set the global override
`OUTPUT_BEHAVIOUR="same_as_input"`. 
The best locations for this are yet to be determined.

This is to allow users to preserve functionality of legacy code, without having to
work through the code in a potentially large number of output interface locations.
