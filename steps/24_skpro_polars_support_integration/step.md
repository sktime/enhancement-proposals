# Improving polars support for skpro Design Document

Contributors: [https://github.com/julian-fong]

## Introduction

This design document contains a roughly consolidated list of potential ideas that could be implemented inside the current skpro package in order to extend the current functionality of polars. Feedback or ideas are extremely welcome. Note that we are primarily dealing with the polars eager table type at the moment, implementation for lazy formatting will be implemented down the line

For preliminary discussions of the proposal presented here, see issue: [https://github.com/sktime/skpro/issues/342]

## Contents

1. Problem statement
2. Current functionality of polars inside `skpro.regression`
3. Extending functionality of `skpro.regression` and description of proposed solution
4. Complete end to end polars support for `skpro` regressors
5. Extending functionality of `skpro.survival` and description of proposed solution
6. Current functionality of polars inside  `skpro.distribution`
7. Extending functionality of `skpro.distribution` and description of proposed solution
8. Extending base classes to incorporate a potential `set_output` like functionality
9. Other ideas
10. References

## 1) Problem statement/current polars behaviour

Currently functionality of polars inside  `skpro` is currently limited to conversion between pandas and polars eager tables back and forth. This document is a consolidated list of potential extensions that could possibly be implemented to improve polars support on current estimators. The current next steps listed inside #342 is as follows:

1. try `polars` input/output with a few estimators, `fit` / `predict` only. Add tests in a dedicated `polars` test file, non-systematic for the start.
2. `sklearn` now supports `polars` - so, we should try to pass on `polars` frames in some of the estimators, extending the `X_inner_mtype` to both pandas and polars (in a list of str, with `polars_eager_table`). Example: `GaussianProcess`
3. if that works well, we should extend other regresssors wrapping `sklearn` in the same way.
4. next I would work on `Pipeline`, a composite. The `Pipeline` is native to `skpro`.

Polars does not support multi-index indicies or columns. `skpro` primarily uses multi-index columns in some of the `predict_*` functions inside the regression models.

Polars also does not support indexing inside their dataframes.

## 2) Current functionality of polars inside `skpro.regression`

Current functionality of polars inside `skpro` is currently limited to conversions to and from pandas to polars. Conversion functions can be found through `skpro.datatypes._table._convert` and check functions can be found through `skpro.datatypes._table._check`. In the registry for tables - "polars_eager_table" and "polars_lazy_table" are polars Dataframe representations of the data table

The regression models inside `skpro` currently supports functions `fit`, `predict`, which are mandatory to be implemented - and one of `_predict_proba`, `_predict__interval`, `predict_quantile` , and one more non-mandatory function `predict_var`.

The boilerplate layer `BaseProbaRegressor.predict` uses the mtype seen in `fit` and converts the prediction `y_pred `into the same format. Therefore any regression estimator will automatically convert it to the corresponding mtype if `predict` is called i.e if a polars dataframe is passed in during `fit`, then:

* polars will return polars when `predict` is called
* pandas will return pandas when `predict` is called

The output of `_predict_*`  functions will return pandas Dataframes no matter the input passed in as `X`.

~~Question 2.1~~: For underlying interfaces that currently do not support polar Dataframe inputs (eg. statsmodels), is the expectation that we will refactor code that converts the data container into something it accepts or will we use the `_tags` dict via `X_inner_mtype` and `y_inner_mtype` convention to specify to the user what types of data containers it accepts. So for `statsmodels` case, polars Dataframes will be prohibited.

~~Question 2.2~~: Follow up to question 2.1, should we go through every implemented regression estimator right now to see which data containers it accepts, as every interface could be different. Dummy data should be able to suffice?

## 3) Extending functionality of polars inside `skpro.regression` and description of proposed solution for output containers

Lets consider the current inplementation of `predict_interval` seen inside `regression._base`. Assuming we follow section 8.2's implementation, the proposed implementation of a function like `predict_interval` could look as follows:

```python
def predict_interval(self, X=None, coverage=0.90):
        """Compute/return interval predictions.

        If coverage is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas/polars DataFrame, must have same columns as X in `fit`
            data to predict labels for
        coverage : float or list of float of unique values, optional (default=0.90)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame or pl.DataFrame
            For pd.DataFrame: Column has multi-index: first level is variable name from ``y`` in fit,
            second level coverage fractions for which intervals were computed,
            in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is equal to row index of ``X``.
            Entries are lower/upper bounds of interval predictions,
            for var in col index, at nominal coverage in second col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end are equivalent to
            quantile predictions at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.

	    For pl.DataFrame: Column is in single level following similar convention to pd.DataFrame
	    There are three values seperated by underscores "_". The first value indicates the
	    variable name from ``y`` in fit, the second value is the coverage fractions 
	    from which the intervals were computed, and the third value is lower/upper for interval
	    end
        """
        # check that self is fitted, if not raise exception
        self.check_is_fitted()

        # check alpha and coerce to list
        coverage = self._check_alpha(coverage, name="coverage")

        # check and convert X
        X_inner = self._check_X(X=X)

        # pass to inner _predict_interval
        pred_int = self._predict_interval(X=X_inner, coverage=coverage)

	*valid, output_config = _check_output_config(self)
	*if valid:
		convert_to_mtype, convert_to_scitype = _transform_output(output_config)
	else:
		convert_to_mtype = self._y_metadata["mtype"]
		convert_to_scitype = "Table"

	*pred_int = convert(
		pred_int, 
		from_type = self.get_tag("y_inner_mtype"),
		to_type=convert_to_mtype,
		as_scitype=convert_to_scitype,
		store=self._X_converter_store,
		)
        return pred_int

```

We insert a few new lines denoted by * to indicate new functionality. First we ensure that the user has utilized the `set_output` function to pass in a new transform. If the user has a "transform" defined (see section 8.2), then we first check that it is a valid "transform" via `_check_output_config`. Once we have a valid output container from the user, we locate the corresponding `mtype` and `scitype` based on the function in question ('Proba' for quantiles and interval, and 'Table' for var and predict) using `_transform_output`. Finally, the conversion methods in `convert` will then handle the conversions between the original type to the desired user output data container.

If the `set_output` function was not used (i,e valid is False), then we skip this code block and move straight to return the prediction dataframe.

Similar adaptations can be followed inside `predict_quantiles`, `predict_var` and. We will first call `._check_X` and convert input X if necessary into the correct mtype. Then, if the user specified a new transform container through `set_output`, we verify that the transform value passed in from the user is a valid key, and then call the convert output function to transform.

To summarize, all `predict_` functions inside `regression._base.py` will have the new code marked in * to facilitate the checking, convert of the output data container. Validations for user specified output data container will be validated through `_check_output_config`, and then the correct mtype/scitypes will be obtained through `_transform_output`. The actual conversion method `convert `will then convert the data container as specified by the user using `set_output`.

### Potential Conversion combinations

###### Pandas to Polars

In order to facilitate the correct formatting of `_predict_*` functions using polars dataframes, we need to melt down the multi-index columns and convert them into single column format. `sklearn` formatting will be adopted and columns will be named using the double underscore convention. Any instance of a return pandas Dataframe which contains a multi-index column will use this function to convert it into single column format in the polars DataFrame.

Solution code can be found below:

```python
def convert_multiindex_columns_to_single_column(X: pd.DataFrame):
	#make a copy of X
	X_ = X
	cols = []
      	for col in X.columns:
          	cols.append("__".join(str(x) for x in col))
      		X_.columns = cols

      	return X_

```

If a `_predict_*` function returns a pandas DataFrame with a multi-index column, then we need to melt the columns down into a single column DataFrame before conversion to a polars DataFrame

Proposed column formatting of `predict_interval` dataframes in polars:

```python
┌────────────────────────┬────────────────────────┐
│ __target__0.9__lower__ ┆ __target__0.9__upper__ │
│ ---                    ┆ ---                    │
│ f64                    ┆ f64                    │
╞════════════════════════╪════════════════════════╡
│ 66.772658              ┆ 176.318973             │
│ 22.517743              ┆ 132.064058             │
│ 20.072116              ┆ 129.618431             │
│ 177.079295             ┆ 286.62561              │
│ 86.009941              ┆ 195.556256             │
│ …                      ┆ …                      │
│ 94.49407               ┆ 204.040385             │
│ 60.953847              ┆ 170.500162             │
│ 198.167328             ┆ 307.713643             │
│ 196.372881             ┆ 305.919196             │
│ 82.782263              ┆ 192.328578             │
```

Proposed column formatting of `predict_quantiles` dataframes in polars:

```python
┌──────────────────┬─────────────────┬──────────────────┐
│ __target__0.05__ ┆ __target__0.1__ ┆ __target__0.25__ │
│ ---              ┆ ---             ┆ ---              │
│ f64              ┆ f64             ┆ f64              │
╞══════════════════╪═════════════════╪══════════════════╡
│ 66.772658        ┆ 78.870513       ┆ 99.085499        │
│ 22.517743        ┆ 34.615598       ┆ 54.830583        │
│ 20.072116        ┆ 32.169971       ┆ 52.384956        │
│ 177.079295       ┆ 189.17715       ┆ 209.392136       │
│ 86.009941        ┆ 98.107796       ┆ 118.322782       │
│ …                ┆ …               ┆ …                │
│ 94.49407         ┆ 106.591925      ┆ 126.806911       │
│ 60.953847        ┆ 73.051702       ┆ 93.266688        │
│ 198.167328       ┆ 210.265184      ┆ 230.480169       │
│ 196.372881       ┆ 208.470736      ┆ 228.685722       │
│ 82.782263        ┆ 94.880118       ┆ 115.095104       │
└──────────────────┴─────────────────┴──────────────────┘
```

If a `predict_*` function does not require any melting of columns, then we convert the pandas DataFrame to a polars DataFrame as normal. As an example, consider `predict_var`

```python
┌─────────────┐
│ target      │
│ ---         │
│ f64         │
╞═════════════╡
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
│ …           │
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
│ 1108.871039 │
└─────────────┘
```

As per discussion idea 0: If a user wishes to include the index, custom functions can be written to pass in the pandas index. in `skpro`, we will currently assume/limit the index to be single level indices. Then if a user wishes to convert the `predict` output into a polars DataFrame, it will be shown as:

```python
┌───────────┬────────────┐
│ __index__ ┆ target     │
│ ---       ┆ ---        │
│ i64       ┆ f64        │
╞═══════════╪════════════╡
│ 4         ┆ 121.545815 │
│ 63        ┆ 77.2909    │
│ 10        ┆ 74.845273  │
│ 0         ┆ 231.852453 │
│ 35        ┆ 140.783099 │
│ …         ┆ …          │
│ 39        ┆ 149.267228 │
│ 40        ┆ 115.727004 │
│ 16        ┆ 252.940486 │
│ 44        ┆ 251.146038 │
│ 45        ┆ 137.555421 │
└───────────┴────────────┘
```

and for frames that require melting:

```python
┌───────────────┬────────────────────────┬────────────────────────┐
│ ____index____ ┆ __target__0.9__lower__ ┆ __target__0.9__upper__ │
│ ---           ┆ ---                    ┆ ---                    │
│ i64           ┆ f64                    ┆ f64                    │
╞═══════════════╪════════════════════════╪════════════════════════╡
│ 4             ┆ 66.772658              ┆ 176.318973             │
│ 63            ┆ 22.517743              ┆ 132.064058             │
│ 10            ┆ 20.072116              ┆ 129.618431             │
│ 0             ┆ 177.079295             ┆ 286.62561              │
│ 35            ┆ 86.009941              ┆ 195.556256             │
│ …             ┆ …                      ┆ …                      │
│ 39            ┆ 94.49407               ┆ 204.040385             │
│ 40            ┆ 60.953847              ┆ 170.500162             │
│ 16            ┆ 198.167328             ┆ 307.713643             │
│ 44            ┆ 196.372881             ┆ 305.919196             │
│ 45            ┆ 82.782263              ┆ 192.328578             │
└───────────────┴────────────────────────┴────────────────────────┘
```

###### Polars to Pandas

Requires a conversion during the input check from polars to pandas. If this is not handled already in skpro, we can potentially enhance the current functionality within `_check_X` and `_convert` to check what format the input is in and then change it if necessary. Another section may be required to facilitate any code changes as this section only deals with polars output containers.

~~Question 3.1~~) is this already handled? in the scenario where a polars dataframe is passed and `skpro` automatically converts it into a pandas DataFrame to calculate the predictions?

~~Question 3.2~~) What should we do with the `predict` function. Currently it automatically converts whatever is passed into the `predict` function back into the mtype that was seen in fit. Do we need to refactor this as well?

~~Question 3.3)~~ Currently there is no way to allow the user to specify whether they want or do not want to pass back the index when converting from pandas to polars. The user currently only can specify what transform they want via `set_output` and there is no other parameter they can set to specify what configuration they want as the internal code handles the rest of the conversion. How should we tackle this problem?

~~Question 3.4)~~ Is there a way to check via the tags or configs as to which data container the interface is developed in regards to the private `_predict_*` methods? As an example, the `GLMRegressor` has tag 'coded_in_pandas' so we know that all the private `_predict_*` functions are written using pandas

###### Polars to Polars

Requires a round trip conversion from polars input to pandas input to compute the predictions, then back to a polars output via `create_container`. Since the input is a polars DataFrame, it must be converted before the private `_predict_*` method is called. However since this section deals only with outputs in polars DataFrames, is it out of scope. Perhaps another section on how to deal with polars DataFrames during input checking will be facilitated. Note: This is the situation where polars is not defined as a valid mtype for the given regression estimator

## 4) Complete end to end polars support for `skpro` regressors

Supporting end to end polars is a state in which the use of pandas is bypassed entirely. This means that given a specified data container of polars and the estimator also supports polars, there would be no need to handle the middle conversions in between (for estimators that would not support polars inhenerently, see section 3 'Polars to Polars')

To facilitate this, the estimators must contain the value 'polars_eager_table' inside the tags `X_inner_mtype` and `y_inner_mtype`.

Estimator that contain the 'polars_eager_table' tag inside inner mtypes must be able to do the following

1) Boilerplate must be able to handle polars eager tables inputs directly and output them back via the boilerplate code. (already handled?)
   1) should handle the case where 'polars_eager_frame' is passed
   2) should handle the cases where something else is passed i.e 'pandas DataFrame' or some other mtype
2) Inverse directions from section 3. Given `set_output` user specified transformations, all test cases also must pass

Testing will be done to see if any other implementation is required. Need to see current behaviour of a 'polars_eager_table' only estimator and the behaviour if it is part of a list.

## 5) Extending base classes to incorporate a potential `set_output` like functionality

#### 5.1) A very rough outline of `sklearn`'s `set_output` functionality

Link to direct example of functionality: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html#sphx-glr-auto-examples-miscellaneous-plot-set-output-py

Certain `sklearn` modules currently support a build-in method that transforms dataframes into a certain data container when specified. Modules that support `set_output` inherit a parent class called `_SetOutputMixin`. In the below snippet

```python
scaler2 = StandardScaler()

scaler2.fit(X_train)
X_test_np = scaler2.transform(X_test)
print(f"Default output type: {type(X_test_np).__name__}")
#Default output type: ndarray
scaler2.set_output(transform="pandas")
X_test_df = scaler2.transform(X_test)
print(f"Configured pandas output type: {type(X_test_df).__name__}")
#Configured pandas output type: Dataframe

```

we can see an initialization of a `sklearn` module and we can see the `set_output` in use, such that whenever the `transform` function is subsequently called, the resulting mtype is a pandas Dataframe.

Under the hood, the `set_output` function on line 7 sets a new dictionary attribute to the instance of the object called `_sklearn_output_config` in which the key `"transform"` is set to whatever value is passed in as the argument - in this case "pandas".

When the function `scaler2.transform` is called, the `transform` (probably analogous to the `_predict_*`functions) function then uses various methods `_get_output_config` and `create_container` to ensure a valid "transform" input (either polars or pandas) and then subsequently builds the correct data container specified inside the `sklearn_output_config` attribute.

Function `create_container` is uniquely defined based on the data container adapter specified. There are two classes `PolarsAdapter` and `PandasAdapter` that hold all of the relevant functions that facilitates conversion between various mtypes

All adapter classes and functions `create_container` and `_get_output_config` are specified inside `utils._set_output.py`

#### 5.2) Solution for `skpro` following `sklearn` by introducing `set_output`

`BaseProbaRegressor` will introduce a new function `set_output` so that users will be able to call `set_output` in a similar style to `sklearn`.

```python
def set_output(self, transform_output):
        """Set output container.

        Parameters
        ----------
        transform_output : {"polars", "pandas", "default"}

            Configures the out of any _predict_* function in regression estimators
                - "default" : assumes no transform has been passed in, will use
                default settings

                - "pandas" : all outputs will be converted into pandas DataFrames
                - "polars" : all outputs will be converted into polars DataFrames
            default = "default"
        Returns
        -------
        self : estimator instance
        """
        if transform_output is None:
            return self

        self.set_config(**{"transform_output": transform_output})
        return self
```

Possible values for `set_output` will be (for now) 'polars' and 'pandas', specifying polars and pandas outputs respectively. For polars and pandas, it will always be in DataFrame format (so no pandas Series etc will be possible outputs)

#### 5.3) `set_output` mappings for `table` and `proba` scitypes

In section 3, we described the possible output dataframes for `predict_interval`, `predict_quantiles` and `predict_var`. These functions exist under the `Proba` scitype, while `predict` is in the `table` scitype. 

In our proposed solution, we are proposing that a certain mapping method is designed to ensure the correct output is returned. Since the conversion methods in skpro are done via 'mtype' and 'scitype', this mapping method is required in order to translate the user specified output into an mtype/scitype pair.

The dictionary `SUPPORTED_OUTPUT_MAPPINGS` describes a key value pair where the key is an available `set_output` parameter and the corresponding value is a list of mtype/scitype pairs.

```python
SUPPORTED_OUTPUT_MAPPINGS = {
    "pandas": [("pd_DataFrame_Table", "Table"), ("pred_interval_pandas", "Proba"), ("pred_quantiles_pandas", "Proba"), ("pred_var_pandas", "Proba")],
    "polars": [("polars_eager_table", "Table"), ("pred_interval_polars", "Proba"), ("pred_quantiles_polar", "Proba"), ("pred_var_polar", "Proba")]
}
```

The `SUPPORTED_OUTPUT_MAPPINGS` mappings will be used in the function `_check_output_config`, which ensures that the argument passed via `set_output` is correct. 

```python
def _check_output_config(estimator):
    """Given an estimator, verify the transform key in _config is available.

    Parameters
    ----------
    estimator : a given regression estimator

    Returns
    -------
    dense_config : a dict containing the specified mtype user wishes to convert
        corresponding dataframes to.
        - "dense": specifies the mtype data container tuple (mtype, scitype)
                   in the transform config
            Possible values are located in SUPPORTED_OUTPUTS in
            `skpro.utils.set_output`
    """
    output_config = {}
    transform_output = estimator.get_config()["transform_output"]
    # check if user specified transform_output is in scope
    if transform_output not in SUPPORTED_OUTPUTS:
        raise ValueError(
            f"set_output container must be in {SUPPORTED_OUTPUTS}, "
            f"found {transform_output}."
        )
    elif transform_output != "default":
        valid = True
        output_config["dense"] = SUPPORTED_OUTPUT_MAPPINGS[transform_output]
    else:
        valid = False

    return valid, output_config
```

If the value passed inside `_check_output_config` is valid, then we pass back the corresponding list of output mappings. This will be returned as a variable named `output_config`

The function `_transform_output` will then select the correct mtype/scitype pair from the values based on the function that is being used.

```python
def _transform_output(output_config, function_name):
	"""Return the correct specified output container.
	Given a set of possible mtype/scitype pairs, return the correct one given the current function
	"""
	if function_name == "interval":
		#retrieve corresponding interval mtype and scitype pair
		convert_to_mtype = ...
		convert_to_scitype = ...
	if function_name == "quantiles":
		#retrieve corresponding quantiles mtype and scitype pair
		convert_to_mtype = ...
		convert_to_scitype = ...
	etc..

	return [convert_to_mtype, convert_to_scitype]
```

In the end state, based on what function is being is being called by the user, the code should be able to obtain the correct 'from_type' and the 'to_type' and the correct scitype. For example:

If the user is using `predict` and the `set_output` is configured to `polars`, then using `SUPPORTED_OUTPUT_MAPPINGS`, we can select the correct to mtype "pd_DataFrame_Table" and the correct scitype "Table", and then utilize skpro's convert function to make that conversion.

If the user is using `predict_interval`, then we pass along the mtype `pred_interval_pandas` and the scitype `Proba`. Note that currently for the `Proba` scitype, there are not any specific data container mtypes, so ideas would be appreciated here how we can design the mapping method and what corresponding mtypes should be applicable

We will need to define data containers and conversion methods that can be set as outputs for the `Proba` scitype.

## 6) Other Ideas/Discussion Items

Discussion item 0: Inclusion of `__index__` column inside all `predict/_predict_*` methods when applying pandas -> polars

* Idea 1 (JF): Can potentially leverage a secondary conversion function that utilizes native polars function `from_pandas`. It supports converting the pandas indices into a single column (not sure if it works for multi-indices) and the single column will be called `None` - can potentially rename the column to `__index__` using the rename function
  * afterwards - if user desires `__index__` to be returned it will be specified using a bool param
    * if `return_index = True` polars output will be 2 or more columns - one with the corresponding index and the remaining columns
    * if `return_index = False` only return relevant predicted values

Discussion item 1: Potential plan to integrate (scope of integration to be determined if implemented) of polars.Series

* Idea 1 (JF): Currenty skpro does not support polar Series. there may be some intuitive sense to incorporate
  * Pro: `sklearn` `train_test_split` function returns pandas series if the input `y` is a series. Users typically split their datasets as `X` and `y` typically through `X = df.drop(columns = ['y_value'], axis = 1)` and `y = df['y_value']` making `y` a pandas Series. `skpro` does not currently support conversion between pandas Series to either polars Series or polars DataFrames. Current conversion methods do not support transfer from pandas Series to polars Series so potential users will encounter an error
  * Pro: skpro and sktime currently support pandas Series - so it makes some intuitive sense to build conversion methods between the two similar to dataframes
  * Con: Hassle dealing incorporating a new Series type and it is not user friendly to limit univariate time series stricktly as pd Series and multivariate as pd DataFrame only
    * Idea 2: to prevent any further refactors or re-implementation of current code design - could potentially only specify a polar Series mtype strictly for `y_train` or `y_test` that are in polars/pandas Series type, just so we have some written code that can handle conversions and or outputs between the two. Ideally `X` is not touched
      * remember that users can also potentially input polars series as as their `y_train` and `y_test` so we need would need to configure polars to polars and pandas to polars
      * Currently skpro does not accept polars.Series (i do not think it will pass `_check_y`)

Discussion item 2: interactions between `skpro.Distribution` and `_predict_proba` - knowledge needed?

Discussion item 3: integrating `set_output` like functionality from `sklearn` inside baseProbRegressor? potentially using `set_config` . exploration needed and ideas welcome

Discussion idea 4: Workflow for future contributers/conversions between mtypes during methods.

* Off the top of my head there are two possibilities:
  * Idea 1: Continue to have the contributer develop only the pandas implementation of the `_predict_*` functions
    * Therefore all polars inputs from X must be pre-converted to pandas DataFrames before entering the `_predict_*` function
    * After the resulting pandas DataFrame is outputted from  `_predict_*`, it is then converted back to `polars` if necessary, using a conversion function
    * This results in less work (and the developer does not need to be familiar with `polars` or implement two different methods for different containers) for developement but does increase the runtime for methods
      * as an example: input polars and output polars would require a method to convert the input polars Dataframe into an input pandas DataFrame, calculate the prediction (predict_interval, predict_quantile, or predict_proba), then convert it back to a polars DataFrame using `create_container`
    * New tests inside `check_estimator` will be required to make sure that the developer's implementation of methods passes checks under `create_container` for other potential mtypes
  * Idea 2: Have the developer implement for both data containers
    * Allows for potentially faster runtime as no extra conversions are required
    * Only one conversion is potentially needed after the output is returned from `_predict_*`
    * Requires custom polars tests to ensure developer constructed polars methods correctly
* Personally leaning towards Idea 1, but feedback and other ideas would be appreciated

#TODO - ideas welcome

## 7) References

* Polars API document: https://docs.pola.rs/py-polars/html/reference/
* Modern Polars: https://kevinheavey.github.io/modern-polars/
* introducing `set_output` in sklearn https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html#sphx-glr-auto-examples-miscellaneous-plot-set-output-py
  * `utils.set_output.py` - https://github.com/scikit-learn/scikit-learn/blob/8d0b243cb53ff609d32ecd7aafc5c098381eac86/sklearn/utils/_set_output.py#L152
