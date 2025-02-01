# API rework of pytorch-forecasting and dsipts merge (pytorch-forecasting version 2)

Contributors: @fkiraly, @agobbifbk

## High-level summary 

### The Aim

To create a unified interface for `torch` forecasters, in `pytorch-forecasting` version 2,
suitable for the entire ecosystem of `torch` based forecasters, including `dsip-ts`,
various foundation models, and inspired by `dsip-ts` and `time-series-library`.

### Context

Over 2024, in `sktime`, interfaces to a variety of deep learning and foundation models have been added,
amongst these, `pytorch-forecasting`, and various foundation models.

This exercise showed that the `pytorch-forecasting` design does not generalize to foundation models,
and made some limitations of the package apparent, such as strong coupling to `pandas` and `scikit-learn`
which prevents large-scale use.

At the same time, the `dsip-ts` package (by agobbifbk) emerged, contributing interesting ideas
of API uniformity, and a simple API.

It was decided that both packages - `pytorch-forecasting` and `dsip-ts` - would merge
with the aim to create `pytorch-forecasting v2` - as the "sktime" of `torch` forecasting models.

References:

* `pytorch-forecasting` handover plan (to `sktime`) https://github.com/sktime/pytorch-forecasting/issues/1592
* re-design thread for `pytorch-forecasting` 2.0 with `dsip-ts` https://github.com/sktime/pytorch-forecasting/issues/1736
    * this thread also contains a summary of technical planning meetings and sync design discussions
* umbrella issue `sktime` on foundation models https://github.com/sktime/sktime/issues/6177
* early coordination discussion deep learning, foundation models, `pytorch-forecasting` https://github.com/sktime/sktime/issues/6381
* 2025 `sktime` roadmap with a focus on deep learning and `torch` https://github.com/sktime/sktime/issues/7707

### requirements

* M: unified model API which is easily extensible and composable, similar to `sktime` and DSIPTS, but as closely to the `pytorch` level as possible. The API need not cover forecasters in general, only `torch` based forecasters.
    * M: unified monitoring and logging API, also see https://github.com/sktime/pytorch-forecasting/issues/1700
    * M: extension templates need to be created
    * S: `skbase` can be used to curate the forecasters as records, with tags, etc
    * S: model persistence
    * C: third party extension patterns, so new models can "live" in other repositories or packages, for instance `thuml`
* M: reworked and unified data input API
    * M: support static variables and categoricals
    * S: support for multiple data input locations and formats - pandas, polars, hard drive, distributed, etc
* M: MLops and benchmarking features as in DSIPTS
* S: support for pre-training, model hubs, foundation models, but this could be post-2.0

### The proposed solution

Our proposed solution consists of the following components:

* a two-layered `DataSet` input layer: first layer generic input, second layer. The first layer (layer D1) is unified and model-independent, the second layer (layer D2) is model or model class specific.
* a two-layered model layer: a composite layer. The inner layer is pure `torch` (layer T), the outer layer (layer M) provides a unified interface, is a composite of metadata, references to layer D2, and to layer T.
* downwards compatible migration and refactoring strategies for `pytorch-forecasting`, `dsipts`, and `thuml`, towards a unified whole that also leaves current structures intact.

## Design: `pytorch-forecasting` 2.0

### Conceptual model and layers

Following discussions collected in the linked issue [1736](https://github.com/sktime/pytorch-forecasting/issues/1736),

the design consists of four layers as mentioned above:

* layer D1: unified `DataSet` interface
* layer D2: model specific `DataSet` and `DataLoader` interface
* layer T: raw `torch` models
* layer M: unified model layer: models with metadata and reference to d2 layer

Reasoning for the layers:

* both `DataSet` and `torch` model input/output tend to be specific to implementation.
From examples seen, it is unlikely that they can be sensibly unified.
* therefore, additional unification layers are needed, one for data and one for models.
* discussions in the issue (also motivated by notes of janbeitner in the original code)
converged on two `DataSet` based layers, using the standard `DataLoader`.
* this implies a second unification layer on the other side, for models - given that
unifying model interfaces is the primary goal.

Conceptually, layers align with concepts as follows:

* layer D1: abstract data type of "collection of time series", a `Panel` in `sktime`
parlance. Implementation
can be arbitrary, e.g., `pandas` or `polars`, or hard-drive files.
* layer M: abstract model taking in the `Panel` data for training or inference.
This *includes* data pre-processing, re-sampling, batching.
* layer T: concrete neural network with free parameters, *excluding* data pre-processing, re-sampling, batching.
This starts at data that is already pre-processed, re-sampled, batched.
* layer D2: M minus T.

### Alignment of current packages with layers

`pytorch-forecasting`:

* currently has two layers, a data layer and a model layer
* data layer = D1 plus D2 plus M (lasagna) = `TimeSeriesDataSet`
* model layer = T
* `BaseModel` is similar to M, but assumes data layer
* in particular, there is no uniformization layer for data or models that would cover, e.g., foundation models
* this also makes the design of very limited extensibility beyond certain decoder/encoder models

`dsip-ts`:

* currently has three layers
* pre-processing functions, prior to use of `DataSet` - D1
* data set and data loader: D2 plus M
* model layer = T
* improvement compared to `pytorch-forecasting`, because there is a data uniformization layer
    * but unfortunately D1 is not in the form of `DataSet` which would allow scaling
    * model uniformization layer from layer D2 onwards, but not D1


### mid-level interface designs

#### layer D1

Aim: model `Panel` data as closely as possible, while satisfying data requirements

Data requirements:

* agnostic towards data location - `pandas`, `polars`, hard drive
* capturing metadata: numeric/categorical, past/future known, dynamic/static

Design:

* `DataSet` extension API with unified `__getitem__` output, defined by `BaseTSDataSet`
* `__init__` captures input that can vary
    * for downwards capability, current inputs in `pytorch-forecasting` and `dsipts` supported
* inheritance pattern and strategy pattern
* simplest-as-possible `__getitem__` return

##### interface: proposed `__getitem__` return of `BaseTSDataSet`

As implemented in draft [PR 1757](https://github.com/sktime/pytorch-forecasting/pull/1757)

Precise specs to be discussed.

```
    Sampling via ``__getitem__`` returns a dictionary,
    which always has following str-keyed entries:
    * t: tensor of shape (n_timepoints)
      Time index for each time point in the past or present. Aligned with ``y``,
      and ``x`` not ending in ``f``.
    * y: tensor of shape (n_timepoints, n_targets)
      Target values for each time point. Rows are time points, aligned with ``t``.
      Columns are targets, aligned with ``col_t``.
    * x: tensor of shape (n_timepoints, n_features)
      Features for each time point. Rows are time points, aligned with ``t``.
    * group: tensor of shape (n_groups)
      Group ids for time series instance.
    * st: tensor of shape (n_static_features)
      Static features.
    * y_cols: list of str of length (n_targets)
      Names of columns of ``y``, in same order as columns in ``y``.
    * x_cols: list of str of length (n_features)
      Names of columns of ``x``, in same order as columns in ``x``.
    * st_cols: list of str of length (n_static_features)
      Names of entries of ``st``, in same order as entries in ``st``.
    * y_types: list of str of length (n_targets)
      Types of columns of ``y``, in same order as columns in ``y``.
      Types can be "c" for categorical, "n" for numerical.
    * x_types: list of str of length (n_features)
      Types of columns of ``x``, in same order as columns in ``x``.
      Types can be "c" for categorical, "n" for numerical.
    * st_types: list of str of length (n_static_features)
      Types of entries of ``st``, in same order as entries in ``st``.
    * x_k: list of int of length (n_features)
      Whether the feature is known in the future, encoded by 0 or 1,
      in same order as columns in ``x``.
      0 means the feature is not known in the future, 1 means it is known.
    Optionally, the following str-keyed entries can be included:
    * t_f: tensor of shape (n_timepoints_future)
      Time index for each time point in the future.
      Aligned with ``x_f``.
    * x_f: tensor of shape (n_timepoints_future, n_features)
      Known features for each time point in the future.
      Rows are time points, aligned with ``t_f``.
    * weight: tensor of shape (n_timepoints), only if weight is not None
    * weight_f: tensor of shape (n_timepoints_future), only if weight is not None
```

##### Extension pattern

* inherit from `BaseTSDataSet`
* custom `__init__` input, can be anything, including file locations
* dataclass-like
* logic only needs to comply with `__getitem__` expectation


#### layer D2

Aim: prepare unified data input from layer D1 for `torch` model

Design:

* `DataSet` extension API with unified `__init__` input, expecting `BaseTSDataSet`
* further `__init__` fields may be arbitrarily present, dataclass-like
* `__getitem__` return is specific to a limited range of `torch` models
* default assumption is standard `DataLoader`
* optionally, custom `DataLoader` may be supplied

##### Example, based on current `pytorch-forecasting` models

Current `TimeSeriesDataSet(data, **params)` to be replaced with

```python
tsd = PandasTSDataSet(df, **metadata)  # layer D1
DecoderEncoderData(tsd, **params_without_metadata)  # layer D2
```

* where `metadata` is as above in layer D1
* and `params_without_metadata` contains decoder/encoder specific variables
    * `max_encoder_length`
    * `min_encoder_length`
    * `max_decoder_length`
    * `min_decoder_length`
    * `constant_fill_strategy`
    * `allow_missing_timesteps`
    * `lags`
    * and so on

The return of the `DecoderEncoderData` instance` should be exactly the same
as of current `TimeSeriesDataSet`, when invoked with equivalent parameters and data.

##### Example, based on current `dsip-ts` models

For custom data, this should work

```python
tsd = PandasTSDataSet(df, **metadata)  # layer D1
DsiptsPipeline(tsd, **params_without_metadata)  # layer D2
```

For pre-defined datasets, this should work

```python
tsd = BenchmarkDataSet(name:str, config)  # layer D1
DsiptsPipeline(tsd, **params_without_metadata)  # layer D2
```

#### layer T

The model layer contains layers and full models using `pytorch-lightning` interfaces.

These are simple loose classes as currently present in all packages, i.e., `nn.Module` subclasses.


#### layer M

Some unknowns here and work in progress.

Suggested design:

* class design: metadata class with pointer to layer T and D2, plus metadata
* `scikit-base` compatible collection of parameters
    * neural network parameters (`torch.nn`)
    * training and inference parameters
* switch between training and inference mode
* directly interfaces with layer D1 on the outside
    * possibility to construct `from_dataset` or similar, like in ptf


```python
class MyNetwork(BasePtfNetwork):

    _tags = {
        "capability:categorical": True,
        "capability:futureknown": False,
        "capability:static": False,
        "etc"
    }

    def __init__(
        self,
        **network_params,
        **network_configs,
        **loader_params,
    )

    def ref_network(self):  # pointer to network, could be more complicated
        from somewhere import MyTorchNetwork

        return MyTorchNetwork

    def ref_dataloader(self):  # pointer to dataloader
        from somewhere import D2LoaderForMyTorchnetwork

        return D2LoaderForMyTorchnetwork

    @classmethod
    def from_dataset(cls, dataset):  # sets parameters from dataset
        return cls(**get_params_from(dataset))

    def should_we_forward_lignthing_methods(self, **kwargs):  # ?

    def train(self, dataset):
        # logic related to training

    def predict(self, dataset)
        # logic related to inference
```


#### usage vignette

should maybe center more around the data loader

```python

data_loader = my_class(configs).get_dataloader(more_configs)

# need training and validation data loader separately
data_loader_validation = my_class(configs).get_dataloader(more_configs)

```

action AG - can you write a speculative usage vignette?

Let us use `lightning` as much as possible?

Change the class as necessary



## Change and deprecation

### `pytorch-forecasting`

* Networks can be left as-is mostly, for downwards compatibility

* `TimeSeriesDataSet` should alias this, see above

```python
tsd = PandasTSDataSet(df, **metadata)  # layer D1
DecoderEncoderData(tsd, **params_without_metadata)  # layer D2
```

It should be possible to keep interfaces as-is with this aliasing.

### `dsip-ts`

* need to introduce a D1-to-D2 `DataSet`
* current pipeline can still be used

## Implementation phases

### Phase 0 - design

Agreement on this design document and target state

### Phase 1 - `DataSet` layer

Suggested to use `pytorch-forecasting` and introduce D1/D2 separation as an API
preserving refactor of `TimeSeriesDataSet`, as follows:

1. add D1 `BaseTSDataSet` and `PandasTSDataSet` child class, and tests
2. add `DecoderEncoderData` to obtain interface on par with `TimeSeriesDataSet`
3. change `TimeSeriesDataSet` to alias the D1/D2 composite
4. add one or two further `BaseTSDataSet` as proof-of-concept: `polars` or hard drive files
    * use this to improve `DecoderEncoderData` to avoid too high in-memory usage


### Phase 2a - `dsipts` `DataSet` layer integration

Can start in middle of phase 1, at a stage where `BaseTsDataSet` is consolidated.

1. refactor current data pipeline to be a single `DataSet` class.
2. rebase pipeline on `BaseTSDataSet` interface, ensure refactor and API consistency

### Phase 2b - Model layer

1. `BasePtfNetwork` experimental design and full API tests, using phase 1 objects
2. refactor at least two `pytorch-forecasting` models to this design, design iteration

### Phase 3 - ecosystem

* `dsip-ts` models
* `pytorch-forecasting` models
* `thuml` models
