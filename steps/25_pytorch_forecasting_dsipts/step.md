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
with the aim to creat `pytorch-forecasting` - as the "sktime" of `torch` forecasting models.

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
* a two-layered model layer: a composite layer. The inner layer is pure `torch` (layer T), the outer layer (layer C) is a composite of metadata, references to layer D2, and to layer T.
* downwards compatible migration and refactoring strategies for `pytorch-forecasting`, `dsipts`, and `thuml`, towards a unified whole that also leaves current structures intact.

## Design: foo

### Conceptual model: bar


### foo

## Change and deprecation

### Change of foo

## Implementation phases

### Phase 1 - bar
