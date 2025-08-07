# Design Document for Prediction pipeline of `ptf-v2`
## Table of Contents
[TOC]

## Aim


Current beta version of `ptf-v2` doesnot have any functionality to do the predicitions and this design document aims to provide some possible ideas to implement the prediction pipeline

## Description of Status Quo (of v1)

The pipeline for v1 is quite straight forward:
1. Define the dataset and pass it to`TimeSeriesDataset`.
2. Initialise the model using `.from_dataset`.
3. Train the model using `Trainer.fit`
4. Perform predictions using `model.predict`.

###  1. Define the dataset and pass it to`TimeSeriesDataset`.

First load the data and define necessary params like `max_encoder_length`, `max_prediction_length` etc

#### Input
* DataFrame with columns: `time_idx`, `target`, `series`, `value`, etc.
* Static features and time-varying features split manually.
* Additional parameters like: 
    * `max_encoder_length`, `max_prediction_length`
    * `categorical_encoders` (e.g. `NaNLabelEncoder`)
    * `group_ids`

#### Type assumptions:
* Data must be properly typed and column-cast before passing.
* `series` must be cast to `str` manually - not inferred.

#### Output:
Instance of `TimeSeriesDataset` that knows the:
* Grouping
* Variable types
* Sequence lengths (input/output)
* Encoders

#### Code Snippets

##### Load Data
```python
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
data = data.astype(dict(series=str)) # confusingly this step is always required to make the data "ready" for the TimSeriesDataset
# One of the pain points of the current API as said by ahmet (a ptf user) as it is not a standard workflow
```

##### Define the Datalaoders and TimeSeriesDataset
```python
# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

# Dataset Creation
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    static_categoricals=[
        "series"
    ],  
    time_varying_unknown_reals=["value"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
batch_size = 128

# Dataloader creation
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)

```

(The code is taken from the [tutorial](https://github.com/sktime/pytorch-forecasting/blob/main/docs/source/tutorials/deepar.ipynb) for `DeepAR`)

### 2. Initialise the model

To initialise the model, we use `.from_dataset`. 

#### Inputs:
* `TimeSeriesDataset`
* Hyperparameters (e.g. `hidden_size`, `loss`, etc.) to the model

#### Type guarantees:
Extracts from dataset:
* input/output sizes,
* variable encoders,
* group information

#### Output:
Fully initialised model (e.g., `DeepAR` instance)

```python
import lightning.pytorch as pl

# Define Trainer
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)


net = DeepAR.from_dataset(
    training, # Dataset
    
    # Hyperparams
    learning_rate=1e-2,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=MultivariateNormalDistributionLoss(rank=30),
)
```

### 3. Train the model
To train the model we make use of `lightning.pytorch.Trainer`

#### Inputs:
* `Trainer` (from `lightning`)
* Dataloaders

#### Guarantees:
* Uses `TimeSeriesDataSet`'s batching
* Optional callbacks (e.g., early stopping, logging)

```python
trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

### 4. Perform predictions using `model.predict`.

Now comes the most interesting part of the pipeline - prediction. `pytorch-forecasting` provides multiple "modes" for predictions like "raw", "quantiles" or "prediction".
* `mode="prediction"`
    * Returns point forecasts, typically the mean (or median) of the predictive distribution.
    * Uses the loss function’s `to_prediction()` method to generate usable forecasts.
* `mode="quantiles"`
    * Uses `to_quantiles()` from the loss .
    * Outputs a tensor shaped `(batch_size, horizon, n_quantiles)`. 
* `mode="raw"` 
    * Returns raw output from `forward()` before any post‑processing.
    * The returned object is a dictionary with all internal outputs.
    * If used with a tuple like `("raw", output_name)`, it extracts a specific field, where `output_name` is a name in the dictionary returned by `forward()`

#### Inputs:
* DataFrame or Dataloader
* mode: "raw", "prediction", "quantiles"
* Optional `trainer_kwargs`

##### Guarantees:
* If given a DataFrame:
    * It wraps it into a `TimeSeriesDataset` (reusing `.from_dataset`)
    * Constructs new dataloaders
* Internally uses `trainer.predict()` with `PredictCallback`

#### Output Types:
Here `N` is the size of validation data
* "prediction" -> tensor of shape`(N, prediction_length)`
* "quantiles" -> tensor of shape`(N, prediction_length, n_quantiles)`
* "raw" -> `dict` pf  raw predictions of `forward()` with shape`(N, prediction_length)` or `(N, prediction_length, params)` (where `params` can be `n_quantiles` or `params` for `DistributionLoss`) depending upon the type of loss we are using.
    * The keys of the output `dict` can have keys other than `"prediction"` as well, like `"decoder_attention"` etc, depending upon the model used. Some models(like `DeepAR`) return just a `dict` with one key - `"prediction"` while other models (like `TFT`) can return other keys as well.

*If you look at the first step, there we do `max_prediction_length=prediction_length` in `TimeSeriesDataset`, so here we can use `prediction_length` and `max_prediction_length` interchangeably*

```python
net.predict(
    val_dataloader, mode="raw", trainer_kwargs=dict(accelerator="cpu")
)
```
`predict` of `BaseModel` convert the data that is provided for the prediction to the `TimeSeriesDataset` (if dataframe is inputted) and creates the dataloaders as well if required.
It internally uses a special callback called `PredictCallback`([source](https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/models/base/_base_model.py#L213)) that is used to return the predictions in the desired mode along with other functionalities.

Some useful functionalities of `PredictCallback`:
* Save the predictions to a specific `output_dir` either at the end of every `epoch`([source](https://github.com/sktime/pytorch-forecasting/blob/81b5303eca5a1dd28e12945ecb1a9d34fa47211e/pytorch_forecasting/models/base/_base_model.py#L349)) or `batch`([source](https://github.com/sktime/pytorch-forecasting/blob/81b5303eca5a1dd28e12945ecb1a9d34fa47211e/pytorch_forecasting/models/base/_base_model.py#L334))
* Provide predictions in a desired mode (raw, quantile or prediction)
* Concatenates tensors across batches with NaN handling.

> ***NOTE: `trainer.predict()` can be used directly for prediction, but `model.predict()` wraps it with additional logic, like converting raw data to TimeSeriesDataSet, preparing loaders, and adding `PredictCallback`***

#### Getting final predictions
To get the final predicitons, `mode="prediction"` is used in `.predict()`.
There if you want to return specific parts (like `x`, actual `y` or `index`), you need to specify it separately.
```python
prediction = net.predict(
    val_dataloader,
    mode="prediction",
    return_index=True,
    return_x=True,
    return_y=True
)
```
*Outputs*
```python
prediction.output  # model's predicted output
prediction.index   # dataframe with time_idx and group identifiers
prediction.x       # original input tensors used (dict with keys like 'encoder_cont', etc.)
prediction.y       # actual target values used during prediction

```
In the above snippet, you will get `predictions` that has `index`, `x` and `y` returned as well and then you can change it to dataframe if you want manually.
> **Thought: Maybe we should add some `to_dataframe` or similar function that does it internally and user dont have to manually transform the tensors to dataframe (in v2).**

### Save and Reload the Model

The user uses `model.save()` and `model.load()` to persist both model weights and necessary metadata like hyperparameters, encoders, and dataset config.

#### Saving the Model
##### Inputs:
* Trained model (instance of a subclass of `BaseModel`)
* Path to save the model checkpoint (`.ckpt` or `.pt`)

##### Guarantees:
Saves:
* `state_dict` (i.e., model weights)
* Hyperparameters
* Categorical encoders and variable transformations (important for reproducibility)

##### Outputs:
A `.ckpt` or `.pt` file containing the model and metadata
```python
# Save the trained model
net.save("deep_ar_model.pt")
```

#### Reloading the Model

##### Inputs:
* Saved checkpoint path

##### Guarantees:
* Restores:
    * Architecture
    * Optimizer state (if using `.load_from_checkpoint()`)
    * All dataset-related metadata
* Ensures model can be used for further training or inference without re-defining encoders manually.

##### Outputs:
A fully initialized model (e.g., `DeepAR`) with weights and configuration loaded
```python
# Reload model from .pt
model = DeepAR.load("deep_ar_model.pt")
```

Alternatively, if using `lightning` checkpointing:

```python
model = DeepAR.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=29-step=1500.ckpt")
```


## Proposal for predict of v2

We should learn from the prediction pipeline of v1 and have some functionalities like v1 in v2 as well.
Some things that we should borrow:
* `mode` in `.predict()` 
 We should allow the user to decide what kind of prediction (fully processed or raw) they want. It should have atleast the already available modes:
    * `prediction`
    * `quantiles`
    * `raw`
* Use of `PredictCallback`
We could use `trainer.predict()` to make the predictions, but `PredictCallBack` provides some special customisations like a way to save the predictions directly in a `output_dir` and customised prediction writing to the output_dir at epoch_end or batch_end. `model.predict()` will internally call `trainer.predict()` but with `callback=PredictCallBack`.
* `model.predict()` should accept D2 layer or the dataloaders.


Other important features:
* We should add some inbuilt util function that provides the final dataframe (or csv file, if predictions are very large to be saved in a memory) from the prediction tensors (if mode="prediction").
* predict should return a `dict` of tensors (or a D1 layer?).
* We should also add some utils to plot predictions and the actual values

### Code snippets

* ***PredictCallBack (similar to the one in v1)***


```python
from lightning.pytorch.callbacks import BasePredictionWriter
class PredictCallback(BasePredictionWriter):
    """Internally used callback to capture predictions and optionally write them to disk.
    
    This callback is used internally by a model's `.predict()` method. It
    captures the raw output from the model, processes it according to the
    specified `mode` (e.g., converting to point predictions or quantiles),
    and collects any other requested information like input data (`x`),
    actuals (`y`), and indices.

    The final, collated results can be accessed through the `.result` property,
    which returns a dictionary of tensors(or should it be the D1 layer?).

    Parameters
    ----------
    mode : str
        The prediction mode. Determines how the raw model output is processed.
        Should be one of "prediction", "quantiles", or "raw".
    return_index : bool
        If `True`, returns the group and time indices for each prediction.
    return_decoder_lengths : bool
        If `True`, returns the lengths of the decoder sequence for each prediction.
    return_y : bool
        If `True`, returns the actual target values (`y`) corresponding to
        the predictions.
    return_x : bool
        If `True`, returns the full input dictionary (`x`) for each prediction.
    write_interval : str, default="batch"
        When to perform the collection step.
    output_dir : Optional[str]
        Path to a directory where predictions can be saved. If provided,
        predictions will be written to files in this directory.

    Attributes
    ----------
    result :  Dict[str, Any]
        The final, collated prediction result tensors. 

    Notes
    -----
    This callback is designed to be instantiated and used within a model's
    `.predict()` method, which handles passing the necessary configuration.
    """  

    def __init__(
        self,
        mode, # prediction mode
        return_index, # return time_idx, groups           
        return_decoder_lengths, # return decoder_lengths    
        return_y, # return actual y                         
        return_x, # return x                              
        write_interval,  # when to write to the disk?
        output_dir,
        **anyother_param_and_kwargs
    ):
        super().__init__(write_interval=write_interval)
        # initialise params
        self.mode = mode
        self.return_x = return_x
        self.return_index = return_index
        self.return_decoder_lengths = return_decoder_lengths
        self.return_y = return_y
        self.output_dir = output_dir

        # any other initialisatioins
    
    def _reset_data(self):
        self._output = []
        self._decoder_lengths = []
        self._x= []
        self._index = []
        self._y = []
        self._result = []
        
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # override the func from base class to write output based on modes
        if mode=="prediction":
            pl_module.to_prediction()
        if mode=="quantiles":
            pl_module.to_quantiles()
            
           
        if self.return_x:
            self._x = x
            # save x
        if self.return_index:
            self._index = index
            # save index
        if self.return_decoder_lengths:
            self._decoder_lengths = decoder_lengths
            # save decoder length
        if self.return_y:
            self._y = y
            # save y
            
        # other logic
        
        
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        # write the data
        
        self._reset_data()
    
    # anyother function if required
 
            
    @property
    def result:
        if self.mode=="prediction":
            result = self.get_final_preds(self.return_type)
        else:
            result = self._result
        return result
```
* ***`model.predict()`***

```python
# Inside the BaseModel

def predict(
    data, # Dataloader, D2 layer 
    mode, # prediction mode
    return_index, # return time_idx, groups
    return_decoder_lengths, # return decoder_lengths
    return_y, # return actual y
    write_interval,  # when to write to the disk?
    return_x, # return x
    output_dir,
    trainer_kwargs, # kwargs for trainer like enable_progress_bar etc
    **anyother_param_and_kwargs
):
    """
     Generate predictions for new data.

    This method provides a high-level interface for making forecasts. It can
    accept either a `DataModule` or a dataloader.

    Parameters
    ----------
    data : Union[DataModule, DataLoader]
        The data to predict on. Can be one of:
        - `DataModule` (D2 Layer): A pre-configured
          data module.
        - `DataLoader`: A pre-built DataLoader for prediction.
    mode : str
        The prediction mode. One of "prediction", "quantiles", or "raw".
        - "prediction": Returns final predictions.
        - "quantiles": Returns a forecast for each quantile defined in the
          model's loss function.
        - "raw": Returns the raw, unprocessed output of the network.
    return_index : bool
        If `True`, include the time and group index in the output. 
    return_decoder_lengths : bool, default=False
        If `True`, include the decoder lengths in the output dictionary
    return_y : bool
        If `True`, include the actual target values (`y`) corresponding to
        the predictions. Requires `y` to be present in the prediction data.
    return_x : bool
        If `True`, include the full input dictionary (`x`) for each prediction
    output_dir : Optional[str]
        Path to a directory where predictions can be saved. 
    **kwargs
        Additional keyword arguments passed to the model's processing methods,
        such as `to_prediction()` or `to_quantiles()`. For example, you can
        override the default quantiles by passing `quantiles=[0.1, 0.5, 0.9]`.


    Returns
    -------
    Dict[str, Any]
        The final, collated prediction result tensors. 
    """
    if isinstance(data, d2):
        # create dataloaders
    
    predict_callback=PredictCallback(
            mode=mode,
            return_index=return_index,
            return_decoder_lengths=return_decoder_lengths,
            write_interval=write_interval,
            return_x=return_x,
            output_dir=output_dir,
            return_y=return_y,
            **kwargs
    )
    trainer_kwargs.setdefault(
            "callbacks", trainer_kwargs.get("callbacks", []) + [predict_callback]
        )
    
    trainer = Trainer(fast_dev_run=fast_dev_run, **trainer_kwargs)
    trainer.predict(self, dataloader)
    # logging logic
    
    return predict_callback.result
```
* ***`model.to_prediction()` & `model.to_quantiles`***

```python
# Inside the BaseModel (similar to v1)

def to_prediction(self, out: dict[str, Any], **kwargs):

        if isinstance(self.loss, MultiLoss):
                out = [
                    Metric.to_prediction(loss, out["prediction"][idx])
                    for idx, loss in enumerate(self.loss)
                ]
        else:
                out = Metric.to_prediction(self.loss, out["prediction"])
                
        return out

def to_quantiles(self, out: dict[str, Any], **kwargs):

        if isinstance(self.loss, MultiLoss):
                out = [
                    Metric.to_quantiles(
                        loss,
                        out["prediction"][idx],
                        quantiles=kwargs.get("quantiles", loss.quantiles),
                    )
                    for idx, loss in enumerate(self.loss)
                ]
        else:
                out = Metric.to_quantiles(
                    self.loss,
                    out["prediction"],
                    quantiles=kwargs.get("quantiles", self.loss.quantiles),
                )
                
        return out
```
* ***Another util to transform the tensors to a dataframe***
```python
    def to_dataframe(predictions, D2layer):
        # transform the prediction tensors to dataframe
```
> **NOTE: We still have to try and see if we could fit it in `PredictCallBack` without requiring the metadata from D2 layer (maybe just `return_index` and other params if made `True` are sufficient).**

### open questions:
<!-- * I think `return_type` should only work with `prediction` mode, for other modes, only returning tensors should suffice?  -->
* Should the `return` be `tensors` or the D1 layer? which is more intitutive and more useful for the user?






