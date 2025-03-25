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

    * ``t``: np.array of shape (n_timepoints), of type ``np.int64``, ``np.float64``, or ``np.datetime64``.
      Time index for each time point in the past or present. Aligned with ``y``,
      and ``x``.
    * ``y``: tensor of shape (n_timepoints, n_targets)
      Target values for each time point. Rows are time points, aligned with ``t``.
    * ``x``: tensor of shape (n_timepoints, n_features)
      Features for each time point. Rows are time points, aligned with ``t``.
    * ``group``: tensor of shape (n_groups)
      Group identifiers for time series instances.
    * ``st``: tensor of shape (n_static_features)
      Static features.

    Optionally, the following str-keyed entries can be included:
    * ``w``: tensor of shape (n_timepoints)
      Weights, aligned with ``t``. If not provided, assumed to be equal weights.
    * ``cutoff``: 0-dimensional ``np.int64``, ``np.float64``, or ``np.datetime64``, same as type of ``t``.
    If not provided, assumed to be latest time point in the data set.
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


##### Speculative user vignette


```python
import pytorch_lightning as pl


class BasePtfNetwork(pl.LightningModule):
    @abstractmethod
    def forward(self, batch:dict)-> torch.tensor: 
      ##THE FORWARD LOOP MUST BE DEFINE FOR EACH MODEL

    def predict(self, batch:dict)->torch.tensor:
        return self(batch) #USUALLY inference = forward
        
    def configure_optimizers(self):
        ##Lightning stuff


    def training_step(self, batch, batch_idx):
        ## Lightning does a lot of stuff automatically, (backward, etc) BUT sometimes you need to set  self.automatic_optimization = False and do manually the backward (EXAMPLE SAM) 
        if self.has_sam_optim:
            
            opt = self.optimizers()
            def closure():
                opt.zero_grad()
                y_hat = self(batch)
                loss = self.compute_loss(batch,y_hat)
                self.manual_backward(loss)
                return loss

            opt.step(closure)
            y_hat = self(batch)
            loss = self.compute_loss(batch,y_hat)
      
        else:
            y_hat = self(batch)
            loss = self.compute_loss(batch,y_hat) ##DO NOT BACKWARD, pl will do for you
        return loss

    
    def validation_step(self, batch, batch_idx):
        ##return the loss  self.compute_loss(batch,y_hat)
        ## but you can logg something for example (IN DSIPTS WE USE AIM)
        y_hat = self(batch)
        if batch_idx==0:
            if self.use_quantiles:
                idx = 1
            else:
                idx = 0
            #track the predictions! We can do better than this but maybe it is better to firstly update pytorch-lightening 
            if self.count_epoch%int(max(self.trainer.max_epochs/100,1))==1:

                for i in range(batch['y'].shape[2]):
                    real =  batch['y'][0,:,i].cpu().detach().numpy()
                    pred =  y_hat[0,:,i,idx].cpu().detach().numpy()
                    fig, ax = plt.subplots(figsize=(7,5))  
                    ax.plot(real,'o-',label='real')
                    ax.plot(pred,'o-',label='pred')
                    ax.legend()
                    ax.set_title(f'Channel {i} first element first batch validation {int(100*self.count_epoch/self.trainer.max_epochs)}%')
                    self.logger.experiment.track(Image(fig), name='cm_training_end')
                    #self.log(f"example_{i}", np.stack([real, pred]).T,sync_dist=True)

        return self.compute_loss(batch,y_hat)


    def validation_epoch_end(self, outs):
        ##loggin stuff
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss.item(),sync_dist=True)
        logging.info(f'Epoch: {self.count_epoch} train error: {self.train_loss_epoch:.4f} validation loss: {loss.item():.4f}')

    def training_epoch_end(self, outs):
        #log again
        loss = sum(outs['loss'] for outs in outs) / len(outs)
        self.log("train_loss", loss.item(),sync_dist=True)
        self.count_epoch+=1    
        self.train_loss_epoch = loss.item()

    def compute_loss(self,batch,y_hat):
      ## IN DSIPTS we have a dozen of different losses

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
      
    ##HERE USUALLY THERE ARE THE LAYERS DEFINITION

    def ref_network(self):  # pointer to network, could be more complicated mmh this is a CS stuff, not able to answer which is the best solution
        from somewhere import MyTorchNetwork

        return MyTorchNetwork

    def ref_dataloader(self):  # pointer to dataloader  --> I don't like a to put here the dataloader. If we use PL we need to pass it to the trainer. We can also decide to use pure pytroch but we will miss some good features of PL
        from somewhere import D2LoaderForMyTorchnetwork

        return D2LoaderForMyTorchnetwork

    @classmethod
    def from_dataset(cls, dataset):  # sets parameters from dataset  --> see before
        return cls(**get_params_from(dataset))

    # Since the most of the logic is in the base class we need only to write the forward loop

    def forward(self, batch):##--> batch are suppose to be standardize!

      return tensor_output ## also the output is  standard (in DSIPTS Bs x Len x Channels x MUL --> MUL can be 1 or 3 depending if we use quantile loss

    ##SOME MODELS NEEDS TO OVERWRITE SOME BASIC METHODS
    def predict(self, batch): ##usually predict is the same as forward!
      ##maybe some different respect to forward (generative models?)

    def should_we_forward_lignthing_methods(self, **kwargs):  # ? --> NOT NEEDED

    def train(self, dataset): #--> NOT NEEDED, we use keywords by PL
        # logic related to training


```


#### usage vignette
SEE the class before (JUST AN IDEA)
should maybe center more around the data loader

#### train
```python



data_loader = my_class(configs).get_dataloader(more_configs)
# need training and validation data loader separately
data_loader_validation = my_class(configs).get_dataloader(more_configs)

##this is a good callback for saving the weights
checkpoint_callback = ModelCheckpoint(dirpath=dirpath, ##where to save stuff
                              monitor='val_loss', ##usually monitoring validation loss is a good idea (train may overfit)
                              save_last = True, #save last checkpoint
                              every_n_epochs =1,
                              verbose = verbose,
                              save_top_k = 1, #save only the best!
                              filename='checkpoint')



trainer = pl.Trainer(default_root_dir=dirpath, ##where to save stuff
                             logger = aim_logger, ## logger to use
                             callbacks=[checkpoint_callback,mc], ## callbacks to call
                             auto_lr_find=auto_lr_find, ## this can be useful
                             other_parameters #(gpu, worksers, accelerators for computations)
                  )

if auto_lr_find:
    trainer.tune(model,train_dataloaders=train_dl,val_dataloaders = valid_dl)
    files = os.listdir(dirpath)
    for f in files:
        if '.lr_find' in f: ##PL saves some tmp file
            os.remove(os.path.join(dirpath,f))

##I think we can load HERE some pretrained weights and do only finetuning
model = model.load_from_checkpoint(pretrained_path)

trainer.fit(model, data_loader,data_loader_validation) ##just what you need for the training part

self.checkpoint_file_best = checkpoint_callback.best_model_path #save where the checkpoints are
self.checkpoint_file_last = checkpoint_callback.last_model_path #save where the checkpoints are

##here we may need to save other metadata of the model: scalers etc
```
##### inference

```python
#####load the model
model = MyNetwork(**config_to_use)
if load_last:
  weight_path = os.path.join(self.checkpoint_file_last) 
else:
  weight_path = os.path.join(self.checkpoint_file_best) 
model = model.load_from_checkpoint(weight_path)
```

```python
#####use test set or new data
data_loader_test = my_class(configs).get_dataloader(more_configs_no_shuffle_no_drop) ##--> same as before, in the batch there is the time! 
res = []
real = []
for batch in data_loader_test :
  res.append(model.inference(batch).cpu().detach().numpy())
  real.append(batch['y'].cpu().detach().numpy()) ##if in the test set, otherwise we don't have this key!
  res = np.vstack(res)
  real = np.vstack(real)
  time = dl.dataset.t #time is in the dataloader!
  groups = dl.dataset.groups  #group are in the dataloader!

#here really depends on what you want to produce (pandas dataframe, polars, etc) and depends on the organization of the batches. In DSIPTS we have a ugly-to-see-but-working procedure that provides the output
```





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
