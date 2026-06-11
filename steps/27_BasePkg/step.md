# How to handle `load` and `save` in `BasePkg`

In `pytorch-forecasting` v2, the model checkpoints are saved along with data module config, model configs and other metadata. But there is no way to save and load the scalers. This EP proposes a way to do so while still providing the flexibility to the user to decide if they want to save the scalers or not.
But first let's take a look at the current state of load and saving mechanism in v2. It just saves the metadata, cfgs and model checkpoints. 

## Current state

Currently, in `pytorch-forecasting` v2, we use `fit` to save the checkpoints and the model is loaded when we initialise the new class.

Example of how this is done in v2 currently is given below, but before this, lets assume few things:
- All the classes that we use are already imported from their respective modules
- `dataset`, `model_pkg`, `model_cfg`, `datamodule_cfg` and `trainer_cfg` are defined as follows:
    ```python
    dataset = TimeSeries(
        data=data_df, # data_df is any arbitrary dataframe
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature", "static_feature"],
        cat=["category", "static_feature_cat"],
        known=["future_known_feature"],
        unknown=["x", "category"],
        static=["static_feature", "static_feature_cat"],
    )

    # data module config
    datamodule_cfg = dict(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    # model config
    model_cfg = dict(
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={"mode": "min", "factor": 0.1, "patience": 10},
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
    )

    # trainer config
    trainer_cfg = dict(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # model package
    model_pkg = TFT_pkg_v2(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        datamodule_cfg=datamodule_cfg,
    )
    ```
Now, assuming we are using the above definitions for the next two examples.

##### Example 1: Saving

If we pass `save_ckpt` as `True` in `model_pkg.fit()`, the methods automatically saves the model checkpoints after fitting. `ckpt_dir` is optional and defaults to `"checkpoints"`. `ckpt_kwargs` are also optional arguments passed to `ModelCheckpoint`.

This saves - model checkpoints and all the three cfgs (`model_cfg`, `datamodule_cfg` and `trainer_cfg`) and the metadata from the data module
Currently, these checkpoints and cfgs are saved in this format
```
ckpt_dir/
├── best-epoch=X-step=Y.ckpt
└── model_cfg.pkl
└── datamodule_cfg.pkl
└── trainer_cfg.pkl (there is a bug in _save_artifacts where it doesnt save this cfg - but expected behaviour was it should've saved it)
└── metadata.pkl

```

```python
ckpt_dir = "checkpoints"
best_model_path = model_pkg.fit(
    test_data["train"],
    save_ckpt=True,
    ckpt_dir=ckpt_dir,
    ckpt_kwargs={"monitor": "train_loss_epoch"},
)
```

##### Example 2: Loading

To load the model, we just pass a `ckpt_path` (no need to pass any cfgs in this case as all the cfgs are taken from the saved cfgs), But if you pass the cfgs, these configs will override the saved ones.

But that doesnt mean that a new model would be created from these cfgs, if there is a saved artifact is present, it would be given preference. But other objects like `datamodule` which is always created from cfgs and never stored as a file, would use the new passed cfgs instead. Same for `trainer`.


```python
pkg_loaded = model_pkg(ckpt_path=best_model_path)
predictions = pkg_loaded.predict(test_data["predict"], mode="prediction")
```

### Backend

##### Saving

The saving logic completely sits inside the `.fit()` method of the `Base_pkg`. When we pass `save_ckpt=True` to `fit()`,
the method saved the model checkpoints using `ModelCheckpoint`, while other artifacts (`model_cfg`, `datamodule_cfg` and `metadata` of `datamodule`) are saved by calling `_save_artifacts`.

```python
# inside BasePkg

def _save_artifact(self, output_dir: Path):
        """Save all configuration artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "datamodule_cfg.pkl", "wb") as f:
            pickle.dump(self.datamodule_cfg, f)

        with open(output_dir / "model_cfg.pkl", "wb") as f:
            pickle.dump(self.model_cfg, f)

        if self.datamodule is not None and hasattr(self.datamodule, "metadata"):
            with open(output_dir / "metadata.pkl", "wb") as f:
                pickle.dump(self.datamodule.metadata, f)

def fit(
        self,
        data: TimeSeries | LightningDataModule,
        save_ckpt: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        ckpt_kwargs: dict[str, Any] | None = None,
        **trainer_fit_kwargs,
    ):
        """
        Fit the model to the training data.

        Parameters
        ----------
        data : Union[TimeSeries, LightningDataModule]
            The data to fit on (D1 or D2 layer). This object is responsible
            for providing both training and validation data.
        save_ckpt : bool, default=True
            If True, save the best model checkpoint and the `datamodule_cfg`.
        ckpt_dir : Union[str, Path], default="checkpoints"
            Directory to save artifacts.
        ckpt_kwargs : dict, optional
            Keyword arguments passed to ``ModelCheckpoint``.
        **trainer_fit_kwargs :
            Additional keyword arguments passed to `trainer.fit()`.

        Returns
        -------
        Optional[Path]
            The path to the best model checkpoint if `save_ckpt=True`, else None.
        """
        if isinstance(data, TimeSeries):
            self.datamodule = self._build_datamodule(data)
        else:
            self.datamodule = data
        self.datamodule.setup(stage="fit")

        if self.model is None:
            if not self.model_cfg:
                raise RuntimeError(
                    "`model_cfg` must be provided to train from scratch."
                )
            metadata = self.datamodule.metadata
            self._build_model(metadata)

        callbacks = self.trainer_cfg.get("callbacks", []).copy()
        checkpoint_cb = None
        if save_ckpt:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            default_ckpt_kwargs = {
                "dirpath": ckpt_dir,
                "filename": "best-{epoch}-{step}",
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
            }
            if ckpt_kwargs:
                default_ckpt_kwargs.update(ckpt_kwargs)
            checkpoint_cb = ModelCheckpoint(**default_ckpt_kwargs)
            callbacks.append(checkpoint_cb)
        trainer_init_cfg = self.trainer_cfg.copy()
        trainer_init_cfg.pop("callbacks", None)

        self.trainer = Trainer(**trainer_init_cfg, callbacks=callbacks)

        self.trainer.fit(self.model, datamodule=self.datamodule, **trainer_fit_kwargs)
        if save_ckpt and checkpoint_cb:
            best_model_path = Path(checkpoint_cb.best_model_path)
            self._save_artifact(best_model_path.parent)
            print(f"Artifacts saved in: {best_model_path.parent}")
            return best_model_path
        return None
```
**Note that the `_save_artifacts` methods doesnt save `trainer_cfg` which is a bug i think, we should save it (or always ask for this cfg from the user - both have their own pros and cons)**

##### Loading

Loading happens inside `__init__()`, which calls `_load_config()` to load the saved configs and `_build_model` to build the model from the checkpoint.

```python
# inside BasePkg

def __init__(
        self,
        model_cfg: dict[str, Any] | str | Path | None = None,
        trainer_cfg: dict[str, Any] | str | Path | None = None,
        datamodule_cfg: dict[str, Any] | str | Path | None = None,
        ckpt_path: str | Path | None = None,
    ):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.model_cfg = self._load_config(
            model_cfg, ckpt_path=self.ckpt_path, auto_file_name="model_cfg.pkl"
        )

        self.datamodule_cfg = self._load_config(
            datamodule_cfg,
            ckpt_path=self.ckpt_path,
            auto_file_name="datamodule_cfg.pkl",
        )
        self.trainer_cfg = self._load_config(trainer_cfg)
        self.metadata = self._load_config(
            None, ckpt_path=self.ckpt_path, auto_file_name="metadata.pkl"
        )

        self.model = None
        self.trainer = None
        self.datamodule = None
        if self.ckpt_path:
            self._build_model(metadata=self.metadata, **self.model_cfg)
        else:
            self.model = None
            
@staticmethod
def _load_config(
        config: dict | str | Path | None,
        ckpt_path: str | Path | None = None,
        auto_file_name: str | None = None,
    ) -> dict:
        """
        Loads configuration from a dictionary, YAML file, or Pickle file.
        """
        if config is None:
            if ckpt_path and auto_file_name:
                path = Path(ckpt_path).parent / auto_file_name
                if path.exists():
                    with open(path, "rb") as f:
                        return pickle.load(f)  # noqa : S301
            return {}

        if isinstance(config, dict):
            return config

        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        print(suffix)

        if suffix in [".yaml", ".yml"]:
            with open(path) as f:
                return yaml.safe_load(f) or {}

        elif suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. Use .yaml, .yml, or .pkl"
            )
            
def _build_model(self, metadata: dict, **kwargs):
        """Instantiates the model, either from a checkpoint or from config."""
        model_cls = self.get_cls()
        if self.ckpt_path:
            self.model = model_cls.load_from_checkpoint(
                self.ckpt_path, metadata=metadata, **kwargs
            )
        elif self.model_cfg:
            self.model = model_cls(**self.model_cfg, metadata=metadata)
        else:
            self.model = None

```

### Issues with current design

- There is no clear `load` and `save` methods - this makes adding ways to save new artifacts (like `scalers`) hard as we dont have a specific place where we can keep this logic.
- Everything is intermingled - the same method `_build_model` builds a mpdel from a config and from the checkpoints. There is no clear distinction between the responsibilities of the methods
- Everything is saved in the same directory and there is no clear distinction between different artifacts. Optimally, in the parent directory (lets say `ckpt_dir`) should have separate sub-directories for model-checkpoints, metadata, configs and so on.
   Meaning - the directory shuould be like this:
   ```
   ckpt_dir/
   ├── checkpoints/
         └── best-epoch=X-step=Y.ckpt
   └── configs/
         └── model_cfg.pkl
         └── datamodule_cfg.pkl
         └── trainer_cfg.pkl
   └── metadata/
         └── datamodule_metadata.pkl
   └── scalers/
         └── scalers.pkl
         └── target_normalizer.pkl
   ```
   But currently, there are no such sub-folders (see `Example 1: Saving` section for more info).

## Proposed Design
After discussion with Franz:

The `BasePkg` should stay generic and would act like a coordinator, just passing out the commands it recieved from the user to the respective layer.

Meaning, for `load` and `save`:
- The user calls `save()` either by passing `ckpt_dir` in `fit` or by specifically calling `pkg.save`.
  1. The `pkg.save` calls internally `datamodule.save` and `model.save`.
     - Here, we would also have an arg called `exclude` (present in `fit` and `pkg.save`) which expects a `list` of strings and the user can decide what to "exclude" while saving (like excluding scalers and saving only the model weights). By default, it would be empty
       - But we would always save the `model_cfg` and `data_module_cfg` and user cant pass these to `exclude`.
  2. If the datamodule has scalers (or any other artifact) to save, it would save it and return the path of the saved artifact else, return `None`. The model saves the model weights using `ModelCheckpoint`.
  3. After collecting the paths where all the artifacts have been saved from the D2 and M layers, it would create a `artifacts.json` that would save the aritfacts and the place they are saved
     - This would save us from writing a long `if-else` block and the `load` would simply read through this json and load everything. So, in case user has decided to exclude anything or D2 layer didnt have anything to save (eg, `scalers` were not initializer), `load` would not face any issues.
     - Obviously, if D2 returns `None` meaning it had nothing to `save` we would raise warnings.
  4. The `pkg` class just performs the reconciliation and make sure if everything is saved in correct places or not.
- The user loads by specifically calling `pkg.load()` - the only endpoint for loading. The user just has to pass the `ckpt_dir` or path to `artifacts.json`.
  1. The `pkg.load()` reads `artifacts.json` to see what to load and from where to `load`. It would load everything present in the `artifacts.json`.
     - The `artifacts.json` can be created by the user themselves as well, providing them flexibility to load the artifacts from some totally different places and not necessarily from the same directory.
     - As we assume users create this json themselves, we would assume it is safe to load the artifacts from the paths specified. `pkg.load` would just do the reconciliation and see if the loaded artifacts have been loaded correctly or not.
  2. The `pkg.load()` calls internally `datamodule.load()` (if scalers or any other artifact that data module saves is present in the `artifacts.json`) and `model.load()`.
     - If the datamodule has anything to load (like scalers, metadata etc), it will load it otherwise return `None`. Similarly, the model loads the model weights using `model._load_from_checkpoint()`.
     - The `pkg` class just performs the check if everything is loaded correctly and if there is no error or issue.

As the cfgs(`model_cfg`, `trainer_cfg` and `datamodule_cfg`) are passed to `BasePkg`, it should save these cfgs and then call the respective layers (D2 and M) to save (or load) their artifacts.

### Vignettes


#### Pseudocode

**`BasePkg` root**

Here we would just save the artifacts that are used by all kind of models - model checkpoints, model cfgs, data module cfgs and trainer cfgs.
```python
# inside BasePkg

def __init__(
    self,
        model_cfg: dict[str, Any] | str | Path | None = None,
        trainer_cfg: dict[str, Any] | str | Path | None = None,
        datamodule_cfg: dict[str, Any] | str | Path | None = None,
        ckpt_path: str | Path | None = None,
):
  """``__init__()`` of ``BasePkg``
  
  The user dont have to call ``load`` separately to laod the model and its artifact, just passing the 
  ``ckpt_path`` during initialization would be enough.
  The __init__() would just call ``load`` here if we have ``ckpt_path`` and it will load the model 
  and other artifacts.
  """
    self.model_cfg = model_cfg
    self.trainer_cfg = trainer_cfg
    self.datamodule_cfg = datamodule_cfg
    self.ckpt_path = ckpt_path
    
    if ckpt_path is not None:
        self.load(self.ckpt_path)
        

@staticmethod
def load(ckpt_path):
    """load the model and its artifact.
    
    It would use ``._load_from_checkpoint()`` of ``lightning`` for model ckpts.
    The cfgs would be loaded  using ``_load_configs()`` of the current implementation works (see above) - it would 
    load cfgs from the ``pkl``, ``yaml`` files. If the user passes a cfg as a ``dict``, it would be given
    higher preference than the already saved cfgs files in the checkpoints or the ``yaml`` file passed.
    Parameters
    ----------
    ckpt_path: str, Path
        Path where the checkpoints (like model ckpts, cfgs etc) are stored.
    """
    # load the models and other artifacts here
    
@staticmethod
def save(ckpt_path, ckpt_kwargs):
    """load the model and its artifact.
    
    The method would use ``ModelCheckpoint`` for saving model ckpts inside ``ckpt_path/model_checkpoints`` folder. The cfgs would be saved as ``pkl`` files in 
    ``ckpt_path/configs`` folder. ``metadata`` of datamodule is saved as ``pkl`` file in ``ckpt_path/metadata`` folder.
    
    Complete folder structure is like this:
    
    ckpt_path/
   ├── checkpoints/
         └── best-epoch=X-step=Y.ckpt
   └── configs/
         └── model_cfg.pkl
         └── datamodule_cfg.pkl
         └── trainer_cfg.pkl
   └── metadata/
         └── datamodule_metadata.pkl
         
    Parameters
    ----------
    ckpt_path: str, Path
        Path where the checkpoints (like model ckpts, cfgs etc) are to be stored.
    ckpt_kwargs : dict, optional
        Keyword arguments passed to ``ModelCheckpoint``.
    
    Returns
    -------
    Path
        The path to the model checkpoint
    """
    # save the model and other artifacts here
    
def fit(
        self,
        data: TimeSeries | LightningDataModule,
        save_ckpt: bool = True,
        ckpt_dir: str | Path = "checkpoints",
        ckpt_kwargs: dict[str, Any] | None = None,
        **trainer_fit_kwargs
):
    """fit the model and save the checkpoints if needed.
  
    Parameters
    ----------
    data : Union[TimeSeries, LightningDataModule]
        The data to fit on (D1 or D2 layer). This object is responsible
        for providing both training and validation data.
    save_ckpt : bool, default=True
        If True, save the best model checkpoint and other artifacts like cfgs etc.
    ckpt_dir : Union[str, Path], default="checkpoints"
        Directory to save artifacts.
    ckpt_kwargs : dict, optional
        Keyword arguments passed to ``ModelCheckpoint``.
    **trainer_fit_kwargs :
        Additional keyword arguments passed to `trainer.fit()`.

    Returns
    -------
    Optional[Path]
        The path to the BEST model checkpoint if `save_ckpt=True`, else None.
 
    """
    # fit the model 
    # after fitting the model:
    if save_ckpt:
        self.save(ckpt_dir, ckpt_kwargs)
        
```
