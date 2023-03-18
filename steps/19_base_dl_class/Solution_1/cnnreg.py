from cnnnetwork import CNNNetwork
from basedeepreg import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNRegressor(BaseDeepRegressor, CNNRegressor):

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        random_seed=0,
    ):
        _check_dl_dependencies(severity="error")
        super(CNNRegressor, self).__init__(
            batch_size=batch_size,
        )
        self.n_conv_layers = n_conv_layers
        self.avg_pool_size = avg_pool_size
        self.kernel_size = kernel_size
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_seed = random_seed


    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        if self.callbacks is None:
            self._callbacks = []

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model.summary()

        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )
        return self
