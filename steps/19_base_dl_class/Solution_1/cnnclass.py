
from sklearn.utils import check_random_state
from cnnnetwork import CNNNetwork
from basedeepclass import BaseDeepClassifier
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNClassifier(BaseDeepClassifier, CNNNetwork):

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
        random_state=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(CNNClassifier, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.avg_pool_size = avg_pool_size
        self.kernel_size = kernel_size
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None

    def _fit(self, X, y):
        if self.callbacks is None:
            self._callbacks = []

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )
        return self

if __name__ == "__main__":
    cnn = CNNClassifier()
    from sktime.datasets import load_unit_test
    X_train, y_train = load_unit_test(split='train', return_X_y=True)
    cnn.fit(X_train, y_train)
    X_test, y_test = load_unit_test(split='test', return_X_y=True)
    print(cnn.predict(X_test))
    print(y_test)
