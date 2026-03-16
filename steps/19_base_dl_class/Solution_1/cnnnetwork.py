from basedeepnetwork import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNNetwork(BaseDeepNetwork):
    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        activation="sigmoid",
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = [6, 12]
        self.activation = activation

    def build_network(self, input_shape, **kwargs):
        # not sure of the whole padding thing
        from tensorflow import keras

        padding = "valid"
        input_layer = keras.layers.Input(input_shape)
        # sort this out, why hard coded to 60?
        if input_shape[0] < 60:
            padding = "same"

        # this does what?
        if len(self.filter_sizes) > self.n_conv_layers:
            self.filter_sizes = self.filter_sizes[: self.n_conv_layers]
        elif len(self.filter_sizes) < self.n_conv_layers:
            self.filter_sizes = self.filter_sizes + [self.filter_sizes[-1]] * (
                self.n_conv_layers - len(self.filter_sizes)
            )
        conv = keras.layers.Conv1D(
            filters=self.filter_sizes[0],
            kernel_size=self.kernel_size,
            padding=padding,
            activation=self.activation,
        )(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        for i in range(1, self.n_conv_layers):
            conv = keras.layers.Conv1D(
                filters=self.filter_sizes[i],
                kernel_size=self.kernel_size,
                padding=padding,
                activation=self.activation,
            )(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
