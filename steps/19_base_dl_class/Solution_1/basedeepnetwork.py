from abc import ABC, abstractmethod
from sktime.base import BaseObject

class BaseDeepNetwork(BaseObject, ABC):

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        ...

    def build_model(self, input_shape, n_classes, **kwargs):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )
        return model
