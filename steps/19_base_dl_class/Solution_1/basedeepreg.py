import numpy as np

from sktime.regression.base import BaseRegressor

class BaseDeepRegressor(BaseRegressor, ABC):

    def __init__(self, batch_size=40):
        super(BaseDeepRegressor, self).__init__()

        self.batch_size = batch_size
        self.model_ = None

    def _predict(self, X, **kwargs):
        """
        Find regression estimate for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.

        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        X = X.transpose((0, 2, 1))
        y_pred = self.model_.predict(X, self.batch_size, **kwargs)
        y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
