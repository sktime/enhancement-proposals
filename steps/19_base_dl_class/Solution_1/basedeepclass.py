import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier


class BaseDeepClassifier(BaseClassifier):

    def __init__(self, batch_size=40, random_state=None):
        super(BaseDeepClassifier, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def summary(self):
        return self.history.history

    def _predict(self, X, **kwargs):
        probs = self._predict_proba(X, **kwargs)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _predict_proba(self, X, **kwargs):
        # Transpose to work correctly with keras
        X = X.transpose((0, 2, 1))
        probs = self.model_.predict(X, self.batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def convert_y_to_keras(self, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)
        self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        # categories='auto' to get rid of FutureWarning
        y = self.onehot_encoder.fit_transform(y)
        return y
