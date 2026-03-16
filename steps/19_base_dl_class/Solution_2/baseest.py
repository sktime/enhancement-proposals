from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class BaseDeepEstimator(BaseEstimator):

    def __init__(self, batch_size=40, random_state=None):

        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def summary(self):
        return self.history.history

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
