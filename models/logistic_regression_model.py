from models.model_template import BaseModel
from sklearn.linear_model import LogisticRegression
import pickle as pk


class LogisticRegressionModel(BaseModel):
    def __init__(self, C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto'):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, multi_class=multi_class)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save(self, file_path):
        self._mkdir()
        with open(file_path, 'wb') as f:
            pk.dump(self.model, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pk.load(f)