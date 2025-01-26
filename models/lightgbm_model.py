from models.model_template import BaseModel
from lightgbm import LGBMClassifier
import pickle as pk
from tqdm import tqdm

class LGBM(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, min_child_samples=5, min_split_gain=0.001):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
        )

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