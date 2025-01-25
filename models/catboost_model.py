from models.model_template import BaseModel
from catboost import CatBoostClassifier
import pickle as pk

class CatBoosst(BaseModel):
    def __init__(self):
        self.cb = CatBoostClassifier(task_type="GPU")

    def fit(self, x_train, y_train):
        self.cb.fit(x_train, y_train)

    def predict(self, x):
        return self.cb.predict(x)

    def predict_proba(self, x):
        return self.cb.predict_proba(x)

    def save(self, file_path):
        self._mkdir()
        with open(file_path, 'wb') as f:
            pk.dump(self.cb, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.cb = pk.load(f)