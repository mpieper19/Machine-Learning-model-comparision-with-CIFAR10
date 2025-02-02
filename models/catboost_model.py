from models.model_template import BaseModel
from catboost import CatBoostClassifier
import pickle as pk

class CatBoosst(BaseModel):
    def __init__(self):
        self.model = CatBoostClassifier(task_type="GPU",
                                        iterations=1000,
                                        verbose=10,
                                        eval_metric='MultiClass',
                                        loss_function='MultiClass',
                                        learning_rate=0.1)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

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

    def get_evals_result(self):
        return self.model.get_evals_result()
