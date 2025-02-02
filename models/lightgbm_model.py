from models.model_template import BaseModel
from lightgbm import LGBMClassifier, log_evaluation, record_evaluation
import pickle as pk

class LGBM(BaseModel):
    def __init__(self, n_estimators=1000, learning_rate=0.1, max_depth=-1, min_child_samples=5, min_split_gain=0.001):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
        )
        self.evals_result_ = {}  # Dictionary to store loss history

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """ Train the model and store training loss history """

        eval_set = [(x_train, y_train)]
        eval_names = ["Train"]

        if x_val is not None and y_val is not None:
            eval_set.append((x_val, y_val))
            eval_names.append("Validation")

        # Train the model with correct evaluation storage
        self.model.fit(
            x_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric="multi_logloss",  # Multi-class log loss for CIFAR-10
            callbacks=[
                # Logs training every 10 iterations
                log_evaluation(10),
                # Store eval values
                record_evaluation(self.evals_result_)
            ]
        )

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
