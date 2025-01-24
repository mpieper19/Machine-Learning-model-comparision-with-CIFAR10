from models.model_template import BaseModel
from lightgbm import LGBMClassifier
import pickle as pk
from tqdm import tqdm

class LGBM(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, min_child_samples=5, min_split_gain=0.001):
        self.lgm = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
        )

    def fit(self, x_train, y_train):
        """
        Fits the Gradient Boosting model with a progress bar.
        """
        # Initialize progress bar
        with tqdm(total=self.lgm.n_estimators, desc="Training Progress") as pbar:
            for i in range(self.lgm.n_estimators):
                # Incrementally fit each stage
                self.lgm.set_params(n_estimators=i + 1)  # Update the number of trees
                self.lgm.fit(x_train, y_train)
                pbar.update(1)  # Update the progress bar

    def predict(self, x):
        return self.lgm.predict(x)

    def predict_proba(self, x):
        return self.lgm.predict_proba(x)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pk.dump(self.lgm, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.lgm = pk.load(f)