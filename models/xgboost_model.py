from models.model_template import BaseModel
from xgboost import XGBClassifier
import pickle as pk
from tqdm import tqdm

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.xgb = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Parallelize computation
        )

    def fit(self, x_train, y_train):
        """
        Fits the Gradient Boosting model with a progress bar.
        """
        # Initialize progress bar
        with tqdm(total=self.xgb.n_estimators, desc="Training Progress") as pbar:
            for i in range(self.xgb.n_estimators):
                # Incrementally fit each stage
                self.xgb.set_params(n_estimators=i + 1)  # Update the number of trees
                self.xgb.fit(x_train, y_train)
                pbar.update(1)  # Update the progress bar

    def predict(self, x):
        return self.xgb.predict(x)
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pk.dump(self.xgb, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.xgb = pk.load(f)