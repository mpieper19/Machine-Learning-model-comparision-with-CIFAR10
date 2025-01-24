from models.model_template import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
import pickle as pk
from tqdm import tqdm

class GBCModel(BaseModel):
    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3):
        self.gbc = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )

    def fit(self, x_train, y_train):
        """
        Fits the Gradient Boosting model with a progress bar.
        """
        # Initialize progress bar
        with tqdm(total=self.gbc.n_estimators, desc="Training Progress") as pbar:
            for i in range(self.gbc.n_estimators):
                # Incrementally fit each stage
                self.gbc.set_params(n_estimators=i + 1)  # Update the number of trees
                self.gbc.fit(x_train, y_train)
                pbar.update(1)  # Update the progress bar

    def predict(self, x):
        return self.gbc.predict(x)
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pk.dump(self.knn, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.knn = pk.load(f)