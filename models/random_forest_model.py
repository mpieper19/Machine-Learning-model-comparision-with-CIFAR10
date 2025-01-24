from models.model_template import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from tqdm import tqdm


class RFCModel(BaseModel):
    def __init__(self):
        self.rfc = RandomForestClassifier()

    def fit(self, x_train, y_train):
        """
        Fits the Gradient Boosting model with a progress bar.
        """
        # Initialize progress bar
        with tqdm(total=self.rfc.n_estimators, desc="Training Progress") as pbar:
            for i in range(self.rfc.n_estimators):
                # Incrementally fit each stage
                self.rfc.set_params(n_estimators=i + 1)  # Update the number of trees
                self.rfc.fit(x_train, y_train)
                pbar.update(1)  # Update the progress bar

    def predict(self, x):
        return self.rfc.predict(x)

    def predict_proba(self, x):
        return self.rfc.predict_proba(x)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pk.dump(self.rfc, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.rfc = pk.load(f)