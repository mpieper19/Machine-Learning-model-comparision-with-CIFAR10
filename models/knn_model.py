from models.model_template import BaseModel
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk


class KNNModel(BaseModel):
    def __init__(self, k=3):
        self.model = KNeighborsClassifier(n_neighbors=k)

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
