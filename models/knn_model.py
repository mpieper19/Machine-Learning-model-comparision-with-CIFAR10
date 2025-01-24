from models.model_template import BaseModel
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk


class KNNModel(BaseModel):
    def __init__(self, k=3):
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x):
        return self.knn.predict(x)

    def predict_proba(self, x):
        return self.knn.predict_proba(x)

    def save(self, file_path):
        self._mkdir()
        with open(file_path, 'wb') as f:
            pk.dump(self.knn, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.knn = pk.load(f)
