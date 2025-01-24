from models.model_template import BaseModel
from sklearn.svm import SVC
import pickle as pk


class SVCModel(BaseModel):
    def __init__(self, probability=True, kernel='linear', C=1.0):
        super().__init__()
        self.svc = SVC(probability=probability, kernel=kernel, C=C)

    def fit(self, x_train, y_train):
        self.svc.fit(x_train, y_train)

    def predict(self, x):
        return self.svc.predict(x)

    def predict_proba(self, x):
        return self.svc.predict_proba(x)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pk.dump(self.svc, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.svc = pk.load(f)
