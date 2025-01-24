from abc import ABC, abstractmethod
import os


class BaseModel(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, file_path):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    def _mkdir(self):
        base_dir = "results"
        sub_dirs = ["models", "reports", "visualisations"]
        for sub_dir in sub_dirs:
            path = os.path.join(base_dir, sub_dir)
            os.makedirs(path, exist_ok=True)
