from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.lightgbm_model import LGBM
from models.catboost_model import CatBoosst
from models.random_forest_model import RFCModel

__all__ = [
    'CNNModel',
    'KNNModel',
    'LGBM',
    'CatBoosst',
    'RFCModel'
]

_model_map = {
    'cnn': CNNModel,
    'knn': KNNModel,
    'lgbm': LGBM,
    'cat': CatBoosst,
    'forest': RFCModel
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _model_map:
        raise ValueError(f"Unknown model: {name}.")
    return _model_map[name](**kwargs)
