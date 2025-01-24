from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.svc_model import SVCModel
from models.logistic_regression_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LGBM
from models.catboost_model import CatBoosst
from models.random_forest_model import RFCModel

__all__ = [
    'CNNModel',
    'KNNModel',
    'SVCModel',
    'LogisticRegressionModel',
    'XGBoostModel',
    'LGBM',
    'CatBoosst',
    'RFCModel'
]

_model_map = {
    'cnn': CNNModel,
    'knn': KNNModel,
    'svc': SVCModel,
    'lgr': LogisticRegressionModel,
    'xgb': XGBoostModel,
    'lgbm': LGBM,
    'cat': CatBoosst,
    'forest': RFCModel
}


def get_model(name, **kwargs):
    """
    Retrieve a machine learning model based on the provided name.

    Parameters:
        name (str): The identifier for the model to retrieve. Must be one of the keys in _model_map.
        **kwargs: Additional keyword arguments to pass to the model's constructor.

    Returns:
        object: An instance of the specified machine learning model.

    Raises:
        ValueError: If the provided name does not match any key in _model_map.
    """
    name = name.lower()
    if name not in _model_map:
        raise ValueError(f"Unknown model: {name}.")
    return _model_map[name](**kwargs)
