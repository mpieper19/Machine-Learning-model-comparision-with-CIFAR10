from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import pandas as pd


def evaluate(name, y_true, y_pred, target_names=None):
    """
    Evaluate a model on a given set of true labels and predicted labels using various metrics.

    Parameters
    ----------
    name : str
        Name of the metric to compute. Can be one of the following:
            - accuracy
            - precision
            - recall
            - f1_score
            - confusion_matrix
            - classification_report
            - classification_report_dict
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    target_names : list of str, optional
        List of names of classes or states, in the order they index the underlying
        data. Used to determine the order of the matrix rows and columns; used only
        if labels is undefined.

    Returns
    -------
    result : object
        The computed metric. The type of the result is determined by the metric name.
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, zero_division=0, target_names=target_names),
        'classification_report_dict': classification_report(y_true, y_pred, zero_division=0, target_names=target_names, output_dict=True)
    }

    if name in results:
        return results[name]
    else:
        raise ValueError(f"Unknown metric name: {name}")


def save_classification_report(report, name, path):
    """
    Save a classification report to a CSV file.

    Parameters
    ----------
    report : dict
        Classification report dictionary as returned by sklearn.metrics.classification_report.
    name : str
        Name of the report, used as the filename.
    path : str
        Path to the directory where the report will be saved.

    Returns
    -------
    None
    """
    df = pd.DataFrame(report).transpose()
    full_path = f"{path}/{name}.csv"
    df.to_csv(full_path, index=True)
    print(f"Classification report saved at {full_path}")
