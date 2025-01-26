import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,

)



def evaluate(name, y_true, y_pred, target_names=None):
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
    df = pd.DataFrame(report).transpose()
    full_path = f"{path}/{name}.csv"
    df.to_csv(full_path, index=True)
    print(f"Classification report saved at {full_path}")
