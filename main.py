from utils.data_loader import Dataloader
from models import get_model
from utils.evaluation import evaluate, save_classification_report
# from keras import models, utils
import tensorflow as tf
import numpy as np
from utils.visualisation import Visuals
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load data
loader = Dataloader()
visuals = Visuals()

# This function is for training just the CNN
## One hot encoding for labels is set to True
## Flattening of images set to False
def train_and_evaluate_CNN(one_hot=True, flatten=False):
    save_path = "results/models/cnn_model.keras"
    x_train, x_val, x_test, y_train, y_val, y_test = loader.load_dataset(one_hot=one_hot, flatten=flatten)
    cnn = get_model("cnn", num_classes=10)

    cnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    trained_model = cnn.fit(x_train, y_train, validation_split=0.3, epochs=50, batch_size=256, verbose=1, shuffle=True)
    print("Fitted")

    visuals.plot_train_val_loss(trained_model, "CNN")
    visuals.plot_train_val_accuracy(trained_model, "CNN")

    cnn.save(save_path)
    print("saved")

    with tf.keras.utils.CustomObjectScope({'CNNModel': cnn}):  # Register the custom class
        cnn = cnn.load(save_path)
    print("loaded")

    y_pred_probs = cnn.predict(x_test)  # Outputs probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels
    print("Predicted")

    # visuals.plot_confision_matrix(y_pred, y_test, loader.class_names)
    y_test = np.argmax(y_test, axis=1)
    visuals.plot_ROC_AUC(y_test, y_pred_probs, loader.class_names, "CNN")
    accuracy = evaluate('accuracy', y_test, y_pred)
    report = evaluate('classification_report', y_test, y_pred, target_names=loader.class_names)
    report_dict = evaluate('classification_report_dict', y_test, y_pred, target_names=loader.class_names)
    save_classification_report(report_dict, name="CNN_classification_report", path="results/reports")

    print(f"CNN Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

    return accuracy, report

# This function trains every other model
## Flattening of images set to True
def train_and_evaluate_model(model_code, model_name, flatten=True):
    save_path = f"results/models/{model_name}_model.pkl"
    x_train, _, x_test, y_train, _, y_test = loader.load_dataset(flatten=flatten)
    model = get_model(model_code)
    model.fit(x_train, y_train)
    print("Fitted")
    model.save(save_path)
    print("Saved")
    model.load(save_path)
    print("Loaded")
    y_pred = model.predict(x_test)
    y_pred_probs = model.predict_proba(x_test)
    print("Pridicted")

    accuracy = evaluate('accuracy', y_test, y_pred)
    report = evaluate('classification_report', y_test, y_pred, target_names=loader.class_names)
    report_dict = evaluate('classification_report_dict', y_test, y_pred, target_names=loader.class_names)
    save_classification_report(report_dict, name=f"{model_name}_classification_report", path="results/reports")

    visuals.plot_confision_matrix(y_pred, y_test, loader.class_names, model_name=model_name)
    visuals.plot_ROC_AUC(y_test, y_pred_probs, loader.class_names, model_name=model_name)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

    return accuracy, report


# x_train, x_val, x_test, y_train, y_val, y_test = loader.load_dataset()
# visuals.visualise_cifar10(x_train, y_train, loader.class_names, num_images=4)

# Execution of training CNN
accuracy, results = train_and_evaluate_CNN(num_classes=10)

# Execution of training other models, view model map in models/__init__.py for model codes
accuracy, results = train_and_evaluate_model("knn", "KNN")