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


def train_and_evaluate_CNN(
        num_classes=10,
        save_path="results/models/cnn_model.keras",
        flatten=False
):
    """
    Trains and evaluates a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

    Parameters:
        num_classes (int): The number of classes for classification. Default is 10.
        save_path (str): The file path to save the trained CNN model. Default is "results/models/cnn_model.keras".
        flatten (bool): Whether to flatten the input images. Default is False.

    Returns:
        tuple: A tuple containing the accuracy and the classification report of the model.

    This function loads the dataset, compiles and trains a CNN model, evaluates its performance,
    and generates various visualizations. The trained model is saved and reloaded for making predictions.
    """
    x_train, x_val, x_test, y_train, y_val, y_test = loader.load_dataset(one_hot=True, flatten=flatten)
    cnn = get_model("cnn", num_classes=num_classes)

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


def train_and_evaluate_model(model_code, model_name, flatten=True):
    """
    Trains and evaluates a machine learning model on the CIFAR-10 dataset.

    Parameters:
        model_code (str): The code identifier for the model to be trained.
        model_name (str): The name of the model for saving and reporting purposes.
        flatten (bool): Whether to flatten the input images. Default is True.

    Returns:
        tuple: A tuple containing the accuracy and the classification report of the model.

    This function loads the dataset, compiles and trains the specified model, evaluates its performance,
    and generates various visualizations. The trained model is saved and reloaded for making predictions.
    """

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


x_train, x_val, x_test, y_train, y_val, y_test = loader.load_dataset()
visuals.visualise_cifar10(x_train, y_train, loader.class_names, num_images=4)
# accuracy, results = train_and_evaluate_CNN(num_classes=10)
# accuracy, results = train_and_evaluate_model("knn", "KNNTEST")