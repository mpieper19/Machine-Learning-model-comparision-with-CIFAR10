import matplotlib.pyplot as mp
import numpy as np
import seaborn as sb
from utils.evaluation import evaluate
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class Visuals():

    def visualise_cifar10(self, x_data, y_data, class_names, num_images=4):
        indices = np.random.choice(range(x_data.shape[0]), size=num_images, replace=False)
        mp.figure(figsize=(12, 4))
        for i, idx in enumerate(indices):
            mp.subplot(1, num_images, i + 1)
            mp.imshow(x_data[idx])
            mp.title(f"Label: {class_names[y_data[idx]]}")
            mp.axis('off')
        mp.tight_layout()
        mp.savefig("results/visualisations/4_example_training_images.png")
        mp.show()

    # CNN Visuals
    ## This function plots the traiing and value losses
    def plot_train_val_loss(self, trained_model, model_name):
        """
        Plot the training and validation loss for a given model.

        Parameters
        ----------
        trained_model : object
            The trained model with history attribute.
        model_name : str
            The name of the model, used to label the plot.

        Returns
        -------
        None
        """
        mp.figure(figsize=(5, 4))
        mp.plot(trained_model.history['loss'], label='Train Loss')
        mp.plot(trained_model.history['val_loss'], label='Val Loss')
        mp.title(f'{model_name} Model Loss')
        mp.xlabel('Epochs')
        mp.ylabel('Loss')
        mp.legend()

        mp.tight_layout()
        mp.savefig(f"results/visualisations/{model_name}_loss.png")
        mp.show()

    ## This function plots the training and value accuracies
    def plot_train_val_accuracy(self, trained_model, model_name):
        """
        Plot the training and validation accuracy for a given model.

        Parameters
        ----------
        trained_model : object
            The trained model with history attribute.
        model_name : str
            The name of the model, used to label the plot.

        Returns
        -------
        None
        """
        mp.figure(figsize=(5, 4))
        mp.plot(trained_model.history['accuracy'], label='Train Accuracy')
        mp.plot(trained_model.history['val_accuracy'], label='Val Accuracy')
        mp.title(f'{model_name} Model Accuracy')
        mp.xlabel('Epochs')
        mp.ylabel('Accuracy')
        mp.legend()

        mp.tight_layout()
        mp.savefig(f"results/visualisations/{model_name}_accuracy.png")
        mp.show()

    # Other Visuals
    def plot_confision_matrix(self, y_pred, y_true, classes, model_name):
        """
        Plot the confusion matrix for a given model.

        Parameters
        ----------
        y_pred : array-like
            Predicted labels.
        y_true : array-like
            True labels.
        classes : list
            List of class labels.
        model_name : str
            The name of the model, used to label the plot.

        Returns
        -------
        None
        """
        con_matrix = evaluate("confusion_matrix", y_pred, y_true)
        mp.figure(figsize=(8, 6))
        sb.heatmap(con_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        mp.title(f"Confusion Matrix: {model_name}")
        mp.xlabel("Predicted Labels")
        mp.ylabel("True Labels")
        mp.savefig(f"results/visualisations/{model_name}_confusionmatrix.png")
        mp.show()

    def plot_ROC_AUC(self, y_true, y_pred_probs, classes, model_name):
        """
        Plot the ROC curve and compute the AUC for a given model.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred_probs : array-like
            Predicted probabilities.
        classes : list
            List of class labels.
        model_name : str
            The name of the model, used to label the plot.

        Returns
        -------
        None
        """
        y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
        n_classes = len(classes)

        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_binarized.ravel(), y_pred_probs.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot all ROC curves
        mp.figure(figsize=(10, 8))
        for i in range(n_classes):
            mp.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

        mp.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
        mp.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.50)")
        mp.title(f"ROC Curve: {model_name}")
        mp.xlabel("False Positive Rate")
        mp.ylabel("True Positive Rate")
        mp.legend(loc="best")
        mp.tight_layout()
        mp.savefig(f"results/visualisations/{model_name}_roc_auc.png")
        mp.show()
