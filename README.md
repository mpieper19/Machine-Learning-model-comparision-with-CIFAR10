# Machine-Learning-model-comparision-with-CIFAR10 (Uni Project)
Intro to Machine Learning WBAI056

## Table of Contents
- [About the Proejct](#about-the-project)
- [Built With](#built-with)
- [Usage](#usage)
- [Results](#results)

## About the Project

This project was conducted with two other students, where we comapre the accuracy and effectiveness of a (custom-built) Convolutional Neural Network (CNN), to a handful of other standard machine learning models, on the CIFAR-10 dataset. The following machine learning models are used:
- K-Nearest-Neighbors (KNN)
- Light-Gradient-Boosting-Model (LightGBM)
- CatBoost
- Random Forest Classifier

## Built With
- Python
- Tensorflow
- Keras
- Scikit Learn
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Catboost
- LightGBM

Use `pip install -r requirements.txt` to install required dependencies.

## Usage

To test and run each model, configure the `main.py` file according to the desired model:

### For the CNN Model:
- Use the `train_and_evaluate_CNN()` function to train and evaluate the CNN model.
- Ensure the following:
  - One-hot encoding for labels is enabled (`one_hot=True`).
  - Image flattening is disabled (`flatten=False`), as the CNN processes structured image data.
- Results, including training/validation loss and accuracy curves, ROC/AUC curves, the model object, and classification reports, will be saved in the `results` directory.

Run the CNN model with the following code:
```python
accuracy, results = train_and_evaluate_CNN()
```

### For Classic Machine Learning Models:
- Use the `train_and_evaluate_model(model_code, model_name)` function for classic models. `model_code` and `model_name` can be retrieved from the `_model_map` dictionary found in `models\__init__.py`.
```python
_model_map = {
    'cnn': CNNModel,
    'knn': KNNModel,
    'lgbm': LGBM,
    'cat': CatBoosst,
    'forest': RFCModel
}
```
- Ensure the following:
  - Set `flattenn=True` to prepare the images as 1D vectors, as required by classic ML models.
- Results, including confusion matrices, ROC/AUC curves, the model object, and classification reports, will be saved in the `results` directory.

Run the CNN model with the following code:
```python
accuracy, results = train_and_evaluate_model("knn", "KNN")
```
For example, to train the Random Forest model:
```python
accuracy, results = train_and_evaluate_model("forest", "RandomForest")
```

## Results

The classification reports for the models are saved as CSV files in the `results/reports/` directory. For example:
- [CNN Classification Report](results/reports/CNN_classification_report.csv)
- [KNN Classification Report](results/reports/KNN_classification_report.csv)
- [Random Forest Classifier Classification Report](results/reports/randomforest_classification_report.csv)
- [LightGBM Classification Report](results/reports/LightGBM_classification_report.csv)
- [CatBoost Classification Report](results/reports/CatBoost_classification_report.csv)

### CNN Results:
CNN Accuracy: **70%**

CNN Training and Validation Loss Plot:

![image](https://github.com/user-attachments/assets/7141c367-967f-4cda-83fa-aa63baef773d)


CNN Training and Validation Accuracy Plot:

![image](https://github.com/user-attachments/assets/24139414-1d56-4715-92da-adf74a6753ec)


CNN ROC/AUC Curve:

![image](https://github.com/user-attachments/assets/610943af-d81c-4373-b59f-1c4ae041b205)

### KNN Results:
KNN Accuracy: **32%**

KNN Confusion Matrix:

![image](https://github.com/user-attachments/assets/680a207f-a2cc-4b79-b7de-c40811093a66)

KNN ROC/AUC Curve:

![image](https://github.com/user-attachments/assets/2d21f017-664e-4a9c-beb5-5a0404e79d5a)

### Random Forest Classifier Results:
Random Forest Classifier accuracy: **46%**

Random Forest Classifer Confusion Matrix:

![image](https://github.com/user-attachments/assets/9a212798-3e17-48c7-a692-71e2841ad99f)

Random Forest Classifier ROC/AUC Curve:

![image](https://github.com/user-attachments/assets/6f8961ca-630c-4d1d-a8a7-c68a8215d098)


### LightGBM Results:
LightGBM Accuracy: **53%**

LightGBM Loss Plot:

LightGBM Confusion Matrix:

![image](https://github.com/user-attachments/assets/e76f81ee-c215-46f4-8d4c-400fe483c648)

LightGBM ROC/AUC Curve:

![image](https://github.com/user-attachments/assets/543ef416-8b8e-43ae-89ae-8faab40b0c58)

### CatBoost Results:
CatBoost Accuracy: **58%**

Catboost Loss Plot:

![image](https://github.com/user-attachments/assets/0828f635-deeb-4949-b61c-d2100d632c99)


CatBoost Confusionn Matrix:

![image](https://github.com/user-attachments/assets/6ab7737d-16ef-4392-989e-b11ddc2dff5f)


CatBoost ROC/AUC Curve:

![image](https://github.com/user-attachments/assets/b4a42fc1-14ed-4c72-b639-81f79eacf0c1)


## License
This project is licensed under the MIT License.
