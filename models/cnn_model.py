import tensorflow as tf
from models.model_template import BaseModel
from tensorflow.keras import layers, regularizers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="CustomModels")
class CNNModel(tf.keras.Model, BaseModel):
    def __init__(self, num_classes, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        self.num_classes = num_classes

        # Convolution Blocks
        ## First Block
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.006))
        self.batchnorm1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.drop1 = layers.Dropout(0.25)

        ## Second Block
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.006))
        self.batchnorm2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.drop2 = layers.Dropout(0.25)

        ## Third Block
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.006))
        self.batchnorm3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.drop3 = layers.Dropout(0.35)

        ## Final layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.008))
        self.drop4 = layers.Dropout(0.5)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop4(x)

        return self.output_layer(x)


    def fit(self, x_train, y_train, **kwargs):
        return super().fit(x_train, y_train, **kwargs)

    def predict(self, x):
        return super().predict(x)

    def save(self, file_path):
        super().save(file_path)
        self._mkdir()
        print(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        return tf.keras.models.load_model(file_path, custom_objects={'CNNModel': CNNModel})

    def get_config(self):
        config = super(CNNModel, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)