import tensorflow as tf
import matplotlib as mp
from sklearn.model_selection import train_test_split


class Dataloader:
    def __init__(self):
        self.clas_names = [
            "Airplanes",
            "Cars",
            "Birds",
            "Cats",
            "Deer",
            "Dogs",
            "Frogs",
            "Horses",
            "Ships",
            "Trucks"
        ]

    def load_dataset(self):
        # Load the CIFAR-10 dataset via tensorflow
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize the images
        x_train = x_train / 255.0       # (50000, 32, 32, 3)
        x_test = x_test / 255.0         # (10000, 32, 32, 3)

        # Flatten the normalized images
        x_train = x_train.reshape(x_train.shape[0], -1)     # (50000, 3072)
        x_test = x_test.reshape(x_test.shape[0], -1)        # (10000, 3072)

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        return x_train, x_val, x_test, y_train, y_val, y_test
