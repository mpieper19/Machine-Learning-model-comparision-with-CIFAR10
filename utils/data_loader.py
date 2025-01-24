import tensorflow as tf
from sklearn.model_selection import train_test_split


class Dataloader:
    def __init__(self):
        self.class_names = [
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

    def load_dataset(self, flatten=False, one_hot=False, limit_train=None, limit_test=None):
        # Load the CIFAR-10 dataset via tensorflow
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Add a limit to how many training images
        if limit_train is not None:
            x_train = x_train[:limit_train]
            y_train = y_train[:limit_train]

        if limit_test is not None:
            x_test = x_test[:limit_test]
            y_test = y_test[:limit_test]

        # Normalize the images
        x_train = x_train / 255.0       # (50000, 32, 32, 3)
        x_test = x_test / 255.0         # (10000, 32, 32, 3)

        # Flatten the normalized images if flatten=True
        if flatten:
            x_train = x_train.reshape(x_train.shape[0], -1)     # (50000, 3072)
            x_test = x_test.reshape(x_test.shape[0], -1)        # (10000, 3072)

        if one_hot:
            y_train = tf.keras.utils.to_categorical(y_train, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)
        else:
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        return x_train, x_val, x_test, y_train, y_val, y_test
