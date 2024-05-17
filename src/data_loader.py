import tensorflow as tf # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return X_train, y_train, X_test, y_test
