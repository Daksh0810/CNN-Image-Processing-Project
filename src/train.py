import tensorflow as tf # type: ignore
from data_loader import load_and_preprocess_data
from model import create_cnn_model

def train_model(epochs=10):
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    input_shape = (28, 28, 1)
    num_classes = 10

    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model, history

if __name__ == "__main__":
    model, history = train_model(epochs=10)
    model.save('mnist_cnn_model.h5')