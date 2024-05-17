import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from data_loader import load_and_preprocess_data

def evaluate_model():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    model = load_model('mnist_cnn_model.h5')
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")

    sample_predictions = model.predict(X_test[:5])
    predicted_classes = np.argmax(sample_predictions, axis=1)

    for i in range(5):
        actual_label = np.argmax(y_test[i])
        predicted_label = predicted_classes[i]
        print(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}")
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        plt.show()

if __name__ == "__main__":
    evaluate_model()
