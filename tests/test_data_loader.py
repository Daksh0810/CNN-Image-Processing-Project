from src.data_loader import load_and_preprocess_data

def test_load_and_preprocess_data():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    assert X_train.shape == (60000, 28, 28, 1)
    assert X_test.shape == (10000, 28, 28, 1)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
