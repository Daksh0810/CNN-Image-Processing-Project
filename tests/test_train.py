from src.train import train_model

def test_train_model():
    model, history = train_model(epochs=1)
    assert 'accuracy' in history.history
    assert 'val_accuracy' in history.history
