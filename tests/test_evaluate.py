from src.train import train_model
from src.evaluate import evaluate_model

def test_evaluate_model():
    model, _ = train_model(epochs=1)
    evaluate_model(model)
