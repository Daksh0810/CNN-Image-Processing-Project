from src.model import create_cnn_model

def test_create_cnn_model():
    input_shape = (28, 28, 1)
    num_classes = 10
    model = create_cnn_model(input_shape, num_classes)
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
