# MNIST CNN Project

This project trains a Convolutional Neural Network (CNN) on the MNIST dataset.

## Directory Structure

- `data/`: Directory for datasets.
- `notebooks/`: Jupyter or Colab notebooks.
- `src/`: Source code.
- `tests/`: Unit tests.
- `.gitignore`: Files to ignore in Git.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.
- `setup.py`: Installation script for the project.

## Setup

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the Virtual Environment:
    ```bash
    .\venv\Scripts\Activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Train the model:
    ```bash
    python src\train.py
    ```

5. Evaluate the model:
    ```bash
    python src\evaluate.py
    ```
