from setuptools import setup, find_packages # type: ignore

setup(
    name='mnist_cnn_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib'
    ],
)
