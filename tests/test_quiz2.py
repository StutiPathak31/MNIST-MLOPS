import os
import sys
sys.path.insert(1, '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS/mnist_mlops')
print(os.getcwd())
from sklearn import datasets
from utils import create_splits, preprocess
import numpy as np
import math

def Quiz2():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    resized_images = preprocess(
                images=digits.images, rescale_factor=1
            )
    resized_images = np.array(resized_images)
    data = resized_images.reshape((n_samples, -1))

    target = digits.target
    test_size = 0.2
    valid_size = 0.1
    X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(data, target, test_size, valid_size)
    return X_train, X_test, X_valid, y_train, y_test, y_valid, n_samples
    

def test1():
    X_train, X_test, X_valid, y_train, y_test, y_valid, n_samples = Quiz2()
    assert len(X_train) == int(n_samples * 0.7)

def test2():
    X_train, X_test, X_valid, y_train, y_test, y_valid, n_samples = Quiz2()
    assert len(X_valid) == int(math.ceil(n_samples * 0.1))

def test3():
    X_train, X_test, X_valid, y_train, y_test, y_valid, n_samples = Quiz2()
    assert len(X_test) == int(math.ceil(n_samples * 0.2))

def test4():
    X_train, X_test, X_valid, y_train, y_test, y_valid, n_samples = Quiz2()
    summation = len(X_train) + len(X_test) + len(X_valid)
    assert summation == n_samples
    