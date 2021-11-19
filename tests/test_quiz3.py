
import os
import sys
sys.path.insert(1, '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS/mnist_mlops')
print(os.getcwd())
from sklearn import datasets
from utils import create_splits, preprocess
import numpy as np
from joblib import load
import math

best_model_path_SVM = '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS/models/SVM/tt_0.15_val_0.15_rescale_1_hyperp_0.001/model.joblib'
best_model_path_DT = '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS/models/DecisionTree/tt_0.15_val_0.15_rescale_1_hyperp_40/model.joblib'

digits = datasets.load_digits()
n_samples = len(digits.images)
resized_images = preprocess(
                images=digits.images, rescale_factor=1
                )
resized_images = np.array(resized_images)
data = resized_images.reshape((n_samples, -1))

target = digits.target

clf_SVM = load(best_model_path_SVM)
clf_DT = load(best_model_path_DT)

min_acc = 0.80



def test_SVM_digit_correct_0():
    image = data[0].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==0


def test_SVM_digit_correct_1():
    image = data[1].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==1

def test_SVM_digit_correct_2():
    image = data[2].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==2

def test_SVM_digit_correct_3():
    image = data[3].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==3

def test_SVM_digit_correct_4():
    image = data[4].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==4

def test_SVM_digit_correct_5():
    image = data[15].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==5

def test_SVM_digit_correct_6():
    image = data[6].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==6

def test_SVM_digit_correct_7():
    image = data[7].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==7

def test_SVM_digit_correct_8():
    image = data[8].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==8

def test_SVM_digit_correct_9():
    image = data[9].reshape(1, -1)
    predicted = clf_SVM.predict(image)
    assert predicted==9

def test_DT_digit_correct_0():
    image = data[0].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==0

def test_DT_digit_correct_1():
    image = data[1].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==1

def test_DT_digit_correct_2():
    image = data[2].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==2

def test_DT_digit_correct_3():
    image = data[3].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==3

def test_DT_digit_correct_4():
    image = data[4].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==4

def test_DT_digit_correct_5():
    image = data[15].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==5

def test_DT_digit_correct_6():
    image = data[6].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==6

def test_DT_digit_correct_7():
    image = data[7].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==7

def test_DT_digit_correct_8():
    image = data[8].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==8

def test_DT_digit_correct_9():
    image = data[9].reshape(1, -1)
    predicted = clf_DT.predict(image)
    assert predicted==9

def test_threshold_SVM():


