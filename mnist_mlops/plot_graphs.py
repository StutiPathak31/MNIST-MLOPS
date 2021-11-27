import os

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers
from sklearn import datasets, svm, tree

from skimage import data, color

import numpy as np

from joblib import dump, load

from utils import preprocess, create_splits, test
import pandas as pd


digits = datasets.load_digits()

#parent_directory = '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS'
parent_directory = os.getcwd()
if not os.path.exists(parent_directory + '/models'):
    os.mkdir(parent_directory+'/models')
parent_directory += '/models'
# flatten the images
n_samples = len(digits.images)
#print(parent_directory)
def findBestModel():

    # rescale_factors = [0.25, 0.5, 1, 2, 3]
    gammalist = [10 ** exp for exp in range(-7, 0)]
    max_depth = [5, 6, 7, 8, 9, 10, 20, 30, 40, 80, 100]
    rescale_factors = [1]
    models = ['SVM']
    hyperp = {'SVM':[0.00001, 0.001, 1]}
    bestModel = {}
    data_metric = {'Hyperparameter':[], 'Train_1':[], 'Dev_1':[], 'Test_1':[],'Train_2':[], 'Dev_2':[], 'Test_2':[], 'Train_3':[], 'Dev_3':[], 'Test_3':[]}
    
    for hp in hyperp['SVM']:
        data_metric['Hyperparameter'].append(hp)
    for run in range(3):
        for test_size, valid_size in [(0.15, 0.15)]:
            for rescale_factor in rescale_factors:
                resized_images = preprocess(
                        images=digits.images, rescale_factor=rescale_factor
                    )
                resized_images = np.array(resized_images)
                data = resized_images.reshape((n_samples, -1))
                X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
                        data, digits.target, test_size, valid_size
                    )

                for hp in hyperp['SVM']:
                    clf = svm.SVC(gamma = hp)
                    clf.fit(X_train, y_train)
                    metrics_valid = test(clf, X_valid, y_valid)
                    
                    candidate = {
                        "acc_valid": metrics_valid['acc'],
                        "f1_valid": metrics_valid['f1'],
                        "hyperp": hp,
                        }


                    #model_candidates.append(candidate)
                    output_folder = parent_directory+"/run_{}_tt_{}_val_{}_rescale_{}_hyperp_{}".format(
                        run, test_size, valid_size, rescale_factor, hp
                        )
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)
                    dump(clf, os.path.join(output_folder, "model.joblib"))

                    # Predict the value of the digit on the test subset
                    metrics_train = test(clf, X_train, y_train)
                    metrics_valid = test(clf, X_valid, y_valid)
                    metrics_test = test(clf, X_test, y_test)

                    #print(run+1, hp, metrics_train, metrics_test, metrics_valid)
                    #print(data)
                    #print(data['Hyperparameter'], type(data['Hyperparameter']))
                    #print('Train'+str(run+1), type('Train'+str(run+1)))

                    data_metric['Train_'+str(run+1)].append(metrics_train['acc'])
                    data_metric['Dev_'+str(run+1)].append(metrics_valid['acc'])
                    data_metric['Test_'+str(run+1)].append(metrics_test['acc'])


    return data_metric




#runMultiple()
def singleRun():
    param = findBestModel()
    df = pd.DataFrame(param)
    df['Train_Mean'] = df[['Train_1', 'Train_2', 'Train_3']].mean(axis=1)
    df['Dev_Mean'] = df[['Dev_1', 'Dev_2', 'Dev_3']].mean(axis=1)
    df['Test_Mean'] = df[['Test_1', 'Test_2', 'Test_3']].mean(axis=1)
    print(df)

singleRun()
        