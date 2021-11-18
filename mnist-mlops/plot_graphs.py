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

parent_directory = '/mnt/c/Users/Stuti Pathak/Desktop/IIT Jodhpur/Semester 3/MLOps/MNIST-MLOPS'
if not os.path.exists(parent_directory + '/models'):
    os.mkdir(parent_directory+'/models')
parent_directory += '/models'
# flatten the images
n_samples = len(digits.images)

def findBestModel():

    # rescale_factors = [0.25, 0.5, 1, 2, 3]
    gammalist = [10 ** exp for exp in range(-7, 0)]
    max_depth = [5, 6, 7, 8, 9, 10, 20, 30, 40, 80, 100]
    rescale_factors = [1]
    models = ['SVM', 'DecisionTree']
    hyperp = {'SVM':[10 ** exp for exp in range(-7, 0)], 'DecisionTree':[5, 6, 7, 8, 9, 10, 20, 30, 40, 80, 100]}
    bestModel = {}
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

            for model in models:
                if not os.path.exists(parent_directory+'/'+ model):
                        os.mkdir(parent_directory+'/'+ model)
                model_candidates = []
                for hp in hyperp[model]:
                    if model == 'DecisionTree':
                        clf = tree.DecisionTreeClassifier(max_depth = hp)
                    elif model == 'SVM':
                        clf = svm.SVC(gamma = hp)

                    clf.fit(X_train, y_train)
                    metrics_valid = test(clf, X_valid, y_valid)
                
                    candidate = {
                        "acc_valid": metrics_valid['acc'],
                        "f1_valid": metrics_valid['f1'],
                        "hyperp": hp,
                        }
                    #print(candidate)
                    if metrics_valid['acc'] < 0.11:
                        #print("Skipping for {}".format(hp))
                        continue

                    model_candidates.append(candidate)
                    output_folder = parent_directory+"/"+model+"/tt_{}_val_{}_rescale_{}_hyperp_{}".format(
                        test_size, valid_size, rescale_factor, hp
                        )
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)
                    dump(clf, os.path.join(output_folder, "model.joblib"))

            # Predict the value of the digit on the test subset

                max_valid_f1_model_candidate = max(
                    model_candidates, key=lambda x: x["f1_valid"]
                    )
                best_model_folder = parent_directory+"/"+model+"/tt_{}_val_{}_rescale_{}_hyperp_{}".format(
                    test_size, valid_size, rescale_factor, max_valid_f1_model_candidate['hyperp']
                    )
                clf = load(os.path.join(best_model_folder, "model.joblib"))

                metrics = test(clf, X_test, y_test)
                #print(
                #    "{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format(
                #        resized_images[0].shape[0],
                #        resized_images[0].shape[1],
                #        max_valid_f1_model_candidate["hyperp"],
                #        (1 - test_size) * 100,
                #        test_size * 100,
                #        metrics['acc'],
                #        metrics['f1'],
                #    )
                #)
                bestModel[model] = {'hyperparameter':max_valid_f1_model_candidate['hyperp'], 'acc':metrics['acc'], 'f1':metrics['f1']}
    return bestModel

def runMultiple():
    df = pd.DataFrame(columns=['Run', 'TreeDepth', 'TreeAcc', 'TreeF1', 'SVMGamma', 'SVMAcc', 'SVMF1'])
    for i in range(5):
        param = findBestModel()
        temp = {}
        temp['Run'] = int(i + 1)
        temp['TreeDepth'] = param['DecisionTree']['hyperparameter']
        temp['TreeAcc'] = param['DecisionTree']['acc']
        temp['TreeF1'] = param['DecisionTree']['f1']
        temp['SVMGamma'] = param['SVM']['hyperparameter']
        temp['SVMAcc'] = param['SVM']['acc']
        temp['SVMF1'] = param['SVM']['f1']
        #print(temp)
        df = df.append(temp, ignore_index = True)

    dfmean = df.mean()
    dfmean['Run'] = 'Mean'
    dfstd = df.std()
    dfstd['Run'] = 'StdDev'
    df = df.append(dfmean, ignore_index = True)
    df = df.append(dfstd, ignore_index = True)
    df['TreeDepth'][5] = ""
    df['TreeDepth'][6] = ""
    df['SVMGamma'][5] = ""
    df['SVMGamma'][6] = ""
    print(df)


runMultiple()
        










            

            