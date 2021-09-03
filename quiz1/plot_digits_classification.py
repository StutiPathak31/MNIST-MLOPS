print(__doc__)

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np

def trainModel(testsplit, imgsize):
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    original = digits.images

    resize = []
    for img in original:
      resize.append(rescale(img, imgsize, anti_aliasing=False)) 
    resize = np.array(resize)

    data = resize.reshape((n_samples, -1))
    
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=testsplit, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

 
    accuracy = metrics.accuracy_score(y_test, predicted)
    f1 = metrics.f1_score(y_test, predicted, average = 'macro')
    print(resize.shape[1],'X',resize.shape[1],'->',int((1-testsplit)*100),':',int(testsplit*100),'->', round(accuracy*100, 2),'->',round(f1*100, 2))
    
def runVariousSize():
  print("Image Size -> Train-Test Split -> Accuracy -> F1 Score")
  trainModel(0.1, 0.5)
  trainModel(0.2, 0.5)
  trainModel(0.3, 0.5)
  print()
  trainModel(0.1, 0.75)
  trainModel(0.2, 0.75)
  trainModel(0.3, 0.75)
  print()
  trainModel(0.1, 1)
  trainModel(0.2, 1)
  trainModel(0.3, 1)

runVariousSize()