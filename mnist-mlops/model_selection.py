print(__doc__)

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np


print("ImageSize\tTrain-TestSplit\tGamma\tAccuracy\tF1 Score")
splitList = [0.1, 0.2, 0.3]
rescaleList = [0.5, 0.75, 1]
gammaList = [10**i for i in range(-6, 1)]
for imgsize in rescaleList:
  for testsplit in splitList:
    for gamma in gammaList:
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
      clf = svm.SVC(gamma=gamma)

      # Split data into 50% train and 50% test subsets
      X_train, X_test, y_train, y_test = train_test_split(
          data, digits.target, test_size=testsplit, shuffle=False)

      # Learn the digits on the train subset
      clf.fit(X_train, y_train)

      # Predict the value of the digit on the test subset
      predicted = clf.predict(X_test)

  
      accuracy = metrics.accuracy_score(y_test, predicted)
      f1 = metrics.f1_score(y_test, predicted, average = 'macro')
      print("{} X {}\t\t{} : {}\t{}\t{:.3f}\t\t{:0.3f}".format(
          resize.shape[1], resize.shape[1],
          (1-testsplit)*100, testsplit*100,
          gamma, accuracy*100, f1*100))
    print()
