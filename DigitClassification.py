"""Various number of handwritten digits will be given and we have to train our classifier according to training data"""
# written in python2.7
import cPickle, gzip
import numpy as np

#train data
f = open("mnist_10000.pkl", 'r')
trainData, trainLabels, valData, valLabels, testData, testLabels = cPickle.load(f)
f.close()

print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

#Looking at images
import cv2
image = trainData[1]
image = image.reshape((28, 28))
#cv2.imshow("Image", image)

from sklearn.neighbors import KNeighborsClassifier
print(1)
model = KNeighborsClassifier(n_neighbors = 25, metric="minkowski", p=2) 
print(2)
# p=2 for euclidian distance
model.fit(trainData, trainLabels)
print(3)
# Check score on validation data
print 'for k=25:', model.score(valData, valLabels)

#predict for test data
pred = model.predict(testData)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print "accuracy on test data:", accuracy_score(pred, testLabels)
print "Classification Report is: \n", classification_report(testLabels, pred)
print "Confusion Matrix is: \n", confusion_matrix(testLabels, pred)
