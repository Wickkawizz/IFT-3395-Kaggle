import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
import csv

# Loading the data into memory
with open("train.csv", 'r', newline='') as train:
    with open("train_result.csv", 'r', newline='') as labels:
        # Training the model
        # Reading the data set (takes a while)
        train_reader = csv.reader(train, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        train_data = np.asarray(list(train_reader))
        print(np.shape(train_data))
        # (50001, 1569), 1569 features and 50001 rows

        # Reading the labels
        labels_reader = csv.reader(labels, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        train_labels = np.asarray(list(labels_reader))

        train_labels = np.asarray(train_labels)
        print(np.shape(train_labels))
        # (50001, 2), we only need the second column, because the first one is simply an index

        # TODO Implement a training algorithm with sklearn

        # TODO Implement a validation technique to choose the hyper parameter properly

        # TODO Implement the testing phase with the test.csv data set


        # knn_classifier = sklearn.neighbors.KNeighborsClassifier()
        # knn_classifier.fit(train_data[1:], train_labels[])

        # # Cross validation for SVM
        # clf = sklearn.svm.SVC(kernel='linear', C=1, random_state=42)
        # scores = cross_val_score(clf, X, y, cv=5)
        # scores
        # array([0.96..., 1. , 0.96..., 0.96..., 1. ])
