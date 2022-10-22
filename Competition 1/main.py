import sklearn
from sklearn.model_selection import cross_val_score
import csv

#Loading the data into memory
with open("train.csv", 'r', newline='') as train:
    with open("train_result.csv", 'r', newline='') as labels:
        # Training the model
        train_data = csv.reader(train, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        train_labels = csv.reader(labels, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print(train_data.__next__())
        print(train_labels.__next__())
        print(train_data.__next__())
        print(train_labels.__next__())

# # Cross validation
# clf = sklearn.svm.SVC(kernel='linear', C=1, random_state=42)
# scores = cross_val_score(clf, X, y, cv=5)
# scores
# array([0.96..., 1. , 0.96..., 0.96..., 1. ])