import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# https://dhirajkumarblog.medium.com/logistic-regression-in-python-from-scratch-5b901d72d68e
# https://www.kaggle.com/code/hamzaboulahia/logistic-regression-mnist-classification

# We load the training set, the training labels and the test set
data = pd.read_csv('train.csv')
labels = pd.read_csv('train_result.csv')
test = pd.read_csv('test.csv')
print(data.head(10))
print(labels.head(10))
print(test.head(10))


# Function that normalizes the data
def normalize(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    data_normalized = (data - mean) / std
    return data_normalized


# We normalize the data so it is easier to handle
mnist_data_normalized = normalize(data)

# One vs all method, consists of making a vector for each class we have in our dataset (0-18, 19 classes)
Y_train_0 = (labels.iloc[:, 1] == 0).astype(int)
Y_train_1 = (labels.iloc[:, 1] == 1).astype(int)
Y_train_2 = (labels.iloc[:, 1] == 2).astype(int)
Y_train_3 = (labels.iloc[:, 1] == 3).astype(int)
Y_train_4 = (labels.iloc[:, 1] == 4).astype(int)
Y_train_5 = (labels.iloc[:, 1] == 5).astype(int)
Y_train_6 = (labels.iloc[:, 1] == 6).astype(int)
Y_train_7 = (labels.iloc[:, 1] == 7).astype(int)
Y_train_8 = (labels.iloc[:, 1] == 8).astype(int)
Y_train_9 = (labels.iloc[:, 1] == 9).astype(int)
Y_train_10 = (labels.iloc[:, 1] == 10).astype(int)
Y_train_11 = (labels.iloc[:, 1] == 11).astype(int)
Y_train_12 = (labels.iloc[:, 1] == 12).astype(int)
Y_train_13 = (labels.iloc[:, 1] == 13).astype(int)
Y_train_14 = (labels.iloc[:, 1] == 14).astype(int)
Y_train_15 = (labels.iloc[:, 1] == 15).astype(int)
Y_train_16 = (labels.iloc[:, 1] == 16).astype(int)
Y_train_17 = (labels.iloc[:, 1] == 17).astype(int)
Y_train_18 = (labels.iloc[:, 1] == 18).astype(int)


# We don't have the test labels, so we don't need this
# Y_test_0=(Y_test==0).astype(int)
# Y_test_1=(Y_test==1).astype(int)
# Y_test_2=(Y_test==2).astype(int)
# Y_test_3=(Y_test==3).astype(int)
# Y_test_4=(Y_test==4).astype(int)
# Y_test_5=(Y_test==5).astype(int)
# Y_test_6=(Y_test==6).astype(int)
# Y_test_7=(Y_test==7).astype(int)
# Y_test_8=(Y_test==8).astype(int)
# Y_test_9=(Y_test==9).astype(int)


def initializer(nbr_features):
    W = np.zeros((nbr_features, 1))
    B = 0
    return W, B


def ForwardBackProp(X, Y, W, B):
    m = X.shape[0]
    dw = np.zeros((W.shape[0], 1))
    dB = 0

    Z = np.dot(X, W) + B
    Yhat = sigmoid(Z)
    J = -(1 / m) * (np.dot(Y.T, np.log(Yhat)) + np.dot((1 - Y).T, np.log(1 - Yhat)))
    # TODO Look up why this line gives an error (ValueError: Data must be 1-dimensional)
    dW = (1 / m) * np.dot(X.T, (Yhat - Y))
    dB = (1 / m) * np.sum(Yhat - Y)
    return J, dW, dB


def predict(X, W, B):
    Yhat_prob = sigmoid(np.dot(X, W) + B)
    Yhat = np.round(Yhat_prob).astype(int)
    return Yhat, Yhat_prob


def gradient_descent(X, Y, W, B, alpha, max_iter):
    i = 0
    RMSE = 1
    cost_history = []

    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    while (i < max_iter) & (RMSE > 10e-6):
        J, dW, dB = ForwardBackProp(X, Y, W, B)
        W = W - alpha * dW
        B = B - alpha * dB
        cost_history.append(J)
        Yhat, _ = predict(X, W, B)
        RMSE = np.sqrt(np.mean(Yhat - Y) ** 2)
        i += 1
        if i % 50 == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")  # this ends the progress bar
    return cost_history, W, B, i


# Creating the model function which trains a model and return its parameters.
def LogRegModel(X_train, X_test, Y_train, alpha, max_iter):
    # TODO Analyze this line, I changed it from shape[1] and it gives something odd
    nbr_features = X_train.shape[1]
    W, B = initializer(nbr_features)
    cost_history, W, B, i = gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    Yhat_train, _ = predict(X_train, W, B)
    Yhat, _ = predict(X_test, W, B)

    #train_accuracy = accuracy_score(Y_train, Yhat_train)
    #test_accuracy = accuracy_score(Y_test, Yhat)
    #conf_matrix = confusion_matrix(Y_test, Yhat, normalize='true')

    model = {"weights": W,
             "bias": B,
             #"train_accuracy": train_accuracy,
             #"test_accuracy": test_accuracy,
             #"confusion_matrix": conf_matrix,
             "cost_history": cost_history}
    return model


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

print('Progress bar: 1 step each 50 iteration')
model_0 = LogRegModel(data, test, Y_train_0, alpha=0.01, max_iter=1000)
print('Training completed!')


cost = np.concatenate(model_0['cost_history']).ravel().tolist()
plt.plot(list(range(len(cost))),cost)
plt.title('Evolution of the cost by iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost');


# def optimize(x, y, learning_rate, iterations, parameters):
#     size = x.shape[0]
#     weight = parameters["weight"]
#     bias = parameters["bias"]
#     for i in range(iterations):
#         sigma = sigmoid(np.dot(x, weight) + bias)
#         loss = -1 / size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1 - sigma)
#         dW = 1 / size * np.dot(x.T, (sigma - y))
#         db = 1 / size * np.sum(sigma - y)
#         weight -= learning_rate * dW
#         bias -= learning_rate * db
#
#     parameters["weight"] = weight
#     parameters["bias"] = bias
#     return parameters
#
#
# print(data.shape[1])
# init_parameters = {}
# init_parameters["weight"] = np.zeros(data.shape[1])
# init_parameters["bias"] = 0
#
#
# def train(x, y, learning_rate, iterations):
#     parameters_out = optimize(x, y, learning_rate, iterations, init_parameters)
#     return parameters_out
#
#
# # Training the model
# parameters_out = train(data, labels.iloc[:, 1], learning_rate=0.02, iterations=1000)
#
# # Predictions using the model
# output_values = np.dot(data[:10], parameters_out["weight"]) + parameters_out["bias"]
# predictions = sigmoid(output_values) >= 1 / 2
# print(predictions)
#
# # Loading the data into memory
# with open("train.csv", 'r', newline='') as train:
#     with open("train_result.csv", 'r', newline='') as labels:
#         # Training the model
#         # Reading the data set (takes a while)
#         train_reader = csv.reader(train, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         train_data = np.asarray(list(train_reader))
#         print(np.shape(train_data))
#         # (50001, 1569), 1569 features and 50001 rows
#
#         # Reading the labels
#         labels_reader = csv.reader(labels, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         train_labels = np.asarray(list(labels_reader))
#
#         train_labels = np.asarray(train_labels)
#         print(np.shape(train_labels))
#         # (50001, 2), we only need the second column, because the first one is simply an index
#
#         # TODO Implement a training algorithm with sklearn
#
#         # TODO Implement a validation technique to choose the hyper parameter properly
#
#         # TODO Implement the testing phase with the test.csv data set
#
#         # knn_classifier = sklearn.neighbors.KNeighborsClassifier()
#         # knn_classifier.fit(train_data[1:], train_labels[])
#
#         # # Cross validation for SVM
#         # clf = sklearn.svm.SVC(kernel='linear', C=1, random_state=42)
#         # scores = cross_val_score(clf, X, y, cv=5)
#         # scores
#         # array([0.96..., 1. , 0.96..., 0.96..., 1. ])
