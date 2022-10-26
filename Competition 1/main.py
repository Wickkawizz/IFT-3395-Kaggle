import csv
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# https://dhirajkumarblog.medium.com/logistic-regression-in-python-from-scratch-5b901d72d68e
# https://www.kaggle.com/code/hamzaboulahia/logistic-regression-mnist-classification

# We load the training set, the training labels and the test set
data = pd.read_csv('train.csv')
data = data.to_numpy()[:, :1568]
labels = pd.read_csv('train_result.csv')
labels = labels.to_numpy()
test = pd.read_csv('test.csv')
test = test.to_numpy()[:, :1568]
# print(data.head(10))
# print(labels.head(10))
# print(test.head(10))
print(data)
print(np.shape(data))
print(labels)
print(test)


# Function that normalizes the data
def normalize(data):
    print(np.shape(data))
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_normalized = (data - mean) / std
    print(np.shape(data_normalized))
    return data_normalized


# We normalize the data so it is easier to handle
# mnist_data_normalized_training = normalize(data)
mnist_data_normalized_training = data
# mnist_data_normalized_test = normalize(test)
mnist_data_normalized_test = test
labels = np.array(labels)[:, 1].reshape(labels.shape[0], 1)

# One vs all method, consists of making a vector for each class we have in our dataset (0-18, 19 classes)
Y_train_0 = (labels == 0).astype(int)
Y_train_1 = (labels == 1).astype(int)
Y_train_2 = (labels == 2).astype(int)
Y_train_3 = (labels == 3).astype(int)
Y_train_4 = (labels == 4).astype(int)
Y_train_5 = (labels == 5).astype(int)
Y_train_6 = (labels == 6).astype(int)
Y_train_7 = (labels == 7).astype(int)
Y_train_8 = (labels == 8).astype(int)
Y_train_9 = (labels == 9).astype(int)
Y_train_10 = (labels == 10).astype(int)
Y_train_11 = (labels == 11).astype(int)
Y_train_12 = (labels == 12).astype(int)
Y_train_13 = (labels == 13).astype(int)
Y_train_14 = (labels == 14).astype(int)
Y_train_15 = (labels == 15).astype(int)
Y_train_16 = (labels == 16).astype(int)
Y_train_17 = (labels == 17).astype(int)
Y_train_18 = (labels == 18).astype(int)


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
    sys.stdout.write("[%s]" % ("[" * toolbar_width))
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


# Takes the list of actual labels and its predictions
def accuracy_score(Y_train, Yhat_train):
    good_preds = 0
    # Counting the good predictions made by the model
    for i in range(len(Y_train)):
        if Yhat_train[i] == Y_train[i]:
            good_preds += 1
    # Returning the percentage of good predictions
    return (good_preds / len(Y_train)) * 100


# Creating the model function which trains a model and return its parameters.
def LogRegModel(X_train, X_test, Y_train, alpha, max_iter):
    # TODO Analyze this line, I changed it from shape[1] and it gives something odd
    print(np.shape(X_train))
    nbr_features = np.shape(X_train)[1]
    W, B = initializer(nbr_features)
    cost_history, W, B, i = gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    Yhat_train, _ = predict(X_train, W, B)
    Yhat, _ = predict(X_test, W, B)

    train_accuracy = accuracy_score(Y_train, Yhat_train)
    # test_accuracy = accuracy_score(Y_test, Yhat)
    # conf_matrix = confusion_matrix(Y_test, Yhat, normalize='true')

    model = {"weights": W,
             "bias": B,
             "train_accuracy": train_accuracy,
             # "test_accuracy": test_accuracy,
             # "confusion_matrix": conf_matrix,
             "cost_history": cost_history}
    return model


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def conf_matrix(testlabels, predlabels):
    n_classes = int(max(testlabels))
    matrix = np.zeros((n_classes, n_classes))

    for (test, pred) in zip(testlabels, predlabels):
        # ---> Write code here
        matrix[int(test) - 1, int(pred) - 1] += 1
    return matrix


# This is for the model_0. Only to test out if the code works
# print('Progress bar: 1 step each 50 iteration')
# model_0 = LogRegModel(mnist_data_normalized_training, mnist_data_normalized_test, Y_train_0, alpha=0.01, max_iter=500)
# print('Training completed!')
#
# cost = np.concatenate(model_0['cost_history']).ravel().tolist()
# plt.plot(list(range(len(cost))), cost)
# plt.title('Evolution of the cost by iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.show()
#
# print('The training accuracy of the model', model_0['train_accuracy'])

models_list = []
models_name_list = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6',
                    'model_7', 'model_8', 'model_9', 'model_10', 'model_11', 'model_12', 'model_13', 'model_14',
                    'model_15', 'model_16',
                    'model_17', 'model_18']
Y_train_list = [Y_train_0, Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5, Y_train_6,
                Y_train_7, Y_train_8, Y_train_9, Y_train_10, Y_train_11, Y_train_12, Y_train_13, Y_train_14, Y_train_15,
                Y_train_16, Y_train_17, Y_train_18]

# We can't do the test_list, because we don't have the labels for the tests
# Y_test_list = [Y_test_0, Y_test_1, Y_test_2, Y_test_3, Y_test_4, Y_test_5, Y_test_6, Y_test_7,
#               Y_test_8, Y_test_9]
print('Training of a classifier for each digit:')
for i in range(18):
    print('Training of the model: ', models_name_list[i], ', to recognize the digit: ', i)
    print('Training progress bar: 1 step each 50 iteration')
    model = LogRegModel(mnist_data_normalized_training, mnist_data_normalized_test, Y_train_list[i], alpha=0.01,
                        max_iter=1000)
    print('Training completed!')
    print('Accuracy:', model['train_accuracy'])
    print('-' * 60)
    models_list.append(model)

accuracy_list = []
for i in range(len(models_list)):
    accuracy_list.append(models_list[i]['train_accuracy'])
ove_vs_all_accuracy = np.mean(accuracy_list)
print('The accuracy of the One-Vs-All model is:', ove_vs_all_accuracy)


def one_vs_all(data, models_list):
    pred_matrix = np.zeros((data.shape[0], 19))
    for i in range(len(models_list)):
        W = models_list[i]['weights']
        B = models_list[i]['bias']
        Yhat, Yhat_prob = predict(data, W, B)
        pred_matrix[:, i] = Yhat_prob.T
    max_prob_vec = np.amax(pred_matrix, axis=1, keepdims=True)
    pred_matrix_max_prob = (pred_matrix == max_prob_vec).astype(int)
    labels = []
    for j in range(pred_matrix_max_prob.shape[0]):
        idx = np.where(pred_matrix_max_prob[j, :] == 1)
        labels.append(idx)
    labels = np.vstack(labels).flatten()
    return labels


pred_label = one_vs_all(test, models_list)
#index_list = [range(50000)]
df = pd.DataFrame(pred_label, columns=['Index', 'Class'])
#df['Class'] = pred_label
df.to_csv('test_result.csv', encoding='utf-8')

# We don't have the test labels, so we can't make the confusion matrix
# conf_matrix = conf_matrix(Y_test, pred_label)
