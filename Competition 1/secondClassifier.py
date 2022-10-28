# https://www.kaggle.com/code/recepinanc/mnist-classification-sklearn/notebook#1.-Introduction
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/#:~:text=MNIST%20Handwritten%20Digit%20Classification%20Dataset,-The%20MNIST%20dataset&text=It%20is%20a%20dataset%20of,from%200%20to%209%2C%20inclusively.
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
# https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# We load the training set, the training labels and the test set
X_training = pd.read_csv('train.csv')
X_training = X_training.to_numpy()[:, :1568]
Y_training = pd.read_csv('train_result.csv')
Y_training = Y_training.to_numpy()[:, 1]
print(np.shape(Y_training))
X_test = pd.read_csv('test.csv')
X_test = X_test.to_numpy()[:, :1568]

# Multi-Level perceptron for the classifier
mlp = MLPClassifier(max_iter=500)

# Defining a parameter space to search for the best parameters to use
parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive']
    #'max_iter': [100, 200, 500]
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_training, Y_training)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_pred = clf.predict(X_test)
print(y_pred)
# TODO Fix this
# ValueError: Shape of passed values is (10000, 1), indices imply (10000, 2)
df = pd.DataFrame(y_pred, columns=['Index', 'Class'], index=True)
df.to_csv('test_result_mlp.csv', encoding='utf-8')
