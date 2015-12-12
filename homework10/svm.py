from __future__ import division
import numpy as np
from sklearn.utils import check_X_y, check_array
import sklearn.metrics
from scipy import stats
import random

class SVC(object):
    """Implements a binary classifier SVM using SGD.

    Parameters
    ----------
    reg : float
        The regularization constant

    Attributes
    ----------
    weights_ : [n_features]
        Vector representing the weights of the SVM

    bias_ : float
        Float representing the bias term of the SVM

    n_features_ : int
        The number of features in the dataset.

    n_classes_ : int
        The number of classes in the dataset.
    """

    def __init__(self, reg=1):
        self.reg = reg

        # It's just my habit to suffix variables with '_' if they're
        # instantiated in the "fitting" stage of the algorithm.
        # This is how the scikit-learn maintainers style their code.
        self.weights_ = None
        self.bias_ = 0
        self.n_features_ = None
        self.n_classes_ = None

    def fit(self, X, y, validation_size=0.25, n_epochs=50, n_steps=100, a_const = 1, b_const = 1):
        """Trains the support vector machine
        """
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]

        # Need to make sure y labels are correct
        self.n_classes_ = np.unique(y)
        assert len(self.n_classes_) == 2

        # Converts the binary labels of y into -1 and 1.
        new_y = np.zeros(len(y))
        new_y[y == self.n_classes_[0]] = -1
        new_y[y == self.n_classes_[1]] = 1
        y = new_y

        # Shuffle the training set
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]

        # Create the validation set
        val_ind = len(X) * validation_size
        val_X = X[:val_ind]
        val_y = y[:val_ind]
        X = X[val_ind:]
        y = y[val_ind:]

        # Initialize weights to between [-1, 1)
        self.weights_ = np.random.random(self.n_features_) * 2 - 1
        self.bias_ = random.random()*2-1

        # TODO: Write the gradient step update for the SVM
        # It might be helpful to return the prediction accuracy of
        # the SVM on some evaluation set.
        #
        # See sklearn.metrics.accuracy_score

        accuracies = []

        for epoch in range(1, n_epochs):
            learning_rate = 1 / (a_const * epoch + b_const)
            sample_set = random.sample(list(zip(X, y)), 50)

            for step in range(1, n_steps):
                s_i = random.choice(sample_set)
                x_i = s_i[0]
                y_i = s_i[1]

                f_x_i = self.weights_.dot(x_i.T) + self.bias_
                gradient = (self.reg * self.weights_, 0)\
                    if (y_i * f_x_i) >= 1\
                    else (self.reg * self.weights_ - y_i * x_i, -y_i)

                self.weights_ -= learning_rate * gradient[0]
                self.bias_ -= learning_rate * gradient[1]

                if (step % 10) == 0:
                    accuracies.append(sklearn.metrics.accuracy_score(val_y, self.predict(val_X)))

        return accuracies

    def predict(self, X):
        """Returns the predictions (-1 or 1) on the feature set X.

        """
        X = check_array(X)
        assert X.shape[1] == self.n_features_

        # TODO: You should apply some thresholding here
        predictions = self.weights_.dot(X.T) + self.bias_
        predictions = stats.threshold(predictions, threshmax=0, newval=1)
        predictions = stats.threshold(predictions, threshmin=0, newval=-1)
        return predictions
