# article = https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217


# Decorators
import numpy as np
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from functools import wraps

def my_logger(orig_func):
    import logging
    # TODO what does basicConfig do?
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper


def download():
    # mnist = fetch_mldata('MNIST original')
    mnist = load_digits()
    X = mnist.data.astype('float64')
    y = mnist.target
    return (X, y)


class Normalize(object):
    def normalize(self, X_train, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        print(X_test.shape)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_test)

    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_test)


def split(X, y, splitRatio):


    X_train = X[:splitRatio]
    y_train = y[:splitRatio]
    X_test = X[splitRatio:]
    y_test = y[splitRatio:]
    return (X_train, y_train, X_test, y_test)


class TheAlgorithm(object):

    @my_logger
    @my_timer
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    @my_logger
    @my_timer
    def fit(self):
        normalizer = Normalize()
        self.X_train, self.X_test = normalizer.normalize(self.X_train,
                                                         self.X_test)
        train_samples = self.X_train.shape[0]
        self.classifier = LogisticRegression(
            C=50. / train_samples,
            multi_class='multinomial',
            penalty='l1',
            solver='saga',
            tol=0.1,
            class_weight='balanced',
        )
        self.classifier.fit(self.X_train, self.y_train)
        self.train_y_predicted = self.classifier.predict(self.X_train)
        self.train_accuracy = np.mean(
            self.train_y_predicted.ravel() == self.y_train.ravel()) * 100
        self.train_confusion_matrix = confusion_matrix(self.y_train,
                                                       self.train_y_predicted)
        return self.train_accuracy

    @my_logger
    @my_timer
    def predict(self):
        self.test_y_predicted = self.classifier.predict(self.X_test)
        self.test_accuracy = np.mean(
            self.test_y_predicted.ravel() == self.y_test.ravel()) * 100
        self.test_confusion_matrix = confusion_matrix(self.y_test,
                                                      self.test_y_predicted)
        self.report = classification_report(self.y_test, self.test_y_predicted)
        print("Classification report for classifier:\n %s\n" % (self.report))
        return self.test_accuracy
# The solution
X, y = download()
print('MNIST:', X.shape, y.shape)

splitRatio = 60
X_train, y_train, X_test, y_test = split(X, y, splitRatio)
print(X_test.shape)

np.random.seed(31337)
ta = TheAlgorithm(X_train, y_train, X_test, y_test)
train_accuracy = ta.fit()
print()
print('Train Accuracy:', train_accuracy, '\n')
print("Train confusion matrix:\n%s\n" % ta.train_confusion_matrix)

test_accuracy = ta.predict()
print()
print('Test Accuracy:', test_accuracy, '\n')
print("Test confusion matrix:\n%s\n" % ta.test_confusion_matrix)








