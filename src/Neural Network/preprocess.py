import numpy as np
import scipy.ndimage
from sklearn import preprocessing

num_classes = 26
valid_size = 4800

def shift(X_train, theta = 0.3):
    X_shift = np.zeros(X_train.shape)
    for i in range(0, X_train.shape[0]):
        Xi = np.matrix(X_train[i]).reshape((28, 28))
        # Random shift displacement
        if (np.random.rand() > 0.5):
        	sign = 1
        else:
        	sign = -1
        displacement = np.floor(np.random.rand() / theta) * sign
        # Random shift direction
        if (np.random.rand() > 0.5):
        	direction = 1
        else:
        	direction = 0
        Xi = np.roll(Xi, int(displacement), axis = direction)
        # Reshape
        Xi = Xi.reshape((1, 784))
        X_shift[i] = Xi
    return X_shift

def rotate(X_train, theta = 10):
    X_rotated = np.zeros(X_train.shape)
    for i in range(0, X_train.shape[0]):
        Xi = np.matrix(X_train[i]).reshape((28, 28))
        # Random rotate degree
        degree = np.random.rand() * theta
        sign = np.random.rand()
        if sign > 0.5:
            degree = -degree
        Xi = scipy.ndimage.interpolation.rotate(Xi, degree, reshape = False, mode = 'nearest')
        # Reshape
        Xi = Xi.reshape((1, 784))
        X_rotated[i] = Xi
    return X_rotated

# def resize(X_train):
#     X_resized = np.zeros((X_train.shape[0], 4 * X_train.shape[1]))
#     for i in range(0, X_train.shape[0]):
#         X_i = np.matrix(X_train[i]).reshape((28, 28))
#         big_X_i = scipy.ndimage.zoom(X_i, 2, order=1)
#         o_X_i = big_X_i.reshape((1, 3136))
#         X_resized[i] = o_X_i
#     return X_resized

def one_hot(Y):
    labels = np.zeros([Y.size, num_classes])
    for i in range(0, Y.size):
        labels[i][Y[i][0]] = 1
    return labels

def random_shuffle(X, Y):
    assert len(X) == len(Y)
    p = numpy.random.permutation(len(X))
    return X[p], Y[p]

def preprocess(X_train, Y, X_test):
	X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
	X_val, Y_val = X_train[:valid_size], Y[:valid_size]
	X_train, Y_train = X_train[valid_size:], Y[valid_size:]
	return X_train, one_hot(Y_train), X_val, one_hot(Y_val), X_test
