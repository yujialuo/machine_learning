import numpy as np
import math
from random import randint
import random
from sklearn import preprocessing
import scipy.io as sio
import pickle
import scipy.ndimage

def shift(X, theta = 0.3):
    X_shift = np.zeros(X.shape)
    X = np.matrix(X).reshape((28, 28))
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
    X = np.roll(X, int(displacement), axis = direction)
    # Reshape 
    X_shift = X.reshape((1, 784))
    return X_shift

def rotate(X, theta = 10):
    X_rotated = np.zeros(X.shape)
    X = np.matrix(X).reshape((28, 28))
    # Random rotate degree
    degree = np.random.rand() * theta
    sign = np.random.rand()
    if sign > 0.5:
        degree = -degree
    X = scipy.ndimage.interpolation.rotate(X, degree, reshape = False, mode = 'nearest')
    # Reshape
    X = X.reshape((1, 784))
    X_rotated = X
    return X_rotated

def one_hot(y):
    hot = np.zeros([len(y), 26])
    for i in range(len(y)):
        hot[i][y[i][0] - 1] = 1
    return hot

def preprocess(train_x, train_y, test_x):

    # Normalize
    train_x, test_x = preprocessing.scale(train_x), preprocessing.scale(test_x)

    # Shuffle
    rand = np.random.permutation(len(train_x))
    train_x, train_y = train_x[rand], train_y[rand]

    # Split
    split = 4800
    val_x, val_y = train_x[:split], one_hot(train_y[:split])
    train_x, train_y = train_x[split:], one_hot(train_y[split:])

    # add bias
    # train_x = np.c_[train_x, np.ones(train_x.shape[0]) ]
    val_x = np.c_[val_x, np.ones(val_x.shape[0]) ]
    test_x = np.c_[test_x, np.ones(test_x.shape[0]) ]

    return train_x, train_y, val_x, val_y, test_x

def tanh(z):
    return np.tanh(z)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def soft_loss(z, y, w, lam):
    z = np.reshape(z, (1, 26))[0]
    z[z < 1e-8] = 1e-8
    regularization = lam * np.linalg.norm(w)
    return -1.0 * np.dot(np.log(z), y) + regularization

def save_weights(step, v, w):
    obj = [step, v, w]
    with open("saved/"+str(step)+".p", "wb") as f:
        pickle.dump(obj, f)
    f.close()

def relu(x):
    if x > 0:
        return x
    else:
        return 0
relu = np.vectorize(relu)

def relu_d(x):
    if x > 0:
        return 1
    else:
        return 0
relu_d = np.vectorize(relu_d)


class Graph:
    def __init__(self, n_hidden=400):
        self.n_hidden = n_hidden
        self.V = np.random.normal(loc=0, scale=1.0/math.sqrt(784), size=(self.n_hidden, 785))
        self.W = np.random.normal(loc=0, scale=1.0/math.sqrt(self.n_hidden), size=(26, self.n_hidden+1))
        self.val_accs, self.train_accs, self.losses, self.acc_steps, self.loss_steps = [], [], [], [], []
        self.start = 0

    def train(self, x, y, val_x, val_y, max_steps=100001, lr=1e-2, lam=0.1, stop_step=1000):

        self.lam = lam
        for i in range(self.start, self.start+max_steps):
            # rand = random.sample(range(len(x)), batch_size)
            rand = randint(0,len(x)-1)

            xx = x[rand]
            yy = y[rand]
            xx = shift(xx)
            xx = rotate(xx)
            xx = np.c_[xx, np.ones(xx.shape[0])]
            hh, zz = self._test(xx) # forward pass 26 x 1

            # stochastic gradient descent
            v_grad, w_grad = self.compute_gradients(xx, yy, hh, zz)
            self.V -= (lr * v_grad)
            self.W -= (lr * w_grad)

            # validation
            if i % stop_step == 0 or i == max_steps - 1:
                val_acc = self.evaluate(val_x, val_y)
                train_acc = self.evaluate(np.c_[train_x, np.ones(train_x.shape[0])], train_y)
                loss = soft_loss(zz, yy, self.W, lam)
                print("step %d: train acc = %.3f, validate acc = %.3f, loss = %.3f"%(i, train_acc, val_acc, loss))
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                self.acc_steps.append(i)
                self.losses.append(loss)
                self.loss_steps.append(i)
                save_weights(i, self.V, self.W)



    def _test(self, xx):
        
        xx = np.reshape(xx, (785, 1))
        temp1 = np.matmul(self.V, xx)
        h = tanh(temp1) # 401 x 1
        # h = relu(temp1)
        h = np.append(h, [[1]], axis=0)
        temp2 = np.matmul(self.W, h)
        z = softmax(temp2) # 26 x 1
        return h, z

    def test(self, x):
        x = x.T
        temp1 = np.matmul(self.V, x)
        h = tanh(temp1)
        h = np.append(h, [[1]*len(x[0])], axis=0)
        temp2 = np.matmul(self.W, h)
        z = softmax(temp2).T
        return h, z

    def evaluate(self, x, y):
        correct = 0
        h, z = self.test(x)
        labels = np.argmax(z, axis=1)
        y = np.argmax(y, axis=1)
        correct = np.sum(labels == y)
        return 1.0 * correct / len(y)

    def update_weights(self, step):
        with open("saved/" + str(step) + ".p", "rb") as f:
            obj = pickle.load(f)
        f.close()
        self.V, self.W = obj[1], obj[2]
        self.start = step

    def compute_gradients(self, x, y, h, z):
        y = np.reshape(y, (26, 1))
        w_grad = 1.0 * np.matmul((z - y), h.T)
        w_grad += 2 * self.lam * self.W

        temp = list(np.reshape(np.matmul(self.W.T, (z - y))[:self.n_hidden], (1, self.n_hidden))[0])
        temp2 = list(np.reshape(1 - h[:self.n_hidden] ** 2, (1, self.n_hidden))[0])
        # temp2 = list(np.reshape(relu_d(h[:self.n_hidden]), (1, self.n_hidden))[0])
        product = np.array([a*b for a, b in zip(temp, temp2)])
        v_grad = np.outer(product, x)
        v_grad += 2 * self.lam * self.V

        return v_grad, w_grad


data = sio.loadmat("data/letters_data.mat")
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
train_x, train_y, val_x, val_y, test_x = preprocess(train_x, train_y, test_x)
g = Graph()
g.train(train_x, train_y, val_x, val_y, lr=0.01, max_steps=10000, stop_step=2000)