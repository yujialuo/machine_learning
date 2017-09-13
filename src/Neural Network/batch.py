import numpy as np
import math
from random import randint
import random
from sklearn import preprocessing
import scipy.io as sio
import pickle
import scipy

def one_hot(y):
    hot = np.zeros([len(y), 26])
    for i in range(len(y)):
        hot[i][y[i][0] - 1] = 1
    return hot

def shift(X, theta = 0.3):
    X_shift = np.zeros(X.shape)
    X = np.matrix(X).reshape((28, 28))
    # Random shift displacement
    sign = -1
    if (np.random.rand() > 0.5):
        sign = 1
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

def preprocess(train_x, train_y, test_x):

    # Normalize
    train_x, test_x = preprocessing.scale(train_x), preprocessing.scale(test_x)

    # Shuffle
    rand = np.random.permutation(len(train_x))
    train_x, train_y = train_x[rand], train_y[rand]

    # Split
    split = int(0.2 * len(train_x))
    # split = 4800
    val_x, val_y = train_x[:split], one_hot(train_y[:split])
    train_x, train_y = train_x[split:], one_hot(train_y[split:])

    # add bias
    train_x = np.c_[train_x, np.ones(train_x.shape[0]) ]
    val_x = np.c_[val_x, np.ones(val_x.shape[0]) ]
    test_x = np.c_[test_x, np.ones(test_x.shape[0]) ]

    return train_x, train_y, val_x, val_y, test_x

def tanh(z):
    h = 1.0 * (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return h

def softmax(z):
    exp = np.exp(z.T)
    sum = np.reshape(np.sum(exp, axis=1), (len(exp), 1))
    result = 1.0 * exp / sum
    return result.T

def soft_loss(z, y, lam, w):
    # z = np.reshape(z, (1, 26))[0]
    z[z == 0] = 1e-8
    loss = np.matmul(np.log(z), y.T)
    loss = loss.diagonal()
    regularization = lam * np.linalg.norm(w)
    return -1.0 * loss + regularization

def save_weights(step, v, w):
    obj = [step, v, w]
    with open("saved/"+str(step)+".p", "wb") as f:
        pickle.dump(obj, f)
    f.close()


class Graph:
    def __init__(self, n_hidden=400):
        self.V = np.random.normal(loc=0, scale=1.0/math.sqrt(784), size=(n_hidden, 785))
        self.W = np.random.normal(loc=0, scale=1.0/math.sqrt(n_hidden), size=(26, n_hidden+1))
        self.val_accs, self.train_accs, self.losses, self.acc_steps, self.loss_steps = [], [], [], [], []
        self.start = 0
        self.n_hidden = n_hidden

    def train(self, x, y, val_x, val_y, batch_size=50, max_steps=100000, lr=1e-4, lam=0.007, stop_step=1000):
        self.batch_size = batch_size
        self.lam = lam
        for i in range(self.start, self.start+max_steps):
            rand = random.sample(range(len(x)), batch_size)
            # rand = randint(0,len(x)-1)
            x = x[rand]
            y = y[rand]
            # xx = shift(xx)
            # xx = rotate(xx)
            # xx = np.c_[xx, np.ones(xx.shape[0])]

            h, z = self.test(x) # forward pass 26 x 1
            loss = soft_loss(z, y, lam, self.W)
            self.losses.append(np.mean(loss))
            self.loss_steps.append(i)

            # stochastic gradient descent
            v_grad, w_grad = self.compute_gradients(x, y, h, z)
            self.V -= lr * v_grad
            self.W -= lr * w_grad

            # validation
            if i % stop_step == 0 or i == max_steps - 1:
                val_acc = self.evaluate(val_x, val_y)
                train_acc = self.evaluate(train_x, train_y)
                print("step {}: train acc = {}, validate acc = {}".format(i, train_acc, val_acc))
                # save_weights(i, self.V, self.W)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                self.acc_steps.append(i)

    def _test(self, xx):
        xx = np.reshape(xx, (len(xx), 1))
        temp1 = np.matmul(self.V, xx)
        h = tanh(temp1) # 201 x 50
        h = np.append(h, [[1]], axis=0)
        temp2 = np.matmul(self.W, h)
        z = softmax(temp2) # 26 x 50
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

    # def compute_stochastic_gradients(self, x, y, h, z):
    #     y = np.reshape(y, (26, 1))
    #     w_grad = 1.0 * np.matmul((z - y), h.T)
    #     temp = list(np.reshape(np.matmul(self.W.T, (z - y))[:self.n_hidden], (1, self.n_hidden))[0])
    #     temp2 = list(np.reshape(1 - h[:self.n_hidden] ** 2, (1, self.n_hidden))[0])
    #     product = np.array([a*b for a, b in zip(temp, temp2)])
    #     v_grad = np.outer(product, x)
    #     return v_grad, w_grad

    def compute_gradients(self, x, y, h, z):
        # W
        w_grad = np.matmul((z - y).T, h.T)
        w_grad + 2 * self.lam * self.W

        # V
        temp = np.matmul(self.W.T, (z - y).T)[:self.n_hidden]
        temp2 = 1 - h[:self.n_hidden] ** 2
        product = temp * temp2
        v_grad = np.matmul(product, x)
        v_grad + 2 * self.lam * self.V

        return v_grad, w_grad


data = sio.loadmat("data/letters_data.mat")
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
train_x, train_y, val_x, val_y, test_x = preprocess(train_x, train_y, test_x)
g = Graph()
g.train(train_x, train_y, val_x, val_y)