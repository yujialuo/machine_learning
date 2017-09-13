import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pickle

class Node:
    def __init__(self, depth):
        self.left = None
        self.right = None
        self.threshold = None
        self.feature_index = None
        self.depth = depth

        self.isLeaf = False
        self.label = None

        # jecky
        self.mu_0 = None
        self.sigma_0 = None
        self.mu_1 = None
        self.sigma_1 = None
        self.log_prior_0 = None
        self.log_prior_1 = None
        self.use_qda = False

class DecisionTree:
    def __init__(self, maxDepth=None, n=3, features=None, minSize=10, qda_en=True):
        self.root = Node(0)
        self.nodes = []
        self.maxDepth = maxDepth
        self.n_thres = n
        self.features = features

        # Jecky
        self.minSize = minSize
        self.qda_en = qda_en

    def train(self, curr, x, y):
        if not curr:
            pass
        elif (not self.entropy(y)) or (curr.depth == self.maxDepth) or (len(x) <= self.minSize):  # Leaf
            counts = np.bincount(y)
            curr.label = np.argmax(counts)
            curr.isLeaf = True
        else:
            curr.feature_index, curr.threshold = self.segmentor(x, y)
            leftX, leftY, rightX, rightY = self.split(curr.threshold, curr.feature_index, x, y)

            # Jecky
            linear_entro = self.impurity(leftY, rightY)
            if self.qda_en and self.entropy(y) > 0.1 and curr.depth < 2:
                curr.mu_0, curr.sigma_0, curr.mu_1, curr.sigma_1, curr.log_prior_0, curr.log_prior_1 = self.find_qda(x,
                                                                                                                     y)
                leftX_Q, leftY_Q, rightX_Q, rightY_Q = self.qda_split(curr.mu_0, curr.sigma_0, curr.mu_1, curr.sigma_1,
                                                                      curr.log_prior_0, curr.log_prior_1, x, y)
                if self.impurity(leftY_Q, rightY_Q) < linear_entro:
                    curr.use_qda = True
                    leftX, leftY, rightX, rightY = leftX_Q, leftY_Q, rightX_Q, rightY_Q

            if leftY:
                curr.left = Node(curr.depth + 1)
                self.train(curr.left, leftX, leftY)
            if rightY:
                curr.right = Node(curr.depth + 1)
                self.train(curr.right, rightX, rightY)

    def predict(self, x):
        y_hat = []
        for sample in x:
            y_hat.append(self._predict(sample))
        return y_hat

    def _predict(self, x):  # x is one sample
        curr = self.root
        while not curr.isLeaf:
            # Jecky
            if curr.use_qda and len(x) == len(curr.mu_0):
                prob0 = multivariate_normal.logpdf(x, curr.mu_0,
                                                   curr.sigma_0 + 0.001 * np.identity(curr.sigma_0.shape[0]),
                                                   allow_singular=True) + curr.log_prior_0
                prob1 = multivariate_normal.logpdf(x, curr.mu_1,
                                                   curr.sigma_1 + 0.001 * np.identity(curr.sigma_1.shape[0]),
                                                   allow_singular=True) + curr.log_prior_1
                if prob0 > prob1:
                    curr = curr.left
                else:
                    curr = curr.right
            else:
                if x[curr.feature_index] < curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
        return curr.label

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        accuracy = np.mean(np.equal(y, y_hat).astype(int))
        return accuracy

    def entropy(self, s):
        _, counts = np.unique(s, return_counts=True)
        entro = 1.0 * counts / len(s) * np.log(counts / len(s))
        return -np.sum(entro)

    def segmentor(self, x, y):
        # Find best (feature, threshold) pair
        x_reshape = np.transpose(x)
        impurities = []
        for f in range(len(x[0])):  # Find best threshold for each feature
            thresholds = np.linspace(int(min(x_reshape[f])), int(max(x_reshape[f])), self.n_thres)
            temp = {}
            for thr in thresholds:
                leftX, leftY, rightX, rightY = self.split(thr, f, x, y)
                imp = self.impurity(leftY, rightY)
                temp[thr] = imp
            min_thr = min(temp, key=temp.get)
            impurities.append([min_thr, temp[min_thr]])  # Threshold, Impurity
        f_i = np.argmin([a[1] for a in impurities])
        return f_i, impurities[f_i][0]

    def split(self, threshold, i, x, y):
        leftX, leftY, rightX, rightY = [], [], [], []
        for j in range(len(x)):
            if x[j][i] < threshold:
                leftX.append(x[j])
                leftY.append(y[j])
            else:
                rightX.append(x[j])
                rightY.append(y[j])
        return leftX, leftY, rightX, rightY

    def impurity(self, leftY, rightY):
        lenL, lenR = len(leftY), len(rightY)
        h_after = 1.0 * (self.entropy(leftY) * lenL + self.entropy(rightY) * lenR) / (lenL + lenR)
        return h_after

    def visualize(self):
        # Feature name, split rule, class
        def _print(node, num):
            if node:
                if not node.isLeaf:
                    print('level ', str(num), '-', 'feature:', self.features[node.feature_index],
                          ';', 'split:', node.threshold)
                else:
                    print('level ', str(num), '-', 'class:', str(node.label))

        _print(self.root, 1)
        _print(self.root.left, 2);
        _print(self.root.right, 2)
        _print(self.root.left.left, 3);
        _print(self.root.left.right, 3)
        _print(self.root.right.left, 3);
        _print(self.root.right.right, 3)

    # Jecky
    def find_qda(self, x, y):
        log_prior_0 = math.log((np.count_nonzero(y == 0.0) + 1) / len(y))
        log_prior_1 = math.log((np.count_nonzero(y == 1.0) + 1) / len(y))
        x_0 = []
        y_0 = []
        x_1 = []
        y_1 = []
        for i in range(len(x)):
            if y[i] == 0:
                x_0.append(x[i])
                y_0.append(y[i])
            else:
                x_1.append(x[i])
                y_1.append(y[i])
        return np.mean(x_0, axis=0), np.cov(x_0, rowvar=False), np.mean(x_1, axis=0), np.cov(x_1,
                                                                                             rowvar=False), log_prior_0, log_prior_1

    def qda_split(self, mu_0, sigma_0, mu_1, sigma_1, log_prior_0, log_prior_1, x, y):
        leftX, leftY, rightX, rightY = [], [], [], []
        for i in range(len(x)):
            prob0 = multivariate_normal.logpdf(x[i], mu_0, sigma_0 + 0.001 * np.identity(sigma_0.shape[0]),
                                               allow_singular=True) + log_prior_0
            prob1 = multivariate_normal.logpdf(x[i], mu_1, sigma_1 + 0.001 * np.identity(sigma_1.shape[0]),
                                               allow_singular=True) + log_prior_1
            if prob0 - prob1 > 0:
                leftX.append(x[i])
                leftY.append(y[i])
            else:
                rightX.append(x[i])
                rightY.append(y[i])
        return leftX, leftY, rightX, rightY


df = pd.read_csv('census/census_clean.csv', sep=',')
df.reindex(np.random.permutation(df.index))
y = np.array(df.label)
del df['label']
df = df.drop(df.columns[0], axis=1)
x = np.array(df)
columns = df.columns.tolist()

n = int(0.8 * len(x))
train_x = x[n:]
train_y = y[n:]
val_x = x[:n]
val_y = y[:n]

accs = []
depths = []
for i in range(2, 41):
    print("depth: {}".format(i))
    dt = DecisionTree(maxDepth=i, n=10)
    dt.train(dt.root, train_x, train_y)
    acc = dt.evaluate(val_x, val_y)
    print("depth: {}, accuracy: {}".format(i, acc))
    accs.append(acc)

with open('census_accs.p', 'wb') as handle:
    pickle.dump(accs, handle, protocol=pickle.HIGHEST_PROTOCOL)


fig, ax = plt.subplots()

ax.scatter(depths, accs, color="red")
ax.plot(depths, accs, color="red")

ax.set_xlabel('Max Depth')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Census Decision Tree Max Depth Evaluation')

plt.ylim((0.25, 0.9))
plt.xlim((1, 41))
fig.savefig("plot.png")