from DecisionTree import *
from random import randrange


class RandomForest:
    def __init__(self, max_depth=None, num_trees=None, sample_size=None, features=None, n=3):
        self.trees = []
        self.sample_size = sample_size
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.features = features
        self.n = n

    def train(self, x, y):
        for i in range(self.num_trees):
            sub_x, sub_y = self.sample(x, y)
            tree = DecisionTree(maxDepth=self.max_depth, n=self.n, features=self.features)
            tree.train(tree.root, sub_x, sub_y)
            self.trees.append(tree)

    def predict(self, x):
        y_hat = []
        for sample in x:  # For each point
            logits = []
            for tree in self.trees:  # For each decision tree
                logits.append(tree._predict(sample))
            best = max(set(logits), key=logits.count)
            y_hat.append(best)
        return y_hat

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        accuracy = np.mean(np.equal(y, y_hat).astype(int))
        return accuracy

    def sample(self, x, y):
        sub_x, sub_y = [], []
        while len(sub_x) < self.sample_size:
            i = randrange(len(x))
            sub_x.append(x[i])
            sub_y.append(y[i])
        return sub_x, sub_y
