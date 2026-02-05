import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, task="classification"):
        self.max_depth = max_depth
        self.task = task
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def gini(self, y):
        counts = Counter(y)
        impurity = 1.0
        for c in counts.values():
            p = c / len(y)
            impurity -= p ** 2
        return impurity

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def best_split(self, X, y):
        best_feature, best_thresh = None, None
        best_gain = -1e9

        if self.task == "classification":
            parent_error = self.gini(y)
        else:
            parent_error = self.mse(y)

        n, m = X.shape
        for feature in range(m):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                if self.task == "classification":
                    error = (
                        len(left) / n * self.gini(left)
                        + len(right) / n * self.gini(right)
                    )
                else:
                    error = (
                        len(left) / n * self.mse(left)
                        + len(right) / n * self.mse(right)
                    )

                gain = parent_error - error

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = t

        return best_feature, best_thresh

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(y) == 0:
            self.value = np.mean(y)
            return

        feature, thresh = self.best_split(X, y)
        if feature is None:
            self.value = np.mean(y)
            return

        self.feature = feature
        self.threshold = thresh

        idx_left = X[:, feature] <= thresh
        idx_right = ~idx_left

        self.left = DecisionTree(self.max_depth, self.task)
        self.right = DecisionTree(self.max_depth, self.task)

        self.left.fit(X[idx_left], y[idx_left], depth + 1)
        self.right.fit(X[idx_right], y[idx_right], depth + 1)

    def predict(self, x):
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)
            tree = DecisionTree(self.max_depth, task="classification")
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.trees])

class GradientBoosting:
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=3):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_value = 0.0

    def fit(self, X, y):
        self.base_value = np.mean(y)
        pred = np.full(len(y), self.base_value)

        for _ in range(self.n_trees):
            residuals = y - pred
            tree = DecisionTree(self.max_depth, task="regression")
            tree.fit(X, residuals)

            for i in range(len(y)):
                pred[i] += self.learning_rate * tree.predict(X[i])

            self.trees.append(tree)

    def predict(self, x):
        pred = self.base_value
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(x)
        return pred

def ensemble_predict(rf, gb, x):
    return 0.5 * (rf.predict(x) + gb.predict(x))

def shap_tree(tree, x, baseline):
    phi = {}

    def expected_value(node):
        if node.value is not None:
            return node.value
        return 0.5 * (expected_value(node.left) + expected_value(node.right))

    def walk(node, current):
        if node.value is not None:
            return node.value

        f = node.feature
        if f not in phi:
            phi[f] = 0.0

        E_without = 0.5 * (expected_value(node.left) + expected_value(node.right))

        if x[f] <= node.threshold:
            E_with = expected_value(node.left)
            next_node = node.left
        else:
            E_with = expected_value(node.right)
            next_node = node.right

        phi[f] += E_with - E_without
        return walk(next_node, E_with)

    walk(tree, expected_value(tree))
    return phi

def shap_ensemble(rf, gb, x):
    phi_rf = {}
    for tree in rf.trees:
        phi_t = shap_tree(tree, x, baseline=0.0)
        for k, v in phi_t.items():
            phi_rf[k] = phi_rf.get(k, 0.0) + v / len(rf.trees)

    phi_gb = {}
    for tree in gb.trees:
        phi_t = shap_tree(tree, x, baseline=0.0)
        for k, v in phi_t.items():
            phi_gb[k] = phi_gb.get(k, 0.0) + gb.learning_rate * v / len(gb.trees)

    phi = {}
    for k in set(phi_rf) | set(phi_gb):
        phi[k] = 0.5 * (phi_rf.get(k, 0.0) + phi_gb.get(k, 0.0))

    return phi
