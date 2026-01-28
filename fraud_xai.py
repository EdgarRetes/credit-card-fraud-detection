import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
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

    def best_split(self, X, y):
        best_feature, best_thresh = None, None
        best_gain = 0
        parent_gini = self.gini(y)

        n, m = X.shape
        for feature in range(m):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = parent_gini - (
                    len(left) / n * self.gini(left)
                    + len(right) / n * self.gini(right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = t

        return best_feature, best_thresh

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
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

        self.left = DecisionTree(self.max_depth)
        self.right = DecisionTree(self.max_depth)

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
            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        preds = [tree.predict(x) for tree in self.trees]
        return np.mean(preds)


class GradientBoosting:
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=3):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        pred = np.full(len(y), np.mean(y))

        for _ in range(self.n_trees):
            residuals = y - pred
            tree = DecisionTree(self.max_depth)
            tree.fit(X, residuals)

            for i in range(len(y)):
                pred[i] += self.learning_rate * tree.predict(X[i])

            self.trees.append(tree)

    def predict(self, x):
        pred = 0.0
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(x)
        return pred


def ensemble_predict(rf, gb, x):
    p_rf = rf.predict(x)
    p_gb = gb.predict(x)
    return 0.5 * (p_rf + p_gb)


def shap_tree(tree, x, baseline=0.0):
    phi = {}

    def traverse(node, current_value):
        if node.value is not None:
            return node.value

        feature = node.feature
        if feature not in phi:
            phi[feature] = 0.0

        if x[feature] <= node.threshold:
            next_value = node.left.predict(x)
            phi[feature] += next_value - current_value
            return traverse(node.left, next_value)
        else:
            next_value = node.right.predict(x)
            phi[feature] += next_value - current_value
            return traverse(node.right, next_value)

    traverse(tree, baseline)
    return phi


def shap_ensemble(rf, gb, x):
    phi_rf = {}
    for tree in rf.trees:
        phi_t = shap_tree(tree, x)
        for k, v in phi_t.items():
            phi_rf[k] = phi_rf.get(k, 0.0) + v / len(rf.trees)

    phi_gb = {}
    for tree in gb.trees:
        phi_t = shap_tree(tree, x)
        for k, v in phi_t.items():
            phi_gb[k] = phi_gb.get(k, 0.0) + v / len(gb.trees)

    phi = {}
    for k in set(phi_rf) | set(phi_gb):
        phi[k] = 0.5 * (phi_rf.get(k, 0.0) + phi_gb.get(k, 0.0))

    return phi
