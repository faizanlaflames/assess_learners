#faizan hussain, ml4t proj 3 2025

import numpy as np

class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        # initialize decision tree learner parameters
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def add_evidence(self, data_x, data_y):
        # build decision tree from training data
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # create leaf node if small dataset or uniform y
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # select feature with highest absolute correlation
        best_feature = self.get_best_feature(data_x, data_y)
        split_val = np.median(data_x[:, best_feature])

        # split data into left/right branches
        left_mask = data_x[:, best_feature] <= split_val
        right_mask = data_x[:, best_feature] > split_val

        # handle failed split by returning leaf
        if np.all(left_mask) or np.all(right_mask):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # recursively build left/right subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        # combine root node with subtrees
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def get_best_feature(self, data_x, data_y):
        correlations = np.array([np.abs(np.corrcoef(data_x[:, i], data_y)[0, 1]) for i in range(data_x.shape[1])])
        return np.argmax(correlations)

    def query(self, points):
        # generate predictions for all query points
        predictions = np.array([self.predict(self.tree, point) for point in points])
        return predictions

    def predict(self, tree, point):
        # traverse tree to make prediction
        node = tree[0]
        feature_idx = int(node[0])
        if feature_idx == -1:  # leaf node
            return node[1]
        if point[feature_idx] <= node[1]:  # follow left branch
            return self.predict(tree[1:], point)
        else:  # follow right branch
            return self.predict(tree[int(node[3]):], point)

    def author(self): 

        return "fhussain45"

    def study_group(self):
        return "fhussain45"  