#faizan hussain, ml4t proj 3 2025

import numpy as np
import random

class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        # initialize learner with leaf size, verbosity, and empty tree
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def add_evidence(self, data_x, data_y):
        # build model from training data
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # leaf node if data <= leaf size or all y same
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # randomly pick feature, use median as split value
        best_feature = random.randint(0, data_x.shape[1] - 1)
        split_val = np.median(data_x[:, best_feature])

        # create left/right splits
        left_mask = data_x[:, best_feature] <= split_val
        right_mask = data_x[:, best_feature] > split_val

        # if split fails, return leaf
        if np.all(left_mask) or np.all(right_mask):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # recursively build subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        # combine root + subtrees
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        # make predictions for all input points
        predictions = np.array([self.predict(self.tree, point) for point in points])
        return predictions

    def predict(self, tree, point):
        # traverse tree to predict single point
        node = tree[0]
        feature_idx = int(node[0])
        if feature_idx == -1:  # leaf node, return value
            return node[1]
        split_val = node[1]
        if point[feature_idx] <= split_val:  # follow left branch
            return self.predict(tree[1:], point)
        else:  # follow right branch
            return self.predict(tree[int(node[3]):], point)

    def author(self):
      
        return "fhussain45" 

    def study_group(self):
        
        return "fhussain45" 
