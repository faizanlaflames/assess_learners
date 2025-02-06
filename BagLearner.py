#faizan hussain, ml4t proj 3 2025

import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        # create ensemble of bagged learners
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        # train each learner with bootstrapped sample
        for learner in self.learners:
            # create bootstrap sample indices
            indices = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])

    def query(self, points):
        # collect predictions from all learners
        predictions = np.array([learner.query(points) for learner in self.learners])
        # return average prediction across ensemble
        return np.mean(predictions, axis=0)

    def author(self):
      
        return "fhussain45"

    def study_group(self):
        
        return "fhussain45" 