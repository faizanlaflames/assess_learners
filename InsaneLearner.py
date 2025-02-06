#faizan hussain, ml4t proj 3 2025

import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner:
    def __init__(self, verbose=False):
        # create ensemble of 20 bagged learners
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        # train all learners on same dataset
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        # get predictions from learners
        predictions = np.array([learner.query(points) for learner in self.learners])
        # return average prediction across ensemble
        return np.mean(predictions, axis=0)

    def author(self):
       
        return "fhussain45"  