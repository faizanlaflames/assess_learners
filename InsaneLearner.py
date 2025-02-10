import numpy as np, BagLearner as bl, LinRegLearner as lrl

class InsaneLearner:
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
       
        return np.mean(predictions, axis=0)

    def author(self):
        return "fhussain45"  