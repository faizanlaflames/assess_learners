
## Assess Learners: ML Trading System


```python
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

# DTLearner
learner = dt.DTLearner(leaf_size=1, verbose=False)
learner.add_evidence(Xtrain, Ytrain)
Y_pred = learner.query(Xtest)

# RTLearner 
learner = rt.RTLearner(leaf_size=1, verbose=False)

# BagLearner
learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=20)

# InsaneLearner
learner = it.InsaneLearner(verbose=False)
```


Experiments to run:
1. Analyze overfitting in DTLearner by varying leaf_size
2. Evaluate if bagging reduces overfitting 
3. Compare DTLearner vs RTLearner quantitatively

Use Istanbul Stock Exchange data to predict MSCI Emerging Markets index returns.

Performance targets:
- DTLearner tests: <10 seconds each
- RTLearner tests: <3 seconds each  
- BagLearner tests: <10 seconds each
- InsaneLearner: <10 seconds

Generate charts as .png files. 
