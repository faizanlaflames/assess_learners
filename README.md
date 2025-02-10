Here's a more code-focused and concise version of the project overview:

## Assess Learners: ML Trading System

Implement four CART regression algorithms in Python:

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

Key requirements:
- Use NumPy arrays to represent decision trees
- Implement author() method in each learner
- BagLearner should work with any learner class
- InsaneLearner: 20 BagLearners, each with 20 LinRegLearners

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

Generate charts as .png files. Submit code to Gradescope and report to Canvas.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/28768311/52850d0f-686f-47a8-8308-228b6a12af90/Project-3_-Assess-Learners-Report.pdf

---
Answer from Perplexity: pplx.ai/share