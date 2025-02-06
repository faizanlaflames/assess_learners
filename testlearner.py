""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

def run_experiment(data_file):
    # load data, skip header row
    data = np.genfromtxt(data_file, delimiter=',')
    data = data[1:, 1:]

    # split into training/test sets (60/40)
    train_rows = int(0.6 * data.shape[0])
    train_x = data[:train_rows, :-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, :-1]
    test_y = data[train_rows:, -1]

    # initialize different learner types
    dt_learner = dt.DTLearner(leaf_size=1, verbose=False)
    rt_learner = rt.RTLearner(leaf_size=1, verbose=False)
    bag_learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    insane_learner = il.InsaneLearner(verbose=False)

    # train all models on training data
    dt_learner.add_evidence(train_x, train_y)
    rt_learner.add_evidence(train_x, train_y)
    bag_learner.add_evidence(train_x, train_y)
    insane_learner.add_evidence(train_x, train_y)

    # calculate in-sample metrics
    dt_pred_train = dt_learner.query(train_x)
    rt_pred_train = rt_learner.query(train_x)
    bag_pred_train = bag_learner.query(train_x)
    insane_pred_train = insane_learner.query(train_x)

    dt_rmse_train = np.sqrt(((train_y - dt_pred_train) ** 2).mean())
    rt_rmse_train = np.sqrt(((train_y - rt_pred_train) ** 2).mean())
    bag_rmse_train = np.sqrt(((train_y - bag_pred_train) ** 2).mean())
    insane_rmse_train = np.sqrt(((train_y - insane_pred_train) ** 2).mean())

    dt_corr_train = np.corrcoef(train_y, dt_pred_train)[0, 1]
    rt_corr_train = np.corrcoef(train_y, rt_pred_train)[0, 1]
    bag_corr_train = np.corrcoef(train_y, bag_pred_train)[0, 1]
    insane_corr_train = np.corrcoef(train_y, insane_pred_train)[0, 1]

    # calculate out-of-sample metrics
    dt_pred_test = dt_learner.query(test_x)
    rt_pred_test = rt_learner.query(test_x)
    bag_pred_test = bag_learner.query(test_x)
    insane_pred_test = insane_learner.query(test_x)

    dt_rmse_test = np.sqrt(((test_y - dt_pred_test) ** 2).mean())
    rt_rmse_test = np.sqrt(((test_y - rt_pred_test) ** 2).mean())
    bag_rmse_test = np.sqrt(((test_y - bag_pred_test) ** 2).mean())
    insane_rmse_test = np.sqrt(((test_y - insane_pred_test) ** 2).mean())

    dt_corr_test = np.corrcoef(test_y, dt_pred_test)[0, 1]
    rt_corr_test = np.corrcoef(test_y, rt_pred_test)[0, 1]
    bag_corr_test = np.corrcoef(test_y, bag_pred_test)[0, 1]
    insane_corr_test = np.corrcoef(test_y, insane_pred_test)[0, 1]

    # print training performance
    print("In-Sample Results:")
    print(f"DTLearner RMSE: {dt_rmse_train}, Corr: {dt_corr_train}")
    print(f"RTLearner RMSE: {rt_rmse_train}, Corr: {rt_corr_train}")
    print(f"BagLearner RMSE: {bag_rmse_train}, Corr: {bag_corr_train}")
    print(f"InsaneLearner RMSE: {insane_rmse_train}, Corr: {insane_corr_train}")

    # print testing performance  
    print("\nOut-of-Sample Results:")
    print(f"DTLearner RMSE: {dt_rmse_test}, Corr: {dt_corr_test}")
    print(f"RTLearner RMSE: {rt_rmse_test}, Corr: {rt_corr_test}")
    print(f"BagLearner RMSE: {bag_rmse_test}, Corr: {bag_corr_test}")
    print(f"InsaneLearner RMSE: {insane_rmse_test}, Corr: {insane_corr_test}")

    # create rmse comparison chart
    labels = ['DTLearner', 'RTLearner', 'BagLearner', 'InsaneLearner']
    rmse_train = [dt_rmse_train, rt_rmse_train, bag_rmse_train, insane_rmse_train]
    rmse_test = [dt_rmse_test, rt_rmse_test, bag_rmse_test, insane_rmse_test]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, rmse_train, label='In-Sample RMSE')
    plt.bar(labels, rmse_test, label='Out-of-Sample RMSE', alpha=0.5)
    plt.title('RMSE Comparison of Learners')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('rmse_comparison.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    run_experiment(sys.argv[1])