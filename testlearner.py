"""test a learner.  (c) 2015 tucker balch

copyright 2018, georgia institute of technology (georgia tech)
atlanta, georgia 30332
all rights reserved

template code for cs 4646/7646

georgia tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  this copyright statement should not be removed
or edited.

we do grant permission to share solutions privately with non-students such
as potential employers. however, sharing with other current or future
students of cs 7646 is prohibited and subject to being investigated as a
gt honor code violation.

-----do not edit anything above this line---
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il
import time
import sys

def experiment1(data_file):
   
    np.random.seed(903471054) 
    
    data = np.genfromtxt(data_file, delimiter=',')
    data = data[1:, 1:]  
    
    leaf_sizes = range(1, 51)
    in_sample_rmse = []
    out_sample_rmse = []
    
    # split data once before the loop
    train_rows = int(0.6 * data.shape[0])
    np.random.shuffle(data)
    train_data = data[:train_rows]
    test_data = data[train_rows:]
    
    for ls in leaf_sizes:
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        
        learner = dt.DTLearner(leaf_size=ls, verbose=False)
        learner.add_evidence(train_x, train_y)
        
        pred_train = learner.query(train_x)
        pred_test = learner.query(test_x)
        in_sample_rmse.append(np.sqrt(((train_y - pred_train)**2).mean()))
        out_sample_rmse.append(np.sqrt(((test_y - pred_test)**2).mean()))
    
    # reverse the x-axis for leaf sizes
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, in_sample_rmse, label='In-Sample RMSE')
    plt.plot(leaf_sizes, out_sample_rmse, label='Out-of-Sample RMSE')
    plt.title('Experiment 1: DTLearner RMSE by Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment1_overfitting.png')
    plt.close()

def experiment2(data_file):
    
    np.random.seed(903471054) 
    
    data = np.genfromtxt(data_file, delimiter=',')
    data = data[1:, 1:]
    
    # split data once before the loop
    train_rows = int(0.6 * data.shape[0])
    np.random.shuffle(data)
    train_data = data[:train_rows]
    test_data = data[train_rows:]
    
    leaf_sizes = range(1, 51)
    bag_rmse = []
    single_rmse = []
    
    for ls in leaf_sizes:
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        
        # single dtlearner
        single_learner = dt.DTLearner(leaf_size=ls, verbose=False)
        single_learner.add_evidence(train_x, train_y)
        pred_single = single_learner.query(test_x)
        single_rmse.append(np.sqrt(((test_y - pred_single)**2).mean()))
        
        # bagged dtlearner
        bag_learner = bl.BagLearner(
            learner=dt.DTLearner,
            kwargs={"leaf_size": ls},
            bags=20,
            boost=False,
            verbose=False
        )
        bag_learner.add_evidence(train_x, train_y)
        pred_bag = bag_learner.query(test_x)
        bag_rmse.append(np.sqrt(((test_y - pred_bag)**2).mean()))
    
    # plot with reversed x-axis like experiment 1
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, bag_rmse, label='Bagged DTLearner')
    plt.plot(leaf_sizes, single_rmse, label='Single DTLearner')
    plt.title('Experiment 2: Bagging Effect on Overfitting (DTLearner)')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment2_bagging.png')
    plt.close()

def experiment3(data_file):
    
    np.random.seed(903471054)  
    
    data = np.genfromtxt(data_file, delimiter=',')
    data = data[1:, 1:]
    
    # split data once before the loop
    train_rows = int(0.6 * data.shape[0])
    np.random.shuffle(data)
    train_data = data[:train_rows]
    test_data = data[train_rows:]
    
    leaf_sizes = range(1, 51)
    metrics = {
        'DT_MAE': [],
        'RT_MAE': [],
        'DT_Time': [],
        'RT_Time': []
    }
    
    for ls in leaf_sizes:
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        
        # dtlearner metrics
        start_time = time.time()
        dt_learner = dt.DTLearner(leaf_size=ls, verbose=False)
        dt_learner.add_evidence(train_x, train_y)
        dt_time = time.time() - start_time
        dt_pred = dt_learner.query(test_x)
        dt_mae = np.mean(np.abs(test_y - dt_pred))
        
        # rtlearner metrics
        start_time = time.time()
        rt_learner = rt.RTLearner(leaf_size=ls, verbose=False)
        rt_learner.add_evidence(train_x, train_y)
        rt_time = time.time() - start_time
        rt_pred = rt_learner.query(test_x)
        rt_mae = np.mean(np.abs(test_y - rt_pred))
        
        metrics['DT_MAE'].append(dt_mae)
        metrics['RT_MAE'].append(rt_mae)
        metrics['DT_Time'].append(dt_time)
        metrics['RT_Time'].append(rt_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, metrics['DT_MAE'], label='DTLearner MAE')
    plt.plot(leaf_sizes, metrics['RT_MAE'], label='RTLearner MAE')
    plt.title('Experiment 3: MAE Comparison (DT vs RT)')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment3_mae.png')
    plt.close()
    
    # plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, metrics['DT_Time'], label='DTLearner Training Time')
    plt.plot(leaf_sizes, metrics['RT_Time'], label='RTLearner Training Time')
    plt.title('Experiment 3: Training Time Comparison')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment3_time.png')
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    experiment1(data_file)
    experiment2(data_file)
    experiment3(data_file)
    print("Experiments completed. Check the generated PNG files for results.")
    