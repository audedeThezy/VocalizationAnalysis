from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
import os
import os
import sys
import pandas as pd
import glob
import time
import numpy as np
from sklearn.base import BaseEstimator
os.chdir("C:\\Users\\de Witasse Thézy\\Documents\\ENS\\cours_ENS\\M2\\MoBi\\projet")
from RecordingAnalysis4 import ExtractFeaturesRecordings
from sklearn.svm import SVC
import random as rd
import matplotlib.pyplot as plt

basepath1 = "D:\\data\\playbackT01\\ch2"
basepath2 = "D:\\data\\playback2T01\\ch2"
basepath = basepath2

####

Freference_df = pd.read_csv("D:\\data\\playback2T01\\beg_end_freq.csv")
Freference_df.columns = ["Filename", "True_BeginningFrequency", "True_EndFrequency"]
Freference_df['VocNumber'] = Freference_df.groupby('Filename').cumcount()

reference_df = Freference_df
#####
#TODO : estimate manually the beg freq and end freq, compute an index and use it instead of accuracy.


def OptimizationScore(params, X_train=None, y_train=None,reference_df=reference_df,trade_off_factor=1):
    h, n, fmin, width = params.h, params.n, params.fmin, params.width
    weight, tol = 0.5, 5 #params.weight, params.width, params.tol
    print(width)
    file_paths = glob.glob(os.path.join(basepath, '*.wav'))

    start_time = time.time()
    generated_df = ExtractFeaturesRecordings(file_paths, h=h,n=n, fmin=fmin, weight=weight, width=width, tol=tol)
    execution_time = time.time() - start_time

    generated_df['Filename'] = generated_df['Filename'].astype(int)
    generated_df['VocNumber'] = generated_df.groupby('Filename').cumcount()
    # Merge with reference_df to fill missing lines with zero values
    merged_df = pd.merge(reference_df, generated_df, on=['Filename', 'VocNumber'], how='left').fillna(0)

    # Add 'actual' column
    merged_df['actual'] = merged_df['True_Duration'] > 0
    # Add 'prediction' column
    merged_df['prediction'] = merged_df['Duration'] > 0
    true_negatives = merged_df[(merged_df['actual'] == False) & (merged_df['prediction'] == False)]
    true_positives = merged_df[(merged_df['actual'] == True) & (merged_df['prediction'] == True)]

    confusion_accuracy = (len(true_positives)+len(true_negatives))/len(merged_df)
    combined_score = confusion_accuracy - trade_off_factor * execution_time
    print(params, combined_score)

    return combined_score


def BetterOptimizationScore(params, X_train=None, y_train=None,reference_df=reference_df,trade_off_factor=1):
    h, n, fmin, width = params.h, params.n, params.fmin, params.width
    weight, tol = 0.5, 5 #params.weight, params.width, params.tol

    file_paths = glob.glob(os.path.join(basepath, '*.wav'))

    start_time = time.time()
    generated_df = ExtractFeaturesRecordings(file_paths, h=h,n=n, fmin=fmin, weight=weight, width=width, tol=tol)
    execution_time = time.time() - start_time

    generated_df['Filename'] = generated_df['Filename'].astype(int)
    generated_df['VocNumber'] = generated_df.groupby('Filename').cumcount()
    # Merge with reference_df to fill missing lines with zero values
    merged_df = pd.merge(reference_df, generated_df, on=['Filename', 'VocNumber'], how='left').fillna(0)

    # Add 'actual' column
    merged_df['actual'] = merged_df['True_BeginningFrequency'] > 0
    # Add 'prediction' column
    merged_df['prediction'] = merged_df['BeginningFrequency'] > 0


    true_negatives = merged_df[(merged_df['actual'] == False) & (merged_df['prediction'] == False)]
    true_positives = merged_df[(merged_df['actual'] == True) & (merged_df['prediction'] == True)]

    #start_freq_mae = np.abs(true_positives['BeginningFrequency'] - true_positives['True_BeginningFrequency']).mean()
    #end_freq_mae = np.abs(true_positives['EndFrequency'] - true_positives['True_EndFrequency']).mean()

    start_freq_index = (np.abs(true_positives['BeginningFrequency'] - true_positives['True_BeginningFrequency'])/true_positives['True_BeginningFrequency']).mean()
    end_freq_index = (np.abs(true_positives['EndFrequency'] - true_positives['True_EndFrequency'])/true_positives['True_EndFrequency']).mean()


    # You can compute an overall accuracy index by averaging or combining the MAE values
    accuracy_index = (start_freq_index + end_freq_index) / 2.0
    combined_score = - accuracy_index - trade_off_factor * execution_time/len(file_paths)
    print(params, combined_score)

    return combined_score
#OptimizationScore(300, 300, 38000, 0.5, 10, 5, X_train=X_train, #y_train=y_train,reference_df=reference_df,trade_off_factor=0.33)

######
# Define your parameter space
param_space = {
    'h': (100, 1100),
    'n': (100, 1100),
    'fmin': (20000, 42000),
    #'weight': (0.5, 10),
    'width': (1, 15),
    #'tol': (1, 10),
}

# Split your data into training and validation sets

X_train, y_train = reference_df["Filename"], reference_df["True_BeginningFrequency"]

class CustomEstimator(BaseEstimator):
    def __init__(self, h=100, n=100, fmin=20000, width=5) :#, weight=0.5, width=1, tol=1):
        self.h = h
        self.n = n
        self.fmin = fmin
        #self.weight = weight
        self.width = width
        #self.tol = tol

    def fit(self, X, y=None):
        # Implement your fitting logic here or leave it empty for a dummy estimator
        pass

#best_params = []
#best_accuracy = []

for t in range (10) :
    opt = BayesSearchCV(
        estimator=CustomEstimator(),
        search_spaces=param_space,
        n_iter=30,
        random_state=rd.randint(0,100),
        n_jobs=1,
        cv=3,
        n_points=1,
        scoring=BetterOptimizationScore,
        fit_params={'reference_df': reference_df, 'trade_off_factor': 1},
    )
    # Fit the optimizer on your data
    opt.fit(X_train, y_train)

    best_params.append(opt.best_params_)
    best_accuracy.append(opt.best_score_)

    res =  opt.cv_results_

    save_plot(trial = t, res =  opt.cv_results_)

    #print(f"Best Parameters: {best_params}")
    #print(f"Best Accuracy: {best_accuracy}")
###

res_dfFINAL = pd.DataFrame({'Best Param': best_params, 'Best Accuracy' : best_accuracy})
res_dfFINAL.to_csv("D:\\data\\playback2T01\\plot_opt\\res_df_FINALperfile.csv")

###
#Plot heatmap

def save_plot(trial, res) :
    param_names = ['h', 'n', 'fmin', 'width']# 'weight', 'tol']

    if 'params' in res:
        params = np.array([[x[param] for param in param_names] for x in res['params']])
    else:
        params = np.array([[x[param] for param in param_names] for x in res['param_grid']])
    # Extract the scores from either 'mean_test_score' or 'mean_train_score'
    scores_key = 'mean_test_score' if 'mean_test_score' in res else 'mean_train_score'
    scores = res[scores_key]
    # Use iteration number as the x-axis
    iteration_numbers = np.arange(len(scores))

    # Determine the layout of subplots
    num_rows = (len(param_names) + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 3 * num_rows))

    # Flatten axs if it's a 2D array
    axs = axs.flatten()

    for i, param_name in enumerate(param_names):
        # Choose a specific parameter value to represent on the y-axis (e.g., the first element)
        y_values = params[:, i]

        # Plot the scatter plot for each parameter
        scatter = axs[i].scatter(iteration_numbers, y_values, c=scores, cmap='viridis', marker='o', alpha=0.8)
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel(f'{param_name} Value')
        axs[i].set_title(f'{param_name} at each step, colored by accuracy')

    # Add colorbar to the last subplot
    cbar = plt.colorbar(scatter, ax=axs[-1])
    cbar.set_label('Accuracy', rotation=270, labelpad=15)

    # Adjust layout for better visualization
    plt.tight_layout()

    plt.savefig("D:\\data\\playback2T01\\plot_opt\\LAST%d.png"%trial, transparent=True,)

###

# Sample data
data = {'param': [
    {'fmin': 34119, 'h': 873, 'n': 948},
    {'fmin': 35682, 'h': 1100, 'n': 830},
    {'fmin': 36057, 'h': 1100, 'n': 1100},
    {'fmin': 37975, 'h': 106, 'n': 1100},
    {'fmin': 34148, 'h': 100, 'n': 1100},
    {'fmin': 35461, 'h': 1100, 'n': 1100}
],
    'Score': [-0.062718, -0.067149, -0.068564, -0.068010, -0.065539, -0.072571],
    'Index': [11, 12, 13, 14, 15, 16]
}

df = pd.DataFrame(data)

# Extract individual parameters from 'param' column
df[['fmin', 'h', 'n']] = pd.json_normalize(df['param'])

# Create subplots for each parameter
fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=False, dpi = 200)

# Plot for 'fmin'
axs[0].scatter(df['Score'], df['fmin'], c=df['Index'], cmap='viridis', alpha=0.8)
axs[0].set_xlabel('Score')
axs[0].set_ylabel('fmin')
axs[0].set_title('Scatter Plot for fmin')

# Plot for 'h'
axs[1].scatter(df['Score'], df['h'], c=df['Index'], cmap='viridis', alpha=0.8)
axs[1].set_xlabel('Score')
axs[1].set_ylabel('h')
axs[1].set_title('Scatter Plot for h')

# Plot for 'n'
axs[2].scatter(df['Score'], df['n'], c=df['Index'], cmap='viridis', alpha=0.8)
axs[2].set_xlabel('Score')
axs[2].set_ylabel('n')
axs[2].set_title('Scatter Plot for n')

cbar = plt.colorbar(ax.scatter(df['Score'], df[ax.get_ylabel()], c=df['Index'], cmap='viridis', alpha=0.8), ax=axs[2])
cbar.set_label('Index')

plt.tight_layout()
plt.savefig("D:\\data\\playback2T01\\final.png", transparent=True,)
plt.show()