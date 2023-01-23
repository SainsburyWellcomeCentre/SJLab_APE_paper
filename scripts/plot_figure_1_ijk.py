# -*- coding: utf-8 -*-
"""
Figure 1 IJK
============
recreate figure 1IJK from the paper
"""
# %%
# This notebook performs a statistical analysis on the learning rates between two groups of mice.
# Performs model fitting of individual mice, comparison of parameter, and statistical analysis
# between the groups to conclude differences and points of divergence in learning

# Commented out IPython magic to ensure Python compatibility.
# run this on Colab
# !rm -rf APE_paper/
# !git clone https://github.com/HernandoMV/APE_paper.git
# %pip install mouse-behavior-analysis-tools
# %cd APE_paper/docs/figures_notebooks

#%%
# 1. Import libraries
# ~~~~~~~~~~~~~~~~~~~~

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from IPython.display import clear_output
from itertools import chain
from os.path import exists
import urllib.request

from mouse_behavior_analysis_tools.utils import custom_functions as cuf
from mouse_behavior_analysis_tools.plot import make_figures
from mouse_behavior_analysis_tools.model import behavior_model as bm

clear_output()

# %%
# This dataset has been pre-processed, but conditions have not been selected
# This preprocessing includes removal of disengaged trials and
# removal of the first 5 trials of each session

# 2. Download data
# ~~~~~~~~~~~~~~~~~
dataset_name = 'Chronic-lesions_dataframe.csv'
url = "https://zenodo.org/record/7261639/files/" + dataset_name
dataset_path = '../data/' + dataset_name
# download if data is not there
if not exists(dataset_path):
    print('Downloading data...')
    urllib.request.urlretrieve(url, dataset_path)
else:
    print('Data already in directory')
# load
df_to_plot = pd.read_csv(dataset_path, index_col=0)

# %%
# parameters for the plotting
hue_order = ['Control', 'Lesion']
color_palette = [(0.24715576, 0.49918708, 0.57655991),
                 (160/255, 11/255 , 11/255)]
sns.set_palette(color_palette)

# %%
# maximum number of trials performed per mouse in the dataset:
df_to_plot.groupby(['AnimalID', 'ExperimentalGroup', 'Protocol']).max()['CumulativeTrialNumberByProtocol']

# %%
# 2. Plot performance
# ~~~~~~~~~~~~~~~~~~~~~
# bin trials every 200
df_to_plot["TrialIndexBinned200"] = (df_to_plot.CumulativeTrialNumberByProtocol // 200) * 200 + 100

#%%
# Sanity check on the data to see that it looks good
fig = make_figures.make_figure_performance_trials_animals_bin(df_to_plot)
clear_output()
plt.show(fig)
# uncomment here to save the plot
# data_directory = ''
# plt.savefig(data_directory + 'Performance_by_session_individual_animals.pdf',
#             transparent=True, bbox_inches='tight')

"""Colored dots show the performance of the past 100 trials using an average running window.
Each color represents a distinct session. The black line shows the performance value of
the past 200 trials using trial bining.
"""
# %%
# 3. Model fitting
# ~~~~~~~~~~~~~~~~~~~~~
#### Fit a sigmoid to every mouse to calculate and compare stages and learning rates
"""
The sigmoid function is (perf_end - 0.5) / (1 + np.exp(-slope * (x - bias))) + 0.5

Data is scaled before the fitting and parameters are rescaled afterwards

The maximum performace possible is defined as the maximum of the median of the trials binned every 200
"""

#calculate the maximum performance for every mouse based on the trials binned every 200
df_bin200tr = df_to_plot.groupby(['AnimalID','ExperimentalGroup','TrialIndexBinned200','Protocol']).median().reset_index()
mouse_max_perf = df_bin200tr.groupby('AnimalID').max().reset_index()[['AnimalID', 'CurrentPastPerformance100']]

# fit the model
fit_df = bm.get_df_of_behavior_model_by_animal(df_to_plot, mouse_max_perf)

# find the steepest point of each slope (max of derivative)
der_max_dir = bm.get_steepest_point_of_slope(fit_df)

#%%
# plot the curves again pointing to the maximum
# sanity check to see that these scaled values recreate the curves
fig = make_figures.make_figure_performance_trials_animals_model(df_to_plot, fit_df, der_max_dir)
clear_output()
plt.show(fig)
# plt.savefig(data_directory + 'Sigmoid_fitting.pdf', transparent=True, bbox_inches='tight')

#%%
"""As a learning speed measure I use the point of the maximum learning rate, which is the maximum of the derivative of the fitting, which is also the middle point of the curve (in between 50% and maximum performance)
The bias indicates where this point lies in the x axis: at which trial number this point is reached
"""

# add the maximum value of the derivative to the fit_df dataframe
for key, value in der_max_dir.items():
    fit_df.loc[fit_df.index[fit_df['AnimalID'] == key].tolist()[0], 'max_of_der'] = value[1]

#%%
# 4. Statistics
# ~~~~~~~~~~~~~~~~~~~~~
"""### Differences of parameters between groups"""

# calculate significance
parameters_to_show = ['maximum_performance', 'max_of_der']
# define three levels of significance for the plotting
sig_levels = [0.05, 0.01, 0.001]
pvals = []
print('Kluskal-Wallis tests on the parameters')
for var in parameters_to_show:
    kt = stats.kruskal(fit_df[fit_df.ExperimentalGroup==hue_order[0]][var].dropna(),
                       fit_df[fit_df.ExperimentalGroup==hue_order[1]][var].dropna())
    print( var + ':\t\tpvalue: ' + str(kt.pvalue) )
    pvals.append(kt.pvalue)
# %%
# 5. Plot figure
# ~~~~~~~~~~~~~~~~~~~~~
"""##### The following cell creates **Figures 1 J, K** from the paper"""

# compare the parameters between groups
titles = ['maximum\nperformance', 'maximum\nlearning rate']
ylabs = ['task performance (%)', '\u0394performance \u00D7 trials$\mathregular{^{-1}}$']
yvals = [70, 0.001] # for the statistics

fig = make_figures.make_figure_learning_parameters_between_groups(fit_df, parameters_to_show,
                                                                  titles, ylabs,
                                                                  pvals, sig_levels,
                                                                  color_palette, hue_order, yvals)

# plt.savefig(data_directory + 'Parameters_group_comparison.pdf', transparent=True, bbox_inches='tight')
plt.show(fig)

for i,s in enumerate(sig_levels):
    print(i+1, 'asterisks: pval <', s)

# %%
# 6. Learning curves
# ~~~~~~~~~~~~~~~~~~~~~
"""### calculate the statistical differences of performances between groups at different points in learning
##### Do not consider trials in which the animal is too biased to calculate performance, as the antibias acts making the mice be worse than chance and that can obscure effects of learning
"""

# define the bias threshold as over 2 std of the total dataset
bias_top_threshold = np.nanmean(df_to_plot.RightBias) + 2 * np.nanstd(df_to_plot.RightBias)
bias_bottom_threshold = np.nanmean(df_to_plot.RightBias) - 2 * np.nanstd(df_to_plot.RightBias)
# create a mask
bias_mask = np.logical_and(df_to_plot.RightBias < bias_top_threshold, df_to_plot.RightBias > bias_bottom_threshold)

# %%
# Sanity check on the data to see that it looks good
fig = make_figures.make_figure_performance_trials_animals_biased_trials(df_to_plot, bias_mask)
clear_output()
plt.show(fig)
# plt.savefig(data_directory + 'Sigmoid_fitting.pdf', transparent=True, bbox_inches='tight')

#%% 
"""#### Recalculate performance"""

PAST_WINDOW = 100
CumPerList = []
for Sid in pd.unique(df_to_plot['SessionID']):
    CumPerList.append(cuf.perf_window_calculator(df_to_plot[np.logical_and(bias_mask,
                                                                           df_to_plot['SessionID']==Sid)],
                                                 PAST_WINDOW))
# flatten the list of lists
df_to_plot.loc[bias_mask, 'CurrentPastPerformance100biasremoved'] = np.array(list(chain(*[x for x in CumPerList])))
# %%
"""##### The following cell creates the first part of **Figure 1 I** from the paper"""

# Performance differences between groups across learning
col_to_plot = 'CurrentPastPerformance100biasremoved'
fig1 = make_figures.make_figure_differences_performance_between_groups(df_to_plot, col_to_plot,
                                                                       hue_order, color_palette)

# plt.savefig(data_directory + 'Performance_between_groups.pdf', transparent=True, bbox_inches='tight')
fig1.show()
print('Shaded area indicates std, and performance is calculated using', col_to_plot)


# %%
# 7. Separate performance on high and low tones
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create a mask for the TrialHighPerc columns
high_mask = df_to_plot['TrialHighPerc'] == 98.0
low_mask = df_to_plot['TrialHighPerc'] == 2.0

# calculate the performance taking on account only these trials
CumPerListHigh = []
CumPerListLow = []
for Sid in pd.unique(df_to_plot['SessionID']):
    CumPerListHigh.append(cuf.perf_window_calculator(df_to_plot[np.logical_and(high_mask,
                                                                               df_to_plot['SessionID']==Sid)],
                                                     PAST_WINDOW))
    CumPerListLow.append(cuf.perf_window_calculator(df_to_plot[np.logical_and(low_mask,
                                                                              df_to_plot['SessionID']==Sid)],
                                                    PAST_WINDOW))
# flatten the list of lists
df_to_plot.loc[high_mask, 'CurrentPastPerformance100biasremovedHigh'] = np.array(list(chain(*[x for x in CumPerListHigh])))
df_to_plot.loc[low_mask, 'CurrentPastPerformance100biasremovedLow'] = np.array(list(chain(*[x for x in CumPerListLow])))

# %%
df_to_plot[['CurrentPastPerformance100biasremovedHigh', 'CurrentPastPerformance100biasremovedLow']].tail(100)

# %%
# Plot performance on high and low tones
col_to_plot = 'CurrentPastPerformance100biasremovedLow'
figX = make_figures.make_figure_differences_performance_between_groups(df_to_plot, col_to_plot,
                                                                       hue_order, color_palette)

# plt.savefig(data_directory + 'Performance_between_groups.pdf', transparent=True, bbox_inches='tight')
figX.show()

"""
This approach doesn't seem to work well. It might be because I am using both trials to calculate performance.
Let's eliminate those trials from the dataset
"""

# %%
df_high = df_to_plot[high_mask].copy()
df_low = df_to_plot[low_mask].copy()

# reset the index and the cumulative trial number by protocol
df_high.reset_index(inplace=True)
df_low.reset_index(inplace=True)
df_high.drop('index', axis=1, inplace=True)
df_low.drop('index', axis=1, inplace=True)

# Restart the count of CumulativeTrialNumber for every protocol
df_high['CumulativeTrialNumberByProtocol'] = np.nan
df_low['CumulativeTrialNumberByProtocol'] = np.nan

for Aid in pd.unique(df_high['AnimalID']):
    for Prot in pd.unique(df_high['Protocol']):
        conditions = np.logical_and(df_high['AnimalID'] == Aid, df_high['Protocol'] == Prot)
        df_high.CumulativeTrialNumberByProtocol.loc[df_high[conditions].index] = \
            np.arange(len(df_high[conditions])) + 1

for Aid in pd.unique(df_low['AnimalID']):
    for Prot in pd.unique(df_low['Protocol']):
        conditions = np.logical_and(df_low['AnimalID'] == Aid, df_low['Protocol'] == Prot)
        df_low.CumulativeTrialNumberByProtocol.loc[df_low[conditions].index] = \
            np.arange(len(df_low[conditions])) + 1


# calculate performance
PAST_WINDOW = 50
CumPerListHigh = []
CumPerListLow = []
for Sid in pd.unique(df_to_plot['SessionID']):
    CumPerListHigh.append(cuf.perf_window_calculator(df_high[df_high['SessionID']==Sid],
                                                     PAST_WINDOW))
    CumPerListLow.append(cuf.perf_window_calculator(df_low[df_low['SessionID']==Sid],
                                                     PAST_WINDOW))
# flatten the list of lists
df_high['CurrentPastPerformance100biasremoved'] = np.array(list(chain(*[x for x in CumPerListHigh])))
df_low['CurrentPastPerformance100biasremoved'] = np.array(list(chain(*[x for x in CumPerListLow])))

# %%
# Plot performance on high and low tones
col_to_plot = 'CurrentPastPerformance100biasremoved'
figX = make_figures.make_figure_differences_performance_between_groups(df_low, col_to_plot,
                                                                       hue_order, color_palette)
data_directory = ''
plt.savefig(data_directory + 'Chronic_lesions_Performance_between_groups_low_tones.pdf', transparent=True, bbox_inches='tight')
figX.show()



# %%

"""#### Calculate the significance by resampling: suffle the group labels multiple times and calculate the likelihood of observing this data
The shuffling respects the proportion of mice in every group.

References:

Supplementary figure 4 in here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2562676/

See also methods here: https://www.biorxiv.org/content/10.1101/716274v3.full
"""

# define a 100-trial window to bin the data
xbin = 100
df_to_plot["TrialIndexBinned"] = (df_to_plot.CumulativeTrialNumberByProtocol // xbin) * xbin + xbin / 2
print('Trials are binned in groups of', xbin)

# groupby so each animal has a mean of the performance in each bin
df_bintr = df_to_plot.groupby(['AnimalID','ExperimentalGroup','TrialIndexBinned','Protocol']).mean().reset_index()

# create a scaled version of the performance
df_bintr['Performance'] = df_bintr.FirstPokeCorrect * 100

# calculate the differences of the real means using the binned data
real_data_pd = df_bintr[df_bintr.ExperimentalGroup == hue_order[1]].groupby('TrialIndexBinned').mean()['Performance'] -\
               df_bintr[df_bintr.ExperimentalGroup == hue_order[0]].groupby('TrialIndexBinned').mean()['Performance']

# Select the amount of times to shuffle. Originally this is done 10,000 times
# use a smaller number to speed things up
nsh = 10000
print('Data is shuffled', nsh, 'times')

# select the important columns to get the shuffled data
df_colsel = df_bintr[['AnimalID', 'ExperimentalGroup', 'TrialIndexBinned', 'Performance']].copy()
# get the shuffled data
shrdf = cuf.get_shuffled_means_difference_df(df_colsel, hue_order, nsh)

# calculate the confidence intervals for the shuffled data
pos_ci = shrdf.groupby('TrialIndexBinned').quantile(.95)
neg_ci = shrdf.groupby('TrialIndexBinned').quantile(.05)

"""##### The following cell creates the second part of **Figure 1 I** from the paper

This shows how likely is that each time point crosses 'random' line (point-base significance).
"""

fig2 = make_figures.make_figure_differences_performance_significance(real_data_pd, pos_ci, neg_ci)

data_directory = '../../data/'
plt.savefig(data_directory + 'Differences_of_means_significance_by_trial_bins.pdf',transparent=True, bbox_inches='tight')

fig2.show()

"""#### Substitute of the mixed anova: find the likelihood of any point being significant. Shuffle more data and quantify the percentage of times there is a crossing. Generate global bands of confidence"""

quants_to_test = [0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9999]
nsh=1000

global_sig = cuf.get_shuffled_means_difference_global_significance(df_colsel, shrdf, quants_to_test, nsh, hue_order)

# plot the confidence intervals and print their global p-values:
fig = make_figures.make_figure_differences_performance_significance_global(real_data_pd, quants_to_test, shrdf, global_sig, nsh)
fig.show()

