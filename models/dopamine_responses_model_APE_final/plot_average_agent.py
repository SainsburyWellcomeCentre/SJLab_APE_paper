import os
import numpy as np
import matplotlib.pyplot as plt
from helper_functions.plotting_functions import label_axes_change_over_time
from directories import save_dir
import matplotlib

"""
Plots average agent learning curve seen in figures 3 C  G and ED fig8 C, D & E. Need to run save_many_agents.py or use file:
100_agents_classic_exp.npy which is provide along with data
"""

font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

num_agents = 100
peaks = np.load(os.path.join(save_dir, '{}_agents_classic_exp.npy'.format(num_agents), allow_pickle=True))

# plot APE contra signals over learning
APE_fig, APE_axs = plt.subplots(1, 1, figsize=[2, 2])
APE_mean = np.mean(peaks.item().get('APE contra'), axis=0)
APE_axs.plot(APE_mean, color='#3F888F')
APE_fig.set_tight_layout(True)
APE_axs = label_axes_change_over_time(APE_axs)

# plot RPE cue signals over learning
RPE_fig, RPE_axs = plt.subplots(1, 1, figsize=[2, 2])
RPE_mean = np.mean(peaks.item().get('RPE cues'), axis=0)
RPE_axs.plot(RPE_mean, color='#3F888F')
RPE_fig.set_tight_layout(True)
RPE_axs = label_axes_change_over_time(RPE_axs)


# plot all models (including salience, movement and novelty) over learning
fig, axs = plt.subplots(5, 1, figsize=[2, 8])
salience_mean = np.mean(peaks.item().get('Salience cues'), axis=0)
novelty_mean = np.mean(peaks.item().get('Novelty cues'), axis=0)
movement_mean = np.mean(peaks.item().get('Movement contra'), axis=0)

# ape, rpe, salience, novelty, movement
axs[0].plot(APE_mean, color='#3F888F')
axs[1].plot(RPE_mean, color='#3F888F')

axs[2].plot(salience_mean, color='#3F888F')
axs[3].plot(novelty_mean, color='#3F888F')
axs[4].plot(movement_mean, color='#3F888F')
for ax in axs:
    ax = label_axes_change_over_time(ax)
plt.tight_layout()

plt.show()