
import matplotlib.pyplot as plt
import numpy as np

def plot_ED5VW(APE_mean_trace, RTC_mean_trace, APE_sem_trace, RTC_sem_trace, APE_peak_values, RTC_peak_values, APE_time, RTC_time):
    x_range = [-2, 3]
    y_range = [-0.75, 1.5]
    plt.rcParams["figure.figsize"] = (3, 6)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    fig, ax = plt.subplots(1, 2 , figsize=(6, 3)) # width, height

    # Plot with average traces:
    ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
    ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F', label = 'Choice')
    ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5',
                     linewidth=1, alpha=0.6)

    ax[0].plot(RTC_time, RTC_mean_trace, lw=2, color='#e377c2', label = 'Sound')
    ax[0].fill_between(RTC_time, RTC_mean_trace - RTC_sem_trace, RTC_mean_trace + RTC_sem_trace, facecolor='#e377c2',
                     linewidth=1, alpha=0.4)
    ax[0].legend(loc='upper right', frameon=False)
    ax[0].set_ylim(y_range)
    ax[0].set_ylabel('dLight z-score')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_xlim(x_range)
    ax[0].set_ylim(y_range)
    ax[0].yaxis.set_ticks([-0.5, 0, 0.5, 1, 1.5])
    ax[0].xaxis.set_ticks([-2, 0, 2])
    plt.tight_layout()

    # dotplot with peak values:
    for i in range(0, len(APE_peak_values)):
        x_val = [0, 1]
        y_val = [APE_peak_values[i], RTC_peak_values[i]]
        ax[1].plot(x_val, y_val, color='grey', linewidth=0.5)
        ax[1].scatter(0, APE_peak_values[i], color='#3F888F', s=100, alpha=1)
        ax[1].scatter(1, RTC_peak_values[i], color='#e377c2', s=100, alpha=1)

    x_text_values = ['Choice', 'Sound']
    ax[1].set_xticks([0, 1])
    ax[1].set_ylabel('dLight z-score')
    ax[1].set_xlim(-0.2, 1.2)
    ax[1].set_xticklabels(x_text_values)
    ax[1].set_ylim(-0.5, 1.5)
    ax[1].yaxis.set_ticks([-0.5, 0.5, 1.5])
    fig.tight_layout(pad=2)

    return fig


def plot_ED5PQR(all_group_data, curr_data_time, SOR_choice_peak_values, SOR_cue_peak_values):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.tight_layout(pad=4)
    x_range = [-2, 3]
    y_range = [-1, 2]
    nr_mice = len(all_group_data['SOR_cue'])
    a = 0
    align_to = ['SOR_cue', 'SOR_choice']
    color_mean = ['#e377c2','#3F888F']
    color_sem = ['#e377c2','#7FB5B5']
    alphas = [0.4, 0.6]
    labels = ['Sound on return', 'Choice']
    # plotting the z-scored dLight traces
    for a, key in enumerate(all_group_data.keys()):
        curr_data = all_group_data[key]
        curr_data_set_mean = np.mean(curr_data, axis=0)
        curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)

        ax[a].plot(curr_data_time, curr_data_set_mean, lw=2, color=color_mean[a], label=labels[a])
        ax[a].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem,
                               curr_data_set_mean + curr_data_set_sem, color=color_sem[a],
                               linewidth=1, alpha=alphas[a])
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['right'].set_visible(False)
        ax[a].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
        ax[a].set_xlabel('Time (s)')
        ax[a].set_ylabel('dLight z-score')
        ax[a].set_xlim(x_range)
        ax[a].set_ylim(y_range)
        ax[a].legend(loc='upper right', frameon=False)

    # plotting the peak values
    for i in range(0, len(SOR_choice_peak_values)):
        x_val = [0, 1]
        y_val = [SOR_choice_peak_values[i], SOR_cue_peak_values[i]]
        ax[2].plot(x_val, y_val, color='grey', linewidth=0.5)

    zeroes = [0, 0, 0, 0, 0, 0]
    ones = [1, 1, 1, 1, 1, 1]
    ax[2].scatter(zeroes, SOR_choice_peak_values, color='#3F888F', s=100, alpha=1)
    ax[2].scatter(ones, SOR_cue_peak_values, color='#e377c2', s=100, alpha=1)

    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(['Choice', 'Sound on return'])
    ax[2].set_ylabel('dLight z-score')
    ax[2].set_xlim(-0.5, 1.5)
    ax[2].set_ylim(y_range)


def plot_ED5ST(all_group_data, curr_data_time, return_contra_cue_on_values, return_contra_cue_off_values, return_ipsi_cue_on_values, return_ipsi_cue_off_values, group):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 2]})
    fig.tight_layout(pad=4)

    # Plotting the z-scored dLight traces ED5S
    x_range = [-2, 3]
    y_range = [-1, 2]
    nr_mice = len(all_group_data['SOR_return_cueON_contra'])
    labels = ['Contra cue ON', 'Contra cue OFF', 'Ipsi cue ON', 'Ipsi cue OFF']
    for a, key in enumerate(all_group_data.keys()):
        curr_data = all_group_data[key]
        curr_data_set_mean = np.mean(curr_data, axis=0)
        curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)

        if 'contra' in key.lower():
            color_mean = 'cyan'
            color_sem = 'lightcyan'
        else:
            color_mean = 'blue'
            color_sem = 'lightblue'

        if 'cueOFF' in key:
            lstyle = '-.'
        else:
            lstyle = '-'

        alpha = 0.75
        ax[0].plot(curr_data_time, curr_data_set_mean, lw=2, color=color_mean, linestyle= lstyle, label = labels[a])
        ax[0].fill_between(curr_data_time, curr_data_set_mean - curr_data_set_sem, curr_data_set_mean + curr_data_set_sem, color=color_sem,
                                   linewidth=1, alpha=alpha)

        ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False, fontsize=6)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('dLight z-score')
        ax[0].set_xlim(x_range)
        ax[0].set_ylim(y_range)

    # Plotting the average peak dLight values in ED5T early in training:
    mean_peak_values = [np.mean(return_contra_cue_on_values), np.mean(return_contra_cue_off_values), np.mean(return_ipsi_cue_on_values), np.mean(return_ipsi_cue_off_values)]
    sem_peak_values = [np.std(return_contra_cue_on_values) / np.sqrt(len(return_contra_cue_on_values)), np.std(return_contra_cue_off_values) / np.sqrt(len(return_contra_cue_off_values)),
                           np.std(return_ipsi_cue_on_values) / np.sqrt(len(return_ipsi_cue_on_values)), np.std(return_ipsi_cue_off_values) / np.sqrt(len(return_ipsi_cue_off_values))]

    if group == 'EarlyTraining':
        curr_marker = 'o'
    else:
        curr_marker = '+'

    for i in range(0, len(return_contra_cue_on_values)):
        x_val = [0, 1, 2, 3]
        y_val = [return_contra_cue_on_values[i], return_contra_cue_off_values[i], return_ipsi_cue_on_values[i],
                 return_ipsi_cue_off_values[i]]
        ax[1].plot(x_val, y_val, color='#3F888F', linewidth=0, marker=curr_marker, markersize=10)

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    # ax2.set_xticks([0, 1, 2, 3], ["contra cue on", "contra cue off", "ipsi cue on", "ipsi cue off"])
    # ax2.set_xticks([0, 1, 2, 3], labels=["contra cue on", "contra cue off", "ipsi cue on", "ipsi cue off"])
    ax[1].set_ylabel('dLight z-score')

    for i in range(0, 4):
        ax[1].plot([i, i], [mean_peak_values[i] + sem_peak_values[i], mean_peak_values[i] - sem_peak_values[i]],
                 color='r', linewidth=1)
        ax[1].plot([i], [mean_peak_values[i]], marker='o', markersize=10, color='r', linewidth=1)

    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([-1, 0, 1])
    ax[1].set_xlim([-0.5, 3.5])





def plot_ED12FG(all_group_data, time):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    fig.tight_layout(pad=4)
    x_range = [-2, 3]
    y_range = [-1, 2]
    nr_mice = len(all_group_data['choice_contra'])

    align_to = ['choice_ipsi', 'choice_contra', 'reward_correct']
    colors = ['#7AC5CD','#3D59AB', '#458B00']
    for a, key in enumerate(align_to):
        if a < 2:
            c = 0
        else:
            c = 1
        curr_data = all_group_data[key]
        curr_data_set_mean = np.mean(curr_data, axis=0)
        curr_data_set_sem = np.std(curr_data, axis=0) / np.sqrt(nr_mice)

        ax[c].plot(time, curr_data_set_mean, lw=2, color=colors[a], label=key)
        ax[c].fill_between(time, curr_data_set_mean - curr_data_set_sem,
                               curr_data_set_mean + curr_data_set_sem, color=colors[a],
                               linewidth=1, alpha=0.6)

        ax[c].legend(loc='upper right', frameon=False, fontsize=8)

    for a in range(0, 2):
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['right'].set_visible(False)
        ax[a].axvline(0, color='#808080', linewidth=0.5, ls='dashed')
        ax[a].set_xlabel('Time (s)')
        ax[a].set_ylabel('dLight z-score')
        ax[a].set_xlim(x_range)
        ax[a].set_ylim(y_range)
