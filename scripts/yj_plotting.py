
import matplotlib.pyplot as plt
import numpy as np

def plot_SF5TU(APE_mean_trace, RTC_mean_trace, APE_sem_trace, RTC_sem_trace, APE_peak_values, RTC_peak_values, APE_time, RTC_time):
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