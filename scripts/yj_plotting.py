
import matplotlib as plt


def plot_SF5TU(APE_mean_trace, RTC_mean_trace, APE_sem_trace, RTC_sem_trace, APE_peak_values, RTC_peak_values, APE_time, RTC_time):
    x_range = [-2, 3]
    y_range = [-0.75, 1.5]
    plt.rcParams["figure.figsize"] = (3, 3)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    fig, ax = plt.subplots(1, 2 , figsize=(2, 3)) # width, height

    ax[0].axvline(0, color='#808080', linewidth=0.25, linestyle='dashdot')
    ax[0].plot(APE_time, APE_mean_trace, lw=2, color='#3F888F')
    ax[0].fill_between(APE_time, APE_mean_trace - APE_sem_trace, APE_mean_trace + APE_sem_trace, color='#7FB5B5',
                     linewidth=1, alpha=0.6)

    ax[0].plot(RTC_time, RTC_mean_trace, lw=2, color='#e377c2')
    ax[0].fill_between(RTC_time, RTC_mean_trace - RTC_sem_trace, RTC_mean_trace + RTC_sem_trace, facecolor='#e377c2',
                     linewidth=1, alpha=0.4)
    ax[0].ylim(y_range)
    ax[0].ylabel('Z-scored dF/F')
    ax[0].xlabel('Time (s)')
    ax[0].xlim(x_range)
    ax[0].ylim(y_range)
    ax[0].tight_layout()

