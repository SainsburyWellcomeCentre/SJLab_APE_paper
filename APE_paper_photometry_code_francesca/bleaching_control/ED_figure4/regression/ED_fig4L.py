import numpy as np
from utils.plotting_visuals import set_plotting_defaults
from utils.kernel_regression.return_to_centre_plotting_utils import get_regression_data_for_plot, plot_kernels_for_site
import matplotlib.pyplot as plt
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.patches as mpatches

tail_time_stamps, tail_reg_means, tail_reg_sems, tail_reg_sig_times = get_regression_data_for_plot(
    recording_site='tail', reg_type='_return_to_centre_trimmed_traces_300frames_long_turns')


set_plotting_defaults()

fig, axs = plt.subplots(1, 4, figsize=(10, 3), sharey=True)

plot_kernels_for_site(axs[0], axs[1], axs[2], axs[3], tail_reg_means, tail_reg_sems, tail_time_stamps,
                      palette=['#002F3A', '#76A8DA'], legend=True)
makes_plots_pretty(axs)

min_y = axs[0].get_ylim()[0]
max_y = axs[0].get_ylim()[1]
gaps_between_significant_time_stamps = np.diff(tail_reg_sig_times['choice'])
window_starts = [tail_reg_sig_times['choice'][0] - 0.05]
window_ends = []
if any(gaps_between_significant_time_stamps > 0.11):
    window_starts.append(tail_reg_sig_times['choice'][np.where(gaps_between_significant_time_stamps > 0.11)][0] + 0.05)
    window_ends.append(tail_reg_sig_times['choice'][np.where(gaps_between_significant_time_stamps > 0.11)[0]][0] - 0.05)
window_ends.append(tail_reg_sig_times['choice'][-1] + 0.05)

for window_num, start in enumerate(window_starts):
    end = window_ends[window_num]
    rect = mpatches.Rectangle((start, min_y),
                              (end - start),
                              max_y - min_y,
                              fill=True,
                              color="grey",
                              alpha=0.2,
                              linewidth=0)
    axs[0].add_patch(rect)

min_y = axs[3].get_ylim()[0]
max_y = axs[3].get_ylim()[1]
gaps_between_significant_time_stamps = np.diff(tail_reg_sig_times['returns'])
window_starts = [tail_reg_sig_times['returns'][0] - 0.05]
window_ends = []
if any(gaps_between_significant_time_stamps > 0.11):
    window_starts.append(tail_reg_sig_times['returns'][np.where(gaps_between_significant_time_stamps > 0.11)][0] + 0.05)
    window_ends.append(
        tail_reg_sig_times['returns'][np.where(gaps_between_significant_time_stamps > 0.11)[0]][0] - 0.05)
window_ends.append(tail_reg_sig_times['returns'][-1] + 0.05)

for window_num, start in enumerate(window_starts):
    end = window_ends[window_num]
    rect = mpatches.Rectangle((start, min_y),
                              (end - start),
                              max_y - min_y,
                              fill=True,
                              color="grey",
                              alpha=0.2,
                              linewidth=0)
    axs[3].add_patch(rect)

plt.tight_layout()

plt.show()


