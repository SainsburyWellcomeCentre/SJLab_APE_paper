import string

import matplotlib
from utils.kernel_regression.regression_plotting_utils import *
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.patches as mpatches
from set_global_params import spreadsheet_path
import os

sh_path = os.path.join(spreadsheet_path, 'fig2')

font = {'size': 9}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig = plt.figure(figsize=[7.5, 6.5], constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=4)
gs.set_width_ratios([1.5, 1, 1, 1])

model_description_ax1 = fig.add_subplot(gs[0, 0])
model_description_ax2 = fig.add_subplot(gs[1, 0])

ts_cue_ax = fig.add_subplot(gs[0, 1])
ts_move_ax = fig.add_subplot(gs[0, 2], sharey=ts_cue_ax)
ts_rew_ax = fig.add_subplot(gs[0, 3])

vs_cue_ax = fig.add_subplot(gs[1, 1], sharey=ts_cue_ax)
vs_move_ax = fig.add_subplot(gs[1, 2], sharey=ts_move_ax)
vs_rew_ax = fig.add_subplot(gs[1, 3], sharey=ts_rew_ax)

total_perc_exp = fig.add_subplot(gs[2, 0])
perc_exp_cue_ax = fig.add_subplot(gs[2, 1])
perc_exp_move_ax = fig.add_subplot(gs[2, 2], sharey=perc_exp_cue_ax)
perc_exp_rew_ax = fig.add_subplot(gs[2, 3], sharey=perc_exp_cue_ax)


labelled_axes = [model_description_ax1, ts_cue_ax, model_description_ax2, vs_cue_ax, total_perc_exp, perc_exp_cue_ax]

# Requires either repro data or running: 1) get_time_stamps_for_regression_all_cues_matched_trials.py
# and 2) linear_regressions_all_session_different_shifts
tail_time_stamps, tail_reg_means, tail_reg_sems, tail_reg_sig_times = get_regression_data_for_plot(
    recording_site='tail')
nacc_time_stamps, nacc_reg_means, nacc_reg_sems, nacc_reg_sig_times = get_regression_data_for_plot(
    recording_site='Nacc')

plot_kernels_for_site(ts_move_ax, ts_cue_ax, ts_rew_ax, tail_reg_means, tail_reg_sems, tail_time_stamps, palette=['#002F3A', '#76A8DA'], legend=True)
plot_kernels_for_site(vs_move_ax, vs_cue_ax, vs_rew_ax, nacc_reg_means, nacc_reg_sems, nacc_time_stamps, palette=['#E95F32', '#F9C0AF'], legend=True)

min_y = vs_rew_ax.get_ylim()[0]
max_y = vs_rew_ax.get_ylim()[1]
rect = mpatches.Rectangle((nacc_reg_sig_times['outcome'][0] - 0.05, min_y),
                          nacc_reg_sig_times['outcome'][-1] + 0.05 - (nacc_reg_sig_times['outcome'][0] - 0.05),
                          max_y - min_y,
                          fill=True,
                          color="grey",
                          alpha=0.2,
                          linewidth=0)
vs_rew_ax.add_patch(rect)

min_y = ts_rew_ax.get_ylim()[0]
max_y = ts_rew_ax.get_ylim()[1]

min_y = ts_move_ax.get_ylim()[0]
max_y = ts_move_ax.get_ylim()[1]
gaps_between_significant_time_stamps = np.diff(tail_reg_sig_times['choice'])
window_starts = [tail_reg_sig_times['choice'][0] - 0.05]
window_ends =[]
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
    ts_move_ax.add_patch(rect)

# Requires either repro data or running: 1) get_time_stamps_for_regression_all_cues_matched_trials.py,
# 2) linear_regressions_all_session_different_shifts and  3) percentage_variance_explained.py
nacc_exp_var = load_exp_var_data_for_site('Nacc')
tail_exp_var = load_exp_var_data_for_site('tail')

full_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'full')
cue_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'cue')
choice_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'choice')
outcome_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'outcome')

make_box_plot(full_df, total_perc_exp, set_ylims=True, label='Full model')
make_box_plot(cue_df, perc_exp_cue_ax, label='Cue')
make_box_plot(choice_df, perc_exp_move_ax, label='Choice')
make_box_plot(outcome_df, perc_exp_rew_ax, label='Outcome')
print(full_df.groupby('site')['explained variance'].apply(np.median))
print(cue_df.groupby('site')['explained variance'].apply(np.median))
print(choice_df.groupby('site')['explained variance'].apply(np.median))
print(outcome_df.groupby('site')['explained variance'].apply(np.median))

for nm, df in zip(['full', 'cue', 'choice', 'outcome'], [full_df, cue_df, choice_df, outcome_df]):
    df.to_csv(os.path.join(sh_path, f'fig2G_explained_var_{nm}.csv'))

makes_plots_pretty(np.array(
    [ts_move_ax, ts_cue_ax, ts_rew_ax, vs_move_ax, vs_cue_ax, vs_rew_ax, total_perc_exp, perc_exp_cue_ax,
     perc_exp_move_ax, perc_exp_rew_ax]))
make_example_figure(model_description_ax1, model_description_ax2)

plt.show()
