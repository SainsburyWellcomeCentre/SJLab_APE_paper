import pandas as pd

from utils.plotting_visuals import makes_plots_pretty, set_plotting_defaults
from utils.tracking_analysis.head_angle_plotting_functions import *
from set_global_params import camera_sample_rate
from scipy.stats import ttest_1samp, shapiro
from save_to_excel import save_ax_data_to_excel


set_plotting_defaults(font_size=8)

# load group data and shuffled_data 
all_nacc_data, nacc_quantile_data, shuffled_nacc_quantile_data = load_tracking_data('Nacc', save=False, mice=change_over_time_mice)
all_tail_data, tail_quantile_data, shuffled_tail_quantile_data = load_tracking_data('tail', save=False, mice=change_over_time_mice)



fig, axs = plt.subplots(2, 3, figsize=(8, 5))

# Example mouse
im, x, y, angles, cumsum_ang_vel, sigmoid_fit, example_quantile_data = load_example_mouse_movement_data(example_mouse='SNL_photo26', example_date='20200810')
ax = axs[0, 0]
ax.pcolor(im.mean(axis=2), cmap='Greys_r', rasterized=True)
plot_one_trial_head_angle(angles, x[::2],y[::2], ax=ax)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# PLOT EXAMPLE SIGMOID FIT
time = np.arange(0, len(cumsum_ang_vel))/camera_sample_rate
plot_one_trial(time, cumsum_ang_vel, ax=axs[0, 1], alpha=0.3)
plot_one_trial(time, sigmoid_fit, ax=axs[0, 1], alpha=1)
max_cumsum_ang_vel = np.max(sigmoid_fit)
axs[0, 1].axhline(y=max_cumsum_ang_vel, color='k')
axs[0, 1].set_xlabel('time (s)')
axs[0, 1].set_ylabel('head angle (degrees)')


# PLOT APE TRACES BY QUANTILE
plot_quantiles_formatted_data(example_quantile_data, "traces", ax=axs[0, 2])
axs[0, 2].set_xlabel('time from leaving centre port (s)')
axs[0, 2].set_ylabel('z-scored fluorescence')
axs[0, 2].axvline(x=0, color='k')
sh_path = os.path.join(spreadsheet_path, 'ED_fig5')
traces_sh_fn = 'ED_fig5K_traces_example_mouse_quartiles.xlsx'
if not os.path.exists(os.path.join(sh_path, traces_sh_fn)):
    save_ax_data_to_excel(axs[0, 2], os.path.join(sh_path, traces_sh_fn))



# CUMSUM ANG VEL BY QUANTILE
plot_quantiles_formatted_data(example_quantile_data, "sig y", ax=axs[1, 0])
axs[1, 0].set_xlabel('time from entering choice port (s)')
axs[1, 0].set_ylabel('head angle (degrees)')
axs[1, 0].set_ylim(55, 115)
axs[1, 0].set_xlim(-0.3, 0)
turn_angle_sh_fn = 'ED_fig5L_turn_angle_example_mouse_quartiles.xlsx'
if not os.path.exists(os.path.join(sh_path, turn_angle_sh_fn)):
    save_ax_data_to_excel(axs[1, 0], os.path.join(sh_path, turn_angle_sh_fn))


# APE size against max cumsum angular velocity per quantile
x_data_scatter, y_data_scatter = make_quantile_scatter_plot(example_quantile_data, axs[1, 1])
scatter_sh_fn = 'ED_fig5M_scatter_example_mouse_quartiles.csv'
if not os.path.exists(os.path.join(sh_path, scatter_sh_fn)):
    scatter_df = pd.DataFrame({'mean turns': x_data_scatter, 'fluorescence peak size': y_data_scatter})
    scatter_df.to_csv(os.path.join(sh_path, scatter_sh_fn))


# Summary plot and stats
# Data for 'tail' site
tail_real_data = all_tail_data[all_tail_data['recording site'] == 'tail']
tail_shuffled_data = shuffled_tail_quantile_data[shuffled_tail_quantile_data['recording site'] == 'shuffled tail']

tail_p_val_data, tail_proportion = calculate_p_value_and_proportion(tail_real_data, tail_shuffled_data)
print('Tail proportion of shuffles with p-value <= actual p-value:', tail_proportion)

# Data for 'Nacc' site
nacc_real_data = all_nacc_data[all_nacc_data['recording site'] == 'Nacc']
nacc_shuffled_data = shuffled_nacc_quantile_data[shuffled_nacc_quantile_data['recording site'] == 'shuffled Nacc']

nacc_p_val_data, nacc_proportion = calculate_p_value_and_proportion(nacc_real_data, nacc_shuffled_data)
print('Nacc proportion of shuffles with p-value <= actual p-value:', nacc_proportion)

# Combine tail and nacc data for plot
all_sites_data = pd.concat([all_tail_data, all_nacc_data])

# Create a box plot with shuffles (used in original pre-print)
#create_box_plot_with_shuffles(all_sites_data, axs[1,2])

# ttest coefs against zero and make bar plot
coefficient_ttest_barplot(tail_real_data, nacc_real_data, axs[1, 2])
plt.tight_layout()
makes_plots_pretty(axs.ravel())


plt.show()
