from utils.kernel_regression.linear_regression_utils import *
from utils.kernel_regression.return_to_centre_regression_utils import run_regression_return_to_centre_one_mouse_one_session, get_first_x_sessions_reg_rtc
from utils.post_processing_utils import remove_exps_after_manipulations
from set_global_params import processed_data_path, experiment_record_path, mice_average_traces, figure_directory
import gc
import os
import matplotlib
import matplotlib.pyplot as plt

num_sessions = 3
site = 'tail'
mouse_ids = mice_average_traces[site]
experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[(all_experiments_to_process['include return to centre'] != 'no')].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)
var_exps = []
# Initialize a list to store duration data
duration_list = []
within_2sd_duration_list = []
all_trial_durations = []
all_per_trial_exp_vars = []
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    var_exp, duration_list, within_2sd_duration_list = run_regression_return_to_centre_one_mouse_one_session(mouse, date, duration_list, within_2sd_duration_list, all_trial_durations, reg_type='_return_to_centre_300frames_long_turns')
    var_exps.append(var_exp)
    gc.collect()
experiments_to_process['var exp'] = var_exps

var_exp_filename = os.path.join(processed_data_path,'_'.join(mouse_ids) + '_var_exp_with_return_to_centre_300frames_long_turns.p') # '_var_exp_with_return_to_centre.p'
with open(var_exp_filename, "wb") as f:
    pickle.dump(experiments_to_process, f)

# Create a DataFrame from the collected data
duration_df = pd.DataFrame(duration_list)

# Calculate mean and standard deviation for ipsi and contra durations
ipsi_mean = duration_df[duration_df['type'] == 'ipsi']['duration'].mean()
ipsi_std = duration_df[duration_df['type'] == 'ipsi']['duration'].std()
contra_mean = duration_df[duration_df['type'] == 'contra']['duration'].mean()
contra_std = duration_df[duration_df['type'] == 'contra']['duration'].std()

# Plot histograms for ipsi and contra movement durations
bin_width = 0.1
bins = np.arange(0, duration_df['duration'].max() + bin_width, bin_width)

font = {'size': 8.5, 'family':'sans-serif', 'sans-serif':['Arial']}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
plt.figure(figsize=(4, 4))
plt.hist(duration_df[duration_df['type'] == 'ipsi']['duration'], bins=bins, alpha=0.5, label='Ipsi')
plt.hist(duration_df[duration_df['type'] == 'contra']['duration'], bins=bins, alpha=0.5, label='Contra')
plt.axvline(x=1.5, color='k', linestyle='--', linewidth=2)
plt.xlabel('Movement Duration (s)')
plt.ylabel('Frequency')
plt.title('Histogram of Ipsi and Contra Choice Movement Durations')
# Add text for mean and std
plt.text(0.05, plt.ylim()[1] * 0.8, f'Ipsi Mean: {ipsi_mean:.2f}\nIpsi Std: {ipsi_std:.2f}', color='blue')
plt.text(0.05, plt.ylim()[1] * 0.6, f'Contra Mean: {contra_mean:.2f}\nContra Std: {contra_std:.2f}', color='orange')

plt.legend()
plt.savefig(os.path.join(figure_directory, 'choice_movement_durations.pdf'))