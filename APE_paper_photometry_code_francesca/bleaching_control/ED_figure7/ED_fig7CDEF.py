import pandas as pd

from utils.large_reward_omission_utils import make_example_traces_plot, get_unexpected_reward_change_data_for_site, compare_peaks_across_trial_types
from utils.plotting_visuals import set_plotting_defaults
from set_global_params import plotting_colours, reproduce_figures_path, spreadsheet_path
import matplotlib.pyplot as plt
import os

sites = ['Nacc', 'tail']
repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'omissions_large_rewards')
set_plotting_defaults()
for site in sites:
    repro_file = os.path.join(repro_path, f'omissions_large_rewards_downsampled_traces_peaks_{site}.pkl')
    site_data = pd.read_pickle(repro_file)
    make_example_traces_plot(site)
    summary_data = compare_peaks_across_trial_types(site_data, colour=plotting_colours[site][0])
    summary_data = summary_data.drop(columns=['decimated', 'peakidx'])
    subfig = 'D' if site == 'tail' else 'F'
    summary_csv = (os.path.join(spreadsheet_path, 'ED_fig7', f'ED_fig7{subfig}_large_rewards_omissions_{site}_summary.csv'))
    if not os.path.exists(summary_csv):
        summary_data.to_csv(summary_csv)
plt.show()


