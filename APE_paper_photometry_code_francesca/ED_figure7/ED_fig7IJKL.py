from utils.plotting_visuals import set_plotting_defaults
from utils.value_change_plot_utils import make_example_plot, make_group_data_plot, reproduce_figures_path, spreadsheet_path
from set_global_params import plotting_colours
import matplotlib.pyplot as plt
import pandas as pd
import os

sites = ['Nacc', 'tail']

set_plotting_defaults()
repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'value_change')
for site in sites:
    repro_file = os.path.join(repro_path, f'value_change_downsampled_traces_peaks_{site}.pkl')
    site_data = pd.read_pickle(repro_file)
    make_example_plot(site)
    summary_data = make_group_data_plot(site_data, plotting_colours[site][0])
    subfig = 'J' if site == 'tail' else 'L'
    summary_csv = (os.path.join(spreadsheet_path, 'ED_fig7', f'ED_fig7{subfig}_value_change_{site}_summary.csv'))
    if not os.path.exists(summary_csv):
        (summary_data.T).to_csv(summary_csv)
plt.show()
