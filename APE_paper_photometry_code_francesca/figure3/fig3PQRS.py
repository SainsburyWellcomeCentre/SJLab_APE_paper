from utils.plotting_visuals import set_plotting_defaults
from utils.state_change_utils import make_example_plot, get_group_data, pre_post_state_change_plot
from set_global_params import plotting_colours, spreadsheet_path, reproduce_figures_path
import matplotlib.pyplot as plt
import os
import pandas as pd



sites = ['Nacc', 'tail']

set_plotting_defaults()
for site in sites:
    make_example_plot(site)
    subfigno = 'Q' if site == 'tail' else 'S'
    group_data_file = os.path.join(spreadsheet_path, 'fig3', f'fig3{subfigno}_WN_df_{site}.csv')
    if not os.path.exists(group_data_file):
        site_data = get_group_data(site)
        site_data.to_csv(group_data_file)
    else:
        site_data = pd.read_csv(group_data_file, index_col=0)
    group_data_df = pre_post_state_change_plot(site_data, colour=plotting_colours[site][0])

plt.show()


