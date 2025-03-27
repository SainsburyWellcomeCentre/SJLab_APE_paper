from set_global_params import processed_data_path, figure_directory, reproduce_figures_path, spreadsheet_path
from utils.box_plot_utils import *
import os

sites = ['Nacc', 'tail']
peak_times = {}
rise_times = {}
original_dir = os.path.join(processed_data_path,'for_figure')
for site in sites:
    file_name = 'peak_times_and_time_to_slope_ipsi_and_contra_{}_with_means.npz'.format(site)
    if os.path.exists(os.path.join(reproduce_figures_path, 'ED_fig4', file_name)):
        dir = os.path.join(reproduce_figures_path, 'ED_fig4')
    else:
        dir = original_dir
    site_data = np.load(os.path.join(dir, file_name))
    peak_times[site] = site_data['peak_times']
    rise_times[site] = site_data['time_to_slope']

# Create a list of dictionaries with site information
peak_times_list = [{'site': site, 'peak time (s)': time} for site, times in peak_times.items() for time in times]
rise_times_list = [{'site': site, 'rise time (s)': time} for site, times in rise_times.items() for time in times]

# Convert the lists of dictionaries to DataFrames

peak_times_df = pd.DataFrame(peak_times_list).replace({'tail': 'TS', 'Nacc': 'VS'})
rise_times_df = pd.DataFrame(rise_times_list).replace({'tail': 'TS', 'Nacc': 'VS'})

# Save out rise times (ED fig 4G) to a csv
spreadsheet_filename = os.path.join(spreadsheet_path, 'ED_fig4', f'ED_fig4G_rise_times.csv')
if not os.path.exists(spreadsheet_filename):
    rise_times_df.to_csv(spreadsheet_filename)

# Set font and style parameters
font = {'size': 8.5, 'family': 'sans-serif', 'sans-serif': ['Arial']}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42



# Plot peak time comparison
plot_and_save_comparison(peak_times_df, 'peak time (s)', figure_directory, 'peak_time_comparison_TS_VS_ipsi_and_contra.pdf', save=False)

# Plot rise time comparison
plot_and_save_comparison(rise_times_df, 'rise time (s)', figure_directory, 'rise_time_comparison_TS_VS_ipsi_and_contra.pdf', save=False)
plt.show()