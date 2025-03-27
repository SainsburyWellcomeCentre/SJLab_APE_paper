import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import matplotlib
from scipy.stats import shapiro, ttest_1samp
from utils.plotting import output_significance_stars_from_pval
import seaborn as sns
from set_global_params import change_over_time_mice
from utils.tracking_analysis.first_three_session_cumsum_ang_vel import get_first_three_sessions_dlc
import scipy.stats as stats
from utils.kernel_regression.regression_plotting_utils import make_box_plot_with_shuffles
from utils.tracking_analysis.dlc_processing_utils import get_camera_trigger_times, find_nearest_trials
from utils.stats import cohen_d_one_sample
from utils.tracking_analysis.head_angle_plotting_functions import *
from set_global_params import post_processed_tracking_data_path, processed_data_path, reproduce_figures_path, spreadsheet_path


def create_box_plot_with_shuffles(all_sites_data, shuffle_compare_boxplot_ax, palette=['#FDC5AF', '#fc8d62', '#B6E2D4', '#66c2a5']):
    """
    Creates a box plot comparing actual data to shuffled data
    Args:
        all_sites_data (pandas.core.frame.DataFrame):tail, nacc and both shuffles
        shuffle_compare_boxplot_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot on
        palette (list): list of colour strings (hex)

    Returns:

    """
    make_box_plot_with_shuffles(all_sites_data, dx='recording site', dy='fit slope', fig_ax=shuffle_compare_boxplot_ax,
                                scatter_size=3, pal=palette)

    # Customize the x-axis labels
    #shuffle_compare_boxplot_ax.set_xticklabels(['shuffled \n AudS', 'AudS', 'shuffled \n VS', 'VS'], rotation=30)

    # Add significance annotations
    y = 0.5
    h = 0.1 * y
    shuffle_compare_boxplot_ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], c='k', lw=1)
    shuffle_compare_boxplot_ax.text(0.5, y + h, '***', ha='center', fontsize=12)
    shuffle_compare_boxplot_ax.plot([2, 2, 3, 3], [y, y + h, y + h, y], c='k', lw=1)
    shuffle_compare_boxplot_ax.text(2.5, y + 2 * h, 'n.s.', ha='center', fontsize=8)


def coefficient_ttest_barplot(tail_mouse_data, nacc_mouse_data, ax):
    """

    Args:
        tail_mouse_data ():
        nacc_mouse_data ():
        ax ():

    Returns:

    """
    tail_mouse_means = tail_mouse_data.groupby(['mouse'])['fit slope'].apply(np.mean)
    nacc_mouse_means = nacc_mouse_data.groupby(['mouse'])['fit slope'].apply(np.mean)

    tail_pval, nacc_pval = ttest_1samp_regression_coefs(tail_mouse_means, nacc_mouse_means)
    tail_cohend = cohen_d_one_sample(tail_mouse_means)
    nacc_cohend = cohen_d_one_sample(nacc_mouse_means)
    tail_stars = output_significance_stars_from_pval(tail_pval)
    nacc_stars = output_significance_stars_from_pval(nacc_pval)

    tail_mouse_df = pd.DataFrame({'coefficient': tail_mouse_means, 'recording site': 'TS'})
    nacc_mouse_df = pd.DataFrame({'coefficient': nacc_mouse_means, 'recording site': 'VS'})
    all_df = pd.concat([nacc_mouse_df, tail_mouse_df])
    spread_sheet_file = os.path.join(spreadsheet_path, 'ED_fig5', 'ED_fig5N_regression_coefs.csv')
    if not os.path.exists(spread_sheet_file):
        all_df.to_csv(spread_sheet_file)
    sns.barplot(data=all_df, x='recording site', y='coefficient', ax=ax, palette='Set2', errwidth=1, alpha=0.4)
    sns.swarmplot(data=all_df, x='recording site', y='coefficient', ax=ax, palette='Set2', size=5)
    ax.axhline(0, color='gray')
    y = all_df.coefficient.max()
    h = 0.1 * y
    ax.text(1, y + h, tail_stars, ha='center', fontsize=12)
    ax.text(0, y + h, nacc_stars, ha='center', fontsize=10)
    ax.set_ylim(top=y + 3 * h)


def ttest_1samp_regression_coefs(tail_mouse_means, nacc_mouse_means):
    """

    Args:
        tail_mouse_means ():
        nacc_mouse_means ():

    Returns:

    """
    print('shapiro nacc: ', shapiro(nacc_mouse_means))
    print('shapiro tail: ', shapiro(tail_mouse_means))
    tail_pval = ttest_1samp(tail_mouse_means, 0)[1]
    print('ttest pval tail: ', tail_pval)
    nacc_pval = ttest_1samp(nacc_mouse_means, 0)[1]
    print('ttest pval nacc: ', nacc_pval)
    return tail_pval, nacc_pval


def calculate_p_value_and_proportion(real_data, shuffled_data):
    """
    Calculates p-value of data vs shuffled data
    Args:
        real_data (pandas.core.frame.DataFrame): real data
        shuffled_data (pandas.core.frame.DataFrame): shuffled data

    Returns:
        p_val_data (numpy.float64): p value of t-test of real data fit slopes against shuffled data
        proportion (numpy.float64): proportion of shuffles with p-value less than real p-value
    """
    _, p_val_data = stats.ttest_ind(real_data['fit slope'], shuffled_data['fit slope'])
    print('Real data p-value:', p_val_data)

    p_vals = []
    for shuffle_num in shuffled_data['shuffle number'].unique():
        shuffle_subset = shuffled_data[shuffled_data['shuffle number'] == shuffle_num]
        _, p_val = stats.ttest_ind(real_data['fit slope'], shuffle_subset['fit slope'])
        p_vals.append(p_val)

    proportion = np.where(np.array(p_vals) <= p_val_data)[0].shape[0]/len(p_vals)
    return p_val_data, proportion


def load_tracking_data(site, save=False, load_saved=True, mice=change_over_time_mice):
    """
    Loads saved out formatted tracking data for a recording site for the first three sessions
    Args:
        site (str): 'tail' or 'nacc'
        save (bool): save new df
        load_saved (bool): load saved df
        mice (list): mouse IDs

    Returns:
        all_data (pandas.core.frame.DataFrame): all data (real and shuffled)
        q_data (pandas.core.frame.DataFrame): real data with quartiles based on dopamine response size
        s_data (pandas.core.frame.DataFrame): shuffled data with shuffled quartiles
    """
    mouse_ids = mice[site]
    data_to_save, all_data = get_first_three_sessions_dlc(mouse_ids, site, save=save, load_saved=load_saved)
    q_data = all_data[all_data['recording site'] == site]
    s_data = all_data[all_data['recording site'] == 'shuffled {}'.format(site)]
    return all_data, q_data, s_data


def plot_one_trial(x, y, ax=False, cmap='winter', alpha=0.5):
    """
    PLots a trace (x vs y) for one trials
    Args:
        x (numpy.ndarray): x values to plot
        y (numpy.ndarray): y values to plot
        ax (matplotlib.axes._subplots.AxesSubplot): axis to plot on
        cmap (str): argument for plt.get_cmap
        alpha (float): alpha for plot

    Returns:
        keys (matplotlib.axes._subplots.AxesSubplot): axes
    """
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    MAP = cmap
    NPOINTS = x.shape[0]
    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(cycler('color', [cm(1. * i / (NPOINTS - 1)) for i in range(NPOINTS - 1)]))

    for i in range(NPOINTS - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], alpha=alpha, lw=3)
    return ax


# PLOT HEAD ANGLE
def plot_one_trial_head_angle(angles, x, y, ax=False, cmap=matplotlib.cm.winter, head_size=50):
    """

    Args:
        angles (numpy.ndarray): in degrees
        x (numpy.ndarray): nose x values to plot
        y (numpy.ndarray): nose y values to plot
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot on
        cmap (str): argument for plt.get_cmap
        head_size (int): size fore head triangle

    Returns:
        keys (matplotlib.axes._subplots.AxesSubplot): axes
    """
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    NPOINTS = angles.shape[0]
    colours = cmap(np.linspace(0, 1, NPOINTS))
    for i in range(NPOINTS-1):
        triangle = Polygon(((x[i], y[i]), (x[i] - head_size/2, y[i] + head_size), (x[i] + head_size/2, y[i] + head_size)), fc=colours[i], alpha=0.6)
        transform = Affine2D().rotate_deg_around(*(x[i], y[i]), -angles[i]) + ax.transData
        triangle.set_transform(transform)
        ax.scatter(x[i],y[i], marker='o', s=8, zorder =3, color=colours[i])
        ax.add_artist(triangle)
    ax.set_xlim(50, 540)
    ax.set_ylim(100, 400)
    return ax


def plot_quantiles_formatted_data(formatted_data, key, sort_by='APE peaks', ax=None, filter_by=None, filter_value=None,
                                  num_divisions=4, colourmap=matplotlib.cm.viridis, plot_means=True, align_end=True):
    """
    Plots data based on key, dividing data into quantiles based on 'sort_by'
    Args:
        formatted_data (pandas.core.frame.DataFrame):
        key (str):
        sort_by (str):
        ax (matplotlib.axes._subplots.AxesSubplot):
        filter_by (str): column to select data by if not None
        filter_value (): value to filter data by
        num_divisions (int): number of quantiles
        colourmap (matplotlib.colors.ListedColormap): colormap for plot
        plot_means (bool): plot the mean?
        align_end (bool): only for time series data, align to start or end?

    Returns:

    """
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    if ax is None:
        fig, ax = plt.subplots()
    formatted_data = formatted_data.sort_values(by=sort_by, ascending=True, ignore_index=True)
    formatted_data['quantile'] = pd.qcut(formatted_data[sort_by], q=num_divisions)
    num_divisions = formatted_data['quantile'].unique().shape[0]
    all_xs = []
    all_ys = []
    for q in range(0, num_divisions):
        x_s = []
        y_s = []
        lengths = []
        peaks = []
        quantile = formatted_data['quantile'].unique()[q]
        quantile_data = formatted_data.loc[formatted_data['quantile'] == formatted_data['quantile'].unique()[q]]
        if filter_by:
            filtered_data = quantile_data.loc[quantile_data[filter_by] == filter_value]

        else:
            filtered_data = quantile_data
        print(filtered_data.shape)
        for i, row in filtered_data.iterrows():
            if type(key) == str:
                x = row[key]
                if not plot_means:
                    ax.plot(x, color=colours[q], alpha=0.5)
                if not np.isnan(x[0]):
                    x_s.append(x)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)
            else:
                y = row[key[1]]
                x = row[key[0]]
                if not plot_means:
                    ax.plot(x, y, color=colours[q], alpha=0.5)
                if not np.isnan(x[0]):
                    x_s.append(x)
                    y_s.append(y)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)

        if type(key) == str:
            x_array = np.empty((num_trials, max(lengths)))
            x_array[:] = np.nan
            for i in range(0, len(x_s)):
                x_array[i, max(lengths) - lengths[i]:] = x_s[i]
                mean_xs = np.mean(x_array, axis=0)
                mean_xs = mean_xs[np.logical_not(np.isnan(mean_xs))]
                all_xs.append(mean_xs)

            if plot_means: # this is for stuff where the x axis is always time
                if key == "traces":
                    time_points = (np.arange(len(mean_xs)) / 10000) - (len(mean_xs) / 10000 / 2)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
                else:
                    if align_end:
                        time_points = (np.flip(np.arange(len(mean_xs)) * -1) / 30)
                    else:
                        time_points = (np.arange(len(mean_xs)) / 30)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
        else: # here there is x and y data
            x_array = np.empty((num_trials, max(lengths)))
            x_array[:] = np.nan
            y_array = np.empty((num_trials, max(lengths)))
            y_array[:] = np.nan
            for i in range(0, len(x_s)):
                print(max(lengths) - lengths[i])
                x_array[i, max(lengths) - lengths[i]:] = x_s[i]
                y_array[i, max(lengths) - lengths[i]:] = y_s[i]
            mean_xs = np.mean(x_array, axis=0)
            mean_ys = np.mean(y_array, axis=0)
            all_xs.append(mean_xs)
            all_ys.append(mean_ys)
            if plot_means:
                ax.plot(mean_xs, mean_ys, color=colours[q], lw=1)




def load_example_mouse_movement_data(example_mouse='SNL_photo26', example_date='20200810', example_trial=50):
    """
    Loads example mouse data for movement analysis plot based on dopamine response quartiles
    Args:
        example_mouse (str): mouse to load example data for
        example_date (str): YYYYMMDD
        example_trial (int): trial number

    Returns:
        im (numpy.ndarray): image of box with mouse in it to plot the head angle trajectory on top of
        x (numpy.ndarray): nose x for one trial choice movement
        y (numpy.ndarray): nose y for one trial choice movement
        angles (numpy.ndarray): head angles for one trial choice movement
        cumsum_ang_vel (numpy.ndarray): cumulative angular velocity for one trial choice movement
        sigmoid_fit (numpy.ndarray): the sigmoid fit for cumsum_ang_vel
        quantile_data (pandas.core.frame.DataFrame): example mouse movement and dopamine data categorise into quartiles based on dopamine response size
    """
    # load example mouse data
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig5', 'movement_inside_task')
    save_out_folder = os.path.join(repro_path, example_mouse)
    if not os.path.exists(save_out_folder):
        os.makedirs(save_out_folder)
    movement_param_file = os.path.join(save_out_folder,
                                       'contra_APE_tracking{}_{}.pkl'.format(example_mouse, example_date))
    if os.path.isfile(movement_param_file):
        quantile_data = pd.read_pickle(movement_param_file)

    # Load an example trajectory
    x, y = (quantile_data['head x'].iloc[example_trial], quantile_data['head y'].iloc[example_trial])

    # Load an image
    # first, extract a frame number corresponding to the start of the trajectory for the given trial
    saving_folder = save_out_folder #processed_data_path + example_mouse + '\\'
    restructured_data_filename = example_mouse + '_' + example_date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
    camera_triggers, trial_start_stamps = get_camera_trigger_times(example_mouse, example_date,
                                                                   'Two_Alternative_Choice')

    first_cot = (trial_data[(trial_data['State name'] == 'CueDelay') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    first_cot = first_cot.astype(int)

    first_cot_triggers = find_nearest_trials(first_cot, camera_triggers)
    frame_number = first_cot_triggers[int(quantile_data['trial numbers'].iloc[example_trial])]

    # now load in the frame
    frame_dir = os.path.join(post_processed_tracking_data_path, "exampleVideo")
    im = plt.imread(os.path.join(frame_dir, "frame-{}.jpg".format(frame_number)), format="jpeg")
    angles = quantile_data['head angles'].iloc[example_trial][::2]
    cumsum_ang_vel = quantile_data['cumsum ang vel'].iloc[example_trial]
    sigmoid_fit = quantile_data['sig y'].iloc[example_trial]
    return im, x, y, angles, cumsum_ang_vel, sigmoid_fit, quantile_data


def make_quantile_scatter_plot(example_quantile_data, ax):
    """

    Args:
        example_quantile_data ():
        ax ():

    Returns:

    """
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, 4))

    quantile_midpoints = []
    mean_turns = []
    for i, q in enumerate(example_quantile_data['APE quantile'].unique()[::-1]):
        quantile_midpoint = q.mid
        trials = example_quantile_data.loc[example_quantile_data['APE quantile'] == q]
        quantile_midpoints.append(quantile_midpoint)
        mean_turns.append(np.nanmedian(trials['fitted max cumsum ang vel'].values))
        ax.scatter(np.abs(np.nanmedian(trials['fitted max cumsum ang vel'].values)), quantile_midpoint,
                   color=colours[i])

    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_turns, quantile_midpoints)

    ax.axline((0, intercept), slope=slope, color="k")
    ax.set_xlim([70, 130])
    ax.set_ylim([0, 4])
    ax.set_xlabel('max turn angle')
    ax.set_ylabel('fluorescence \n peak size')
    return mean_turns, quantile_midpoints

