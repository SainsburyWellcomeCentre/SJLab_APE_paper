from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd


def plot_one_trial(trial_number, coordinates, cot_times, trial_start_times, ax=False, cmap='winter'):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    MAP = cmap
    x = coordinates[0][cot_times[trial_number]: trial_start_times[trial_number]]
    y = coordinates[1][cot_times[trial_number]: trial_start_times[trial_number]]
    NPOINTS = x.shape[0]
    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(cycler('color', [cm(1. * i / (NPOINTS - 1)) for i in range(NPOINTS - 1)]))
    
    for i in range(NPOINTS - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], alpha=0.5)
    ax.set_xlim(50, 600)
    ax.set_ylim(0, 400)
    return ax


def plot_trials_by_param(param_to_sort_by,trial_numbers, coordinates, cot_times, trial_start_times, num_divisions=5, ax=False):
    if not ax:
        fig, ax = plt.subplots(1, num_divisions + 2, figsize=((num_divisions + 1)*3, 3))
    clean_peaks = [c for c in param_to_sort_by if c.size != 0]
    clean_param_to_sort_by = np.asarray(clean_peaks)

    vals, bins, q = ax[0].hist(clean_param_to_sort_by, bins=int(clean_param_to_sort_by.shape[0]/num_divisions), color='k')
    colours = matplotlib.cm.gray(np.linspace(0, 0.8, num_divisions))
    trial_numbers = trial_numbers.astype(int)
    for q in range(0, num_divisions):
        chunk = 1/num_divisions
        if q >= 1 and q < num_divisions - 1:
            lower_quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            upper_quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            ax[0].axvline(lower_quantile, 0, max(vals), color='r')
            ax[0].axvline(upper_quantile, 0, max(vals), color='r')
        elif q == 0:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by < quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        elif q == num_divisions - 1:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by >= quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        for i, trial_number in enumerate(trials_in_quantile):
            if trial_number != np.max(trial_numbers):
                x = coordinates[0][cot_times[trial_number]: trial_start_times[trial_number]]
                y = coordinates[1][cot_times[trial_number]: trial_start_times[trial_number]]

                ax[q + 1].plot(x,y, color=colours[q], lw=1)
                ax[num_divisions + 1].plot(x, y, color=colours[q], alpha=0.5, lw=1)
        ax[q + 1].set_title('quantile: {}'.format(q))

    ax[num_divisions + 1].set_title('all quantiles')

    ax[0].set_xlabel('APE size (z-scored peak)')
    ax[0].set_ylabel('counts')
    plt.tight_layout()
    return ax


def plot_mean_trajectory_by_param(param_to_sort_by,trial_numbers, coordinates, cot_times, trial_start_times, num_divisions=5, ax=False, colourmap=matplotlib.cm.viridis):
    if not ax:
        fig, ax = plt.subplots(1, num_divisions + 2, figsize=((num_divisions + 1)*3, 3))
    clean_peaks = [c for c in param_to_sort_by if c.size != 0]
    clean_param_to_sort_by = np.asarray(clean_peaks)

    vals, bins, q = ax[0].hist(clean_param_to_sort_by, bins=int(clean_param_to_sort_by.shape[0]/num_divisions), color='k')
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    trial_numbers = trial_numbers.astype(int)
    all_xs = []
    all_ys = []
    for q in range(0, num_divisions):
        chunk = 1/num_divisions
        if q >= 1 and q < num_divisions - 1:
            lower_quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            upper_quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            peaks_in_quantile = param_to_sort_by[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            ax[0].axvline(lower_quantile, 0, max(vals), color='r')
            ax[0].axvline(upper_quantile, 0, max(vals), color='r')
        elif q == 0:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by < quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by < quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        elif q == num_divisions - 1:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by >= quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by >= quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')

        x_s = []
        y_s = []
        lengths = []
        peaks = []
        for i, trial_number in enumerate(trials_in_quantile):
            if trial_number != np.max(trial_numbers):
                x = coordinates[0][cot_times[trial_number]: trial_start_times[trial_number]]
                y = coordinates[1][cot_times[trial_number]: trial_start_times[trial_number]]
                x_s.append(x)
                y_s.append(y)
                lengths.append(x.shape[0])
                peaks.append(peaks_in_quantile[i])
        num_trials = len(x_s)
        x_array = np.empty((num_trials, max(lengths)))
        x_array[:] = np.nan
        y_array = np.empty((num_trials, max(lengths)))
        y_array[:] = np.nan
        for i in range(0, len(x_s)):
            x_array[i, 0: lengths[i]] = x_s[i]
            y_array[i, 0: lengths[i]] = y_s[i]
        all_xs.append(x_array)
        all_ys.append(y_array)
        mean_xs = np.mean(x_array, axis=0)
        mean_ys = np.mean(y_array, axis=0)
        ax[q + 1].plot(mean_xs, mean_ys, color=colours[q], lw=1)
        ax[num_divisions + 1].plot(mean_xs, mean_ys,color=colours[q], alpha=0.5, lw=2)
        ax[q + 1].set_title('quantile: {}'.format(q))
    ax[num_divisions + 1].set_title('all quantiles')

    ax[0].set_xlabel('APE size (z-scored peak)')
    ax[0].set_ylabel('counts')
    plt.tight_layout()

    return ax, all_xs, all_ys

def plot_mean_ang_v_by_param(param_to_sort_by,trial_numbers, ang_v, cot_times, choice_times, num_divisions=5, ax=False, colourmap=matplotlib.cm.viridis):
    if not ax:
        fig, ax = plt.subplots(1, num_divisions + 2, figsize=((num_divisions + 1)*3, 3))
    clean_peaks = [c for c in param_to_sort_by if c.size != 0]
    clean_param_to_sort_by = np.asarray(clean_peaks)
    vals, bins, q = ax[0].hist(clean_param_to_sort_by, bins=int(clean_param_to_sort_by.shape[0]/num_divisions), color='k')
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    trial_numbers = trial_numbers.astype(int)
    all_xs = []

    for q in range(0, num_divisions):
        chunk = 1/num_divisions
        if q >= 1 and q < num_divisions - 1:
            lower_quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            upper_quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            peaks_in_quantile = param_to_sort_by[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            ax[0].axvline(lower_quantile, 0, max(vals), color='r')
            ax[0].axvline(upper_quantile, 0, max(vals), color='r')
        elif q == 0:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by < quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by < quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        elif q == num_divisions - 1:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by >= quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by >= quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        x_s = []
        lengths = []
        peaks = []
        for i, trial_number in enumerate(trials_in_quantile):
            if trial_number != np.max(trial_numbers):
                #x = np.cumsum(ang_v[cot_times[trial_number]: choice_times[trial_number]])
                x = ang_v[cot_times[trial_number]: choice_times[trial_number]]
                x_s.append(x)
                lengths.append(x.shape[0])
                peaks.append(peaks_in_quantile[i])
        num_trials = len(x_s)
        x_array = np.empty((num_trials, max(lengths)))
        x_array[:] = np.nan
        for i in range(0, len(x_s)):
            x_array[i, 0: lengths[i]] = x_s[i]
        all_xs.append(x_array)
        mean_xs = np.mean(x_array, axis=0)
        ax[q + 1].plot(mean_xs, color=colours[q], lw=1)
        ax[num_divisions + 1].plot(mean_xs, color=colours[q], alpha=0.5, lw=2)
        ax[q + 1].set_title('quantile: {}'.format(q))
    ax[num_divisions + 1].set_title('all quantiles')
    ax[0].set_xlabel('APE size (z-scored peak)')
    ax[0].set_ylabel('counts')
    plt.tight_layout()
    return ax, all_xs

def plot_all_trials_ang_v_by_param(param_to_sort_by,trial_numbers, ang_v, cot_times, choice_times, num_divisions=5, ax=False, colourmap=matplotlib.cm.viridis):
    if not ax:
        fig, ax = plt.subplots(1, num_divisions + 2, figsize=((num_divisions + 1)*3, 3))
    clean_peaks = [c for c in param_to_sort_by if c.size != 0]
    clean_param_to_sort_by = np.asarray(clean_peaks)
    vals, bins, q = ax[0].hist(clean_param_to_sort_by, bins=int(clean_param_to_sort_by.shape[0]/num_divisions), color='k')
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    trial_numbers = trial_numbers.astype(int)
    for q in range(0, num_divisions):
        chunk = 1/num_divisions
        if q >= 1 and q < num_divisions - 1:
            lower_quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            upper_quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            peaks_in_quantile = param_to_sort_by[np.where(np.logical_and(param_to_sort_by > lower_quantile, param_to_sort_by <= upper_quantile))[0]]
            ax[0].axvline(lower_quantile, 0, max(vals), color='r')
            ax[0].axvline(upper_quantile, 0, max(vals), color='r')
        elif q == 0:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by < quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by < quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        elif q == num_divisions - 1:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            trials_in_quantile = trial_numbers[np.where(param_to_sort_by >= quantile)[0]]
            peaks_in_quantile = param_to_sort_by[np.where(param_to_sort_by >= quantile)[0]]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        x_s = []
        lengths = []
        peaks = []
        for i, trial_number in enumerate(trials_in_quantile):
            if trial_number != np.max(trial_numbers):
                x = ang_v[cot_times[trial_number]: choice_times[trial_number]]

                ax[q + 1].plot(x, color=colours[q], lw=1)
                ax[num_divisions + 1].plot(x, color=colours[q], alpha=0.5, lw=1)
        ax[q + 1].set_title('quantile: {}'.format(q))
    ax[num_divisions + 1].set_title('all quantiles')
    ax[0].set_xlabel('APE size (z-scored peak)')
    ax[0].set_ylabel('counts')
    plt.tight_layout()
    return ax


def plot_quantiles_formatted_data(formatted_data, key, sort_by='APE peaks', filter_by=None, filter_value=None, num_divisions=4, colourmap=matplotlib.cm.viridis, plot_means=True):
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    fig, ax = plt.subplots(1, num_divisions + 1, sharex=True, sharey=True, figsize=(7, 2))
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
                    ax[q].plot(x, color=colours[q])
                    ax[num_divisions].plot(x, color=colours[q])
                if not np.isnan(x[0]):
                    x_s.append(x)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)
            else:
                y = row[key[1]]
                x = row[key[0]]
                if not plot_means:
                    ax[q].plot(x, y, color=colours[q])
                    ax[num_divisions].plot(x, y, color=colours[q])
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
                x_array[i, 0: lengths[i]] = x_s[i]
                #x_array[i, max(lengths) - lengths[i]:] = x_s[i]
            mean_xs = np.mean(x_array, axis=0)
            # v = np.zeros([x_array.shape[1]])
            # for i in range(0, x_array.shape[1]):
            #     is_not_nan = np.invert(np.isnan(x_array[:, i]))
            #     v[i] = np.count_nonzero(is_not_nan) / x_array.shape[1]
            # x_array[np.isnan(x_array)] = np.nan
            # wx = x_array * v
            # mean_xs = np.mean(wx, axis=0)
            all_xs.append(mean_xs)
            if plot_means:
                ax[q].plot(mean_xs, color=colours[q], lw=1)
                ax[num_divisions].plot(mean_xs, color=colours[q], lw=1)
        else:
            x_array = np.empty((num_trials, max(lengths)))
            x_array[:] = np.nan
            y_array = np.empty((num_trials, max(lengths)))
            y_array[:] = np.nan
            for i in range(0, len(x_s)):
                x_array[i, 0: lengths[i]] = x_s[i]
                y_array[i, 0: lengths[i]] = y_s[i]
                #x_array[i, max(lengths) - lengths[i]:] = x_s[i]
                #y_array[i, max(lengths) - lengths[i]:] = y_s[i]
            mean_xs = np.mean(x_array, axis=0)
            mean_ys = np.mean(y_array, axis=0)
            all_xs.append(mean_xs)
            all_ys.append(mean_ys)
            if plot_means:
                ax[q].plot(mean_xs, mean_ys, color=colours[q], lw=1)
                ax[num_divisions].plot(mean_xs, mean_ys, color=colours[q], lw=1)
    return all_xs, all_ys, formatted_data


def plot_psycho_types_formatted_data(formatted_data, key, sort_by='trial type', ax=None,  filter_by=None, filter_value=None, colourmap=matplotlib.cm.viridis, plot_means=True, align_end=True):

    formatted_data = formatted_data.sort_values(by=sort_by, ascending=True, ignore_index=True)
    num_divisions = formatted_data[sort_by].unique().shape[0]
    colours = colourmap(np.linspace(0, 1, num_divisions))
    if ax is None:
        fig, ax = plt.subplots()
    #fig, keys = plt.subplots(1, num_divisions + 1, sharex=True, sharey=True, figsize=(7, 2))
    all_xs = []
    all_ys = []
    for q in range(0, num_divisions):
        x_s = []
        y_s = []
        lengths = []
        peaks = []
        quantile = formatted_data[sort_by].unique()[q]
        quantile_data = formatted_data.loc[formatted_data[sort_by] == quantile]
        if filter_by:
            filtered_data = quantile_data.loc[quantile_data[filter_by] == filter_value]

        else:
            filtered_data = quantile_data
        for i, row in filtered_data.iterrows():
            if type(key) == str:
                x = row[key]
                if not plot_means:
                    ax[q].plot(x, color=colours[q])
                    ax[num_divisions].plot(x, color=colours[q])
                if not np.isnan(x[0]):
                    x_s.append(x)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)
            else:
                y = row[key[1]]
                x = row[key[0]]
                if not plot_means:
                    ax[q].plot(x, y, color=colours[q])
                    ax[num_divisions].plot(x, y, color=colours[q])
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
                #x_array[i, 0: lengths[i]] = x_s[i]
                if align_end:
                    x_array[i, max(lengths) - lengths[i]:] = np.abs(x_s[i])
                else:
                    x_array[i, 0: lengths[i]] = x_s[i]
            mean_xs = np.mean(x_array, axis=0)
            mean_xs = mean_xs[np.logical_not(np.isnan(mean_xs))]
            all_xs.append(mean_xs)
            all_xs.append(mean_xs)
            if plot_means:
                if key == "traces":
                    time_points = (np.arange(len(mean_xs)) / 10000) - (len(mean_xs) / 10000 / 2)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
                else:
                    if align_end:
                        time_points = (np.flip(np.arange(len(mean_xs)) * -1) / 30)
                    else:
                        time_points = (np.arange(len(mean_xs)) / 30)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
        else:
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
