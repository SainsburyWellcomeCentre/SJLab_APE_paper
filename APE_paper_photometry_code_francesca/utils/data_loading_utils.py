import os
import pickle
from utils.zscored_plots_utils import get_example_data_for_recording_site


def load_or_get_and_save_example_heatmap_aligned_to_keys(pickle_path, recording_site, keys_for_aligning, time_window_size=2.):
    """If a pickled cache of the necessary data exists, load heatmap data from pickle. If not, create this cache,
    save from pickle and return heatmap data.

    Args:
        pickle_path (str): location of pickle on disk
        recording_site (str): TS or VS
        keys_for_aligning (list): List of strings stating what the data should be aligned to, e.g. ipsi for ipsilateral
        movement.
        time_window_size (float): Size of time window to save out.

    Returns:

    """
    if not (recording_site == 'TS' or recording_site == 'VS'):
        return ValueError('Recording site has to be VS or TS')

    # Try to load data from pickle
    if os.path.exists(pickle_path):
        print(f"Loading cached {recording_site} data from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            cache = pickle.load(f)
            data_ = cache['data']
            wd = cache['wd']
            flip_sort_order = cache['flip_sort_order']
            y_mins = cache['y_mins']
            y_maxs = cache['y_maxs']
    else:  # if not, load from raw
        # Get data and save to pickle
        print(f"Loading {recording_site} data from raw and saving to cache...")
        pickle_folder, _ = os.path.split(pickle_path)
        if not os.path.exists(pickle_folder):
            os.makedirs(pickle_folder)

        data_, wd, flip_sort_order, y_mins, y_maxs = get_example_data_for_recording_site(recording_site,
                                                                                         keys_for_aligning)
        for obj in data_:
            bool_idx = (obj.time_points < time_window_size) & (obj.time_points >= -time_window_size)
            obj.sorted_traces = obj.sorted_traces[:, bool_idx]
            obj.time_points = obj.time_points[bool_idx]

        # Save to pickle
        cache = {
            'data': data_,
            'wd': wd,
            'flip_sort_order': flip_sort_order,
            'y_mins': y_mins,
            'y_maxs': y_maxs
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump(cache, f)
    return data_, wd, flip_sort_order, y_mins, y_maxs