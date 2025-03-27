from copy import deepcopy
import os
from utils.tracking_analysis.fede_load_tracking import prepare_tracking_data
from utils.tracking_analysis.fede_geometry import *
import pandas as pd
from utils.tracking_analysis.transformation_utils import projective_transform_tracks
import pickle
from set_global_params import running_in_box_dir, out_of_task_movement_mice_dates, running_in_box_tracking_dir


def get_movement_properties_for_session(mouse, date):
    # The camera is not directly above the box, it is at an angle so this transforms the tracking coordinates to fit the actual shape of the box
    file_path = os.path.join(running_in_box_tracking_dir, mouse, date, '{}_cameraDLC_resnet50_heading_angleMar23shuffle1_1030000.h5'.format(mouse))
    body_parts = ('nose', 'L_ear', 'R_ear', 'body', 'tail_base', 'tail_tip')
    tracking_data = prepare_tracking_data(
        tracking_filepath=file_path,
        tracking=None,
        bodyparts=body_parts,
        likelihood_th=0.9999,
        median_filter=True,
        filter_kwargs={'kernel_size': 7},
        compute=True,
        smooth_dir_mvmt=True,
        interpolate_nans=True,
        verbose=False)
    transformed_data = deepcopy(tracking_data)
    for key in transformed_data.keys():
        transformed_data[key]['x'], transformed_data[key]['y'] = projective_transform_tracks(tracking_data[key]['x'],
                                                                                               tracking_data[key]['y'],
                                                                                               np.array([[40, 258], [70, 105], [592, 258], [560,105]]),
                                                                                               np.array([[0, 240], [0, 0], [600, 240], [600, 0]]))


    midpoints_y = (transformed_data['L_ear']['y'] + transformed_data['R_ear']['y']) / 2
    midpoints_x = (transformed_data['L_ear']['x'] + transformed_data['R_ear']['x']) / 2
    head_angles = calc_angle_between_vectors_of_points_2d(transformed_data['nose']['x'].values,
                                                          transformed_data['nose']['y'].values, midpoints_x, midpoints_y)
    head_angular_velocity = calc_ang_velocity(head_angles)
    head_ang_accel = derivative(head_angular_velocity)
    speed = transformed_data['body']['speed'].values
    move_dir = transformed_data['body']['direction_of_movement'].values
    acceleration = derivative(speed)
    return transformed_data, tracking_data, head_angular_velocity, head_ang_accel, speed, move_dir, acceleration, head_angles


def rolling_zscore(x, window=10*10000):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def get_photometry_data(mouse, date):
    loading_folder = os.path.join(running_in_box_dir, 'processed_data', mouse)
    smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    clock_filename = mouse + '_' + date + '_' + 'clock.npy'
    photometry_data = np.load(os.path.join(loading_folder, smoothed_trace_filename))
    z_scored_data = rolling_zscore(pd.Series(photometry_data))
    clock_stamps = np.load(os.path.join(loading_folder, clock_filename))
    return z_scored_data.values[clock_stamps]


if __name__ == '__main__':
    for mouse, date_time in out_of_task_movement_mice_dates.items():

        date = date_time[0:8]
        tracking_data, untransformed_data, head_angular_velocity, head_ang_accel, speed, move_dir, acceleration, head_angles = get_movement_properties_for_session(mouse, date_time)

        photometry_data = get_photometry_data(mouse, date)
        save_dir = os.path.join(running_in_box_dir, 'processed_data')
        np.savez(os.path.join(save_dir, 'preprocessed_speed_by_neurons_transformed_tracking_{}.npz'.format(mouse)), speed=speed, acceleration=acceleration, photometry_data=photometry_data,
                 head_angular_velocity=head_angular_velocity, head_ang_accel=head_ang_accel, move_dir=move_dir, head_angles=head_angles, allow_pickle=True)
        file = open(os.path.join(save_dir, 'tracking_data_{}.p'.format(mouse)), 'wb')
        pickle.dump(tracking_data, file)
        file.close()

