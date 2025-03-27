from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from utils.kernel_regression.linear_regression_utils import *
from tqdm import tqdm
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces, reproduce_figures_path, spreadsheet_path
from utils.kernel_regression.regression_plotting_utils import calculate_significance_windows, organise_data_means_with_rtc, plot_kernels, organise_data_means
import os
import shutil

def plot_kernels_for_site(move_axs, cue_axs, reward_axs, return_axs, means, sems, time_stamps, palette=['#E95F32', '#F9C0AF'], legend=False):
    plot_kernels(move_axs, 'ipsi choices', means, sems, time_stamps, colour=palette[1], legend=legend)
    plot_kernels(move_axs, 'contra choices', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(cue_axs, 'ipsi cues', means, sems, time_stamps, colour=palette[1], legend=legend)
    plot_kernels(cue_axs, 'contra cues', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(reward_axs, 'rewards', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(reward_axs, 'no rewards', means, sems, time_stamps, colour=palette[1], legend=legend)
    plot_kernels(return_axs, 'contra returns', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(return_axs, 'ipsi returns', means, sems, time_stamps, colour=palette[1], legend=legend)


def get_regression_data_for_plot(recording_site='tail', reg_type='_return_to_centre_trimmed_traces_300frames_long_turns'):
    experiment_record = pd.read_csv(experiment_record_path, dtype=str)
    experiment_record['date'] = experiment_record['date'].astype(str)

    mouse_ids = mice_average_traces[recording_site]
    mouse_ids.remove('SNL_photo57') # we only have two sessions with tracking for this mouse so we have to process it after (see below)
    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_unsuitable_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (
                    clean_experiments['recording_site'] == recording_site)].reset_index(drop=True)

    all_experiments_to_process = all_experiments_to_process[
        (all_experiments_to_process['include return to centre'] != 'no') & (
                    all_experiments_to_process['include'] != 'no')].reset_index(
        drop=True)
    experiments_to_process1 = get_first_x_sessions(all_experiments_to_process).reset_index(drop=True)

    mouse_ids2 = ['SNL_photo57']
    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids2)
    clean_experiments = remove_unsuitable_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids2)) & (
                    clean_experiments['recording_site'] == recording_site)].reset_index(drop=True)

    all_experiments_to_process = all_experiments_to_process[
        (all_experiments_to_process['include return to centre'] != 'no') & (
                    all_experiments_to_process['include'] != 'no')].reset_index(
        drop=True)
    experiments_to_process2 = get_first_x_sessions(all_experiments_to_process, x=2).reset_index(drop=True)
    experiments_to_process = pd.concat([experiments_to_process1, experiments_to_process2])
    ipsi_choice_kernels = []
    contra_choice_kernels = []
    ipsi_cue_kernels = []
    contra_cue_kernels = []
    reward_kernels = []
    no_reward_kernels = []
    if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        contra_return_kernels = []
        ipsi_return_kernels = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        data_dir = os.path.join(processed_data_path, mouse)
        repro_data_dir = os.path.join(reproduce_figures_path, 'ED_fig4', mouse)
        if not os.path.exists(repro_data_dir):
            os.makedirs(repro_data_dir)
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        mouse_ipsi_choice_kernel = []
        mouse_contra_choice_kernel = []
        mouse_ipsi_cue_kernel = []
        mouse_contra_cue_kernel = []
        mouse_reward_kernel = []
        mouse_no_reward_kernel = []
        if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
            mouse_contra_return_kernel = []
            mouse_ipsi_return_kernel = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'linear_regression_kernels{}.p'.format(reg_type)
            if not os.path.exists(os.path.join(repro_data_dir, filename)):
                shutil.copy(os.path.join(data_dir, filename), os.path.join(repro_data_dir, filename))

            fiber_side = df[df.date == date]['fiber_side'].iloc[0]
            if fiber_side == 'left':
                ipsi_cue = 'high cues'
                contra_cue = 'low cues'
            else:
                ipsi_cue = 'low cues'
                contra_cue = 'high cues'

            with open(os.path.join(repro_data_dir, filename), 'rb') as f:
                session_kernels = pickle.load(f)
                mouse_ipsi_choice_kernel.append(session_kernels['kernels']['ipsi choices'])
                mouse_contra_choice_kernel.append(session_kernels['kernels']['contra choices'])
                mouse_ipsi_cue_kernel.append(session_kernels['kernels'][ipsi_cue])
                mouse_contra_cue_kernel.append(session_kernels['kernels'][contra_cue])
                mouse_reward_kernel.append(session_kernels['kernels']['rewards'])
                mouse_no_reward_kernel.append(session_kernels['kernels']['no rewards'])
                if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
                    mouse_contra_return_kernel.append(session_kernels['kernels']['contra returns'])
                    mouse_ipsi_return_kernel.append(session_kernels['kernels']['ipsi returns'])

        ipsi_choice_kernels.append(np.mean(mouse_ipsi_choice_kernel, axis=0))
        contra_choice_kernels.append(np.mean(mouse_contra_choice_kernel, axis=0))
        ipsi_cue_kernels.append(np.mean(mouse_ipsi_cue_kernel, axis=0))
        contra_cue_kernels.append(np.mean(mouse_contra_cue_kernel, axis=0))
        reward_kernels.append(np.mean(mouse_reward_kernel, axis=0))
        no_reward_kernels.append(np.mean(mouse_no_reward_kernel, axis=0))
        if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
            contra_return_kernels.append(np.mean(mouse_contra_return_kernel, axis=0))
            ipsi_return_kernels.append(np.mean(mouse_ipsi_return_kernel, axis=0))

    ipsi_choice_kernels = np.array(ipsi_choice_kernels)
    contra_choice_kernels = np.array(contra_choice_kernels)
    ipsi_cue_kernels = np.array(ipsi_cue_kernels)
    contra_cue_kernels = np.array(contra_cue_kernels)
    reward_kernels = np.array(reward_kernels)
    no_reward_kernels = np.array(no_reward_kernels)
    if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        contra_return_kernels = np.array(contra_return_kernels)
        ipsi_return_kernels = np.array(ipsi_return_kernels)

    time_stamps = {}

    time_stamps['ipsi choices'] = session_kernels['shifts']['ipsi choices'] / 10000 * 100
    time_stamps['contra choices'] = session_kernels['shifts']['contra choices'] / 10000 * 100
    time_stamps['ipsi cues'] = session_kernels['shifts']['high cues'] / 10000 * 100
    time_stamps['contra cues'] = session_kernels['shifts']['low cues'] / 10000 * 100
    time_stamps['rewards'] = session_kernels['shifts']['rewards'] / 10000 * 100
    time_stamps['no rewards'] = session_kernels['shifts']['no rewards'] / 10000 * 100
    if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        time_stamps['contra returns'] = session_kernels['shifts']['contra returns'] / 10000 * 100
        time_stamps['ipsi returns'] = session_kernels['shifts']['ipsi returns'] / 10000 * 100
        means, sems = organise_data_means_with_rtc(ipsi_choice_kernels, contra_choice_kernels, ipsi_cue_kernels, contra_cue_kernels,
                                        reward_kernels, no_reward_kernels, contra_return_kernels, ipsi_return_kernels)
        sh_path = os.path.join(spreadsheet_path, 'ED_fig4')
        only_plotted_kernels = [contra_return_kernels, ipsi_return_kernels]
        plotted_keys = ['contra returns', 'ipsi returns']
        only_plotted_time_stamps = {key: time_stamps[key] for key in plotted_keys if key in time_stamps}
        for kernels, ts in zip(only_plotted_kernels, only_plotted_time_stamps.items()):
            n_mice, samples = kernels.shape
            nm = ts[0]
            df = pd.DataFrame()
            df['time'] = ts[1]
            for m in range(n_mice):
                df[f'm{m}'] = kernels[m]
            kernel_fn = f'fig4L_kernels_{recording_site}_{nm}.csv'
            if not os.path.exists(os.path.join(sh_path, kernel_fn)):
                df.to_csv(os.path.join(sh_path, kernel_fn))
    else:
        organise_data_means(ipsi_choice_kernels, contra_choice_kernels, ipsi_cue_kernels, contra_cue_kernels,
                                     reward_kernels, no_reward_kernels)

    # new part where I calculate significance windows for 0.1s bins
    significant_time_bins = {}
    significant_time_stamps, p_choice = calculate_significance_windows(contra_choice_kernels, ipsi_choice_kernels,
                                                                       'ipsi choices', time_stamps)
    significant_time_bins['choice'] = significant_time_stamps

    # ipsi contra cues
    significant_time_stamps, p_cue = calculate_significance_windows(contra_cue_kernels, ipsi_cue_kernels,
                                                                    'ipsi cues', time_stamps)
    significant_time_bins['cue'] = significant_time_stamps

    # reward no reward
    significant_time_stamps, p_outcome = calculate_significance_windows(reward_kernels, no_reward_kernels,
                                                                        'rewards', time_stamps)
    significant_time_bins['outcome'] = significant_time_stamps

    if reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
    # returns ipsi contra
        significant_time_stamps, p_outcome = calculate_significance_windows(contra_return_kernels, ipsi_return_kernels,
                                                                            'ipsi returns', time_stamps)
        significant_time_bins['returns'] = significant_time_stamps

    return time_stamps, means, sems, significant_time_bins
