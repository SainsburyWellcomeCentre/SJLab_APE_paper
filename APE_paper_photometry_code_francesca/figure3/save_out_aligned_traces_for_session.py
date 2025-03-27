from utils.post_processing_utils import get_all_experimental_records, add_experiment_to_aligned_data


if __name__ == '__main__':
    mouse_ids = ['SNL_photo57']
    date = '20211103'
    for mouse_id in mouse_ids:
        all_experiments = get_all_experimental_records()
        if (mouse_id =='all') & (date == 'all'):
            experiments_to_process = all_experiments
        elif (mouse_id == 'all') & (date != 'all'):
            experiments_to_process = all_experiments[all_experiments['date'] == date]
        elif (mouse_id != 'all') & (date == 'all'):
            experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
        elif (mouse_id != 'all') & (date != 'all'):
            experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
        add_experiment_to_aligned_data(experiments_to_process, outcome=False)