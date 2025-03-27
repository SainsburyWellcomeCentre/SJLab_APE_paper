from utils.tracking_analysis.head_angle_plotting_functions import *
from set_global_params import change_over_time_mice

save = False
load_saved = False
for site in ['Nacc', 'tail']:
    mouse_ids = change_over_time_mice[site]
    data_to_save, all_data = get_first_three_sessions_dlc(mouse_ids, site, save=save, load_saved=load_saved)