"""In this module, we save only the relevant data from the tail dataframes to a new csv file.
"""
from set_global_params import processed_data_path
import pandas as pd
import os

# some defaults
site = 'Nacc'
alignment = 'reward'

# Load data
save_path = os.path.join(processed_data_path, 'psychometric_data')

nacc_file = os.path.join(save_path,"all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned_pk5.pkl".format(site, alignment))
print('Loading data for nacc...')
nacc_data = pd.read_pickle(nacc_file)[['mouse', 'session', 'fiber side', 'trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'stay or switch']].dropna().reset_index(drop=True)

print('Saving only necessary data to new pickle...')
nacc_data.to_csv(save_path + '/' + 'nacc_data_for_paper.csv')