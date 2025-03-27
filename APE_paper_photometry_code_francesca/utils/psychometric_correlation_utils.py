import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels as sm


def convert_lateral(data):
    return data['percentage high tones'] if data['fiber side'] == 'right' else round(1 - data['percentage high tones'], 2)


def convert_ipsi_contra(data):
    return data['contra sensory evidence'] if data['side'] == 'contra' else round(1 - data['contra sensory evidence'], 2)


def convert(trial_type):
    percentage_high_tones = {
    1.: .98,
    2.: .82,
    3.: .66,
    4.: .50,
    5.: .34,
    6.: .18,
    7.: .02}
    return percentage_high_tones[trial_type]


def convert_difficulty_non_sided(sensory_evidence):
    return np.round(np.abs(0.5 - sensory_evidence)/0.5, 2)


def prepare_data_for_regression(df):
    df['percentage high tones'] = df['trial type'].apply(convert)
    df['contra sensory evidence'] = df.apply(convert_lateral,axis=1)
    df['choice sensory evidence'] = df.apply(convert_ipsi_contra,axis=1)
    df['discriminatability'] = df['contra sensory evidence'].apply(convert_difficulty_non_sided)
    return df


def get_stats(df, regression_param='choice sensory evidence', outcome=1):
    mouse_ids = df['mouse'].unique()
    results = []
    for mouse in mouse_ids:
        mouse_data = df[(df['mouse'] == mouse) & (df.outcome==outcome)]
        mouse_data = mouse_data.dropna(axis=0)
        Y = mouse_data['norm APE'].reset_index(drop=True)
        X = sm.add_constant(np.abs(mouse_data[regression_param]).reset_index(drop=True))
        #X = pd.concat([np.abs(mouse_data[key]),mouse_data['trial type']], axis=1).reset_index(drop=True)
        model = sm.OLS(Y, X)
        result = model.fit()
        results.append(result)
    return results


def convert_difficulty(sensory_evidence):
    return np.round(-(0.5 - sensory_evidence)/0.5, 2)


def convert_ipsi_contra_numeric(side):
    mapping = {'ipsi': 0, 'contra':1}
    return mapping[side]


def convert_lateral_previous(data):
    return data['previous percentage high tones'] if data['fiber side'] == 'right' else round(1 - data['previous percentage high tones'], 2)


def convert_lateral_next(data):
    return data['next percentage high tones'] if data['fiber side'] == 'right' else round(1 - data['next percentage high tones'], 2)


def convert_ipsi_contra_next_trial(data):
    return data['next contra sensory evidence'] if data['side'] == 'contra' else round(1 - data['next contra sensory evidence'], 2)


def convert_ipsi_contra_last_trial(data):
    return data['previous contra sensory evidence'] if data['side'] == 'contra' else round(1 - data['previous contra sensory evidence'], 2)


def categorise_da_responses(response, cutoff=.65):
    return pd.qcut(response, [0., cutoff, 1.], labels=['low','high'])


def get_diff_psychometrics_da_test(df):
    """
    # the code below loops through mice and, for each mouse, subtracts lowcontra-lowipsi, highcontra-highipsi
    # then, the results are written to a new dataframe (where we also write mouse name), and concatenated for all mice
    """
    mean_per_mouse = df.groupby(['mouse', 'next contra sensory evidence',
                                     'DA response size',
                                     'side'])['next numeric side'].apply(np.mean).reset_index()

    diff_data = []
    for mouse in df.mouse.unique():
        mousedf = mean_per_mouse[mean_per_mouse.mouse==mouse]

        high_ipsi = ((mousedf['side'] == 'ipsi') & (mousedf['DA response size'] == 'high'))
        high_contra =  (mousedf['side'] == 'contra') & (mousedf['DA response size'] == 'high')
        low_ipsi = (mousedf['side'] == 'ipsi') & (mousedf['DA response size'] == 'low')
        low_contra =  (mousedf['side'] == 'contra') & (mousedf['DA response size'] == 'low')

        high_diff = mousedf[high_contra]['next numeric side'].reset_index(drop=True) - mousedf[high_ipsi]['next numeric side'].reset_index(drop=True)
        low_diff = mousedf[low_contra]['next numeric side'].reset_index(drop=True) - mousedf[low_ipsi]['next numeric side'].reset_index(drop=True)


        new_df = pd.DataFrame()
        new_df['next contra sensory evidence'] = mousedf[low_contra]['next contra sensory evidence'].reset_index(drop=True)
        new_df['mouse'] = mouse
        new_df['highdiff'] = high_diff
        new_df['lowdiff'] = low_diff
        diff_data.append(new_df)

    all_diffs = pd.concat(diff_data)
    return all_diffs
