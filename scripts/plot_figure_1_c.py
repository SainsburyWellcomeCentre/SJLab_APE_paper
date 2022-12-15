"""
Figure 1 C
==============================================================
Explanation

"""

"""
Figure_1_C.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_1_C.ipynb
"""

# This notebook compares the psychometric performance of mice on the cloud of tones task,
# according to muscimol injections in the tail of the striatum and dorsomedial striatum

# Commented out IPython magic to ensure Python compatibility.
# run this on Colab
# !rm -rf APE_paper/
# !git clone https://github.com/HernandoMV/APE_paper.git
# %pip install mouse-behavior-analysis-tools
# %cd APE_paper/docs/figures_notebooks

#%%
# 1.2 Example title
# ~~~~~~~~~~~~~~~~~~~~
# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
import pandas as pd
import matplotlib.pylab as plt
import urllib.request
from os.path import exists

from mouse_behavior_analysis_tools.plot import make_figures
from mouse_behavior_analysis_tools.utils import custom_functions as cuf
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

#%%
# 1.2 Example title
# ~~~~~~~~~~~~~~~~~~~~

# download data from server
dataset_name = 'Muscimol-injections_dataframe.csv'
url = "https://zenodo.org/record/7261639/files/" + dataset_name
dataset_path = '../data/' + dataset_name
# download if data is not there
if not exists(dataset_path):
    print('Downloading data...')
    urllib.request.urlretrieve(url, dataset_path)
else:
    print('Data already in directory')
# load
mus_df = pd.read_csv(dataset_path, index_col=0)

#%%
# 1.2 Example title
# ~~~~~~~~~~~~~~~~~~~~

# see a summary of the dataset
fig = make_figures.make_figure_muscimol_sessions_overview(mus_df)
fig
# uncomment here to save the plot
# data_directory = ''
# plt.savefig(data_directory + 'Muscimol-sessions-per-mouse.pdf',
#             transparent=True, bbox_inches='tight')

#%%
# 1.2 Example title
# ~~~~~~~~~~~~~~~~~~~~

# Analyse the different conditions
muscimol_conditions = ['StrTail-bilat-Control', 'StrTail-bilat', 'DMS-bilat']
text_for_figure = ['Saline (str tail)', 'Muscimol (str tail)', 'Muscimol (DMS)']
PP_array = []
for muscond in muscimol_conditions:
    # process the psuchometric performance
    predictDif, PsyPer, fakePredictions, predictPer, EB = \
    cuf.PP_ProcessExperiment(mus_df[mus_df['Muscimol'] == muscond], 500, error_bars='SessionTime')
    # save to array
    PP_array.append([predictDif, PsyPer, fakePredictions, predictPer, EB])

clear_output()

#%%
# 1.2 Example title
# ~~~~~~~~~~~~~~~~~~~~

"""##### The following cell creates **Figures 1 C** from the paper"""

# prepare colors
colorlist = [(0.24715576, 0.49918708, 0.57655991),
             (160/255, 11/255 , 11/255),
             (46/255, 122/255 , 11/255)]

fig = make_figures.make_figure_muscimol_psychometric(PP_array, text_for_figure, colorlist)
fig
# uncomment here to save the plot
# data_directory = ''
# plt.savefig(data_directory + 'Figure1C.pdf',
#             transparent=True, bbox_inches='tight')
# %%
