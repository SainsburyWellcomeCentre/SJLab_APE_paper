"""
Figure 1 C
==============================================================
recreate figure 1C from the paper

"""

#%%

# This notebook compares the psychometric performance of mice on the cloud of tones task,
# according to muscimol injections in the tail of the striatum and dorsomedial striatum

# Commented out IPython magic to ensure Python compatibility.
# run this on Colab
# !rm -rf APE_paper/
# !git clone https://github.com/HernandoMV/APE_paper.git
# %pip install mouse-behavior-analysis-tools
# %cd APE_paper/docs/figures_notebooks

#%%
# 1. Import libraries
# ~~~~~~~~~~~~~~~~~~~~
# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
import pandas as pd
import urllib.request
from os.path import exists

from mouse_behavior_analysis_tools.plot import make_figures
from mouse_behavior_analysis_tools.utils import custom_functions as cuf
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Download data
# ~~~~~~~~~~~~~~~~~

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
# 3. Analyse the psychometric performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Analyse the different conditions
muscimol_conditions = ['StrTail-bilat-Control', 'StrTail-bilat', 'DMS-bilat']
# specify the text for the figure (this is the text that will appear in the legend)
text_for_figure = ['Saline (str tail)', 'Muscimol (str tail)', 'Muscimol (DMS)']
PP_array = []
for muscond in muscimol_conditions:
    # process the psychometric performance
    predictDif, PsyPer, fakePredictions, predictPer, EB = \
    cuf.PP_ProcessExperiment(mus_df[mus_df['Muscimol'] == muscond], 500, error_bars='SessionTime')
    # save to array
    PP_array.append([predictDif, PsyPer, fakePredictions, predictPer, EB])

clear_output()

#%%
# 4. Create **Figures 1 C** from the paper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
#%%
# 5. See a summary of the training and performance of the mice
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see a summary of the dataset
fig = make_figures.make_figure_muscimol_sessions_overview(mus_df)
fig
# uncomment here to save the plot
# data_directory = ''
# plt.savefig(data_directory + 'Muscimol-sessions-per-mouse.pdf',
#             transparent=True, bbox_inches='tight')