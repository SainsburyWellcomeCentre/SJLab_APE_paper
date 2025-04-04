Code:

Datasets: 

# APE_paper 
Reproduction of figures for the [APE paper](https://www.biorxiv.org/content/10.1101/2022.09.12.507572v1)

<p align="center">
  <img src="docs/imgs/summary.png" width=450>
</p>
Francesca Greenstreet*, Hernando Martinez Vergara*, Yvonne Johansson*, Sthitapranjya Pati, Laura Schwarz, Stephen C Lenzi, Jesse P. Geerts, Matthew Wisdom1, Alina Gubanova, Lars Rollik, Jasvin Kaur, Theodore Moskovitz, Joseph Cohen, Emmett Thompson, Troy W Margrie, Claudia Clopath, Marcus Stephenson-Jones, "Dopaminergic action prediction errors serve as a value-free teaching signal." 


### See documentation (scripts to generate figures) [here](https://hernandomv.github.io/APE_paper/) and 'Documentation for code.pdf' file in APE_paper_photometry_code_francesca folder of this repository.

# To run analysis on photometry data (except for panels ED fig 5 O-Y and ED fig 12):
See the documentation provided in Documentation for code.pdf' file in APE_paper_photometry_code_francesca folder of this repository.

All modeling work can be found in folder 'models'.

# To run manipulations, lesion and behavioral analysis as well as photometry for panels ED fig 5 O-Y and ED fig 12:

## To run things locally in your machine:

1. clone this repo

```
git clone https://github.com/HernandoMV/APE_paper.git
```

2. create the environment (optional). Otherwise you might need to install some dependencies manually.

    ```
    conda env create -f environment.yml
    conda activate APE_paper
    ```

3. install the tool to analyse the behavior
    ### Option 1: from PyPi
    ```
    pip install mouse-behaviour-analysis-tools
    ```

    ### Option 2: from the tools repo
    https://github.com/HernandoMV/mouse-behavior-analysis-tools

    With this option it will be easier for you to change things from the functions directly,
    such as plotting options.

    3.2.1. clone the repo

    ```
    git clone https://github.com/HernandoMV/mouse-behavior-analysis-tools.git
    ```

    3.2.2. change the current directory to that of the cloned repo

    ```
    cd mouse-behaviour-analysis-tools
    ```

    3.2.3. in the root directory of the cloned repo type

    ```
    pip install -e .
    ```

4. run the scripts in `scripts` folder, or download them in script or notebook format from the [documentation](https://hernandomv.github.io/APE_paper/)


# Run the Google Colab notebooks directly in your browser:

[Figure 1 C](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_1_C_Nature.ipynb)
<p align="left">
  <img src="docs/imgs/fig1c.png" width=250>
</p>

[Figure 1 EF](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_1_EF_Nature.ipynb)
<p align="left">
  <img src="docs/imgs/fig1ef.png" width=450>
</p>

[Figure 1 HIJLMN](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_1_HIJLMN_Nature.ipynb)
<p align="left">
  <img src="docs/imgs/fig1ijk.png" width=450>
</p>

[Figure 4 CDF 4mW data](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_4_DF_Nature.ipynb)
<p align="left">
  <img src="docs/imgs/fig4d.png" width=200>
  <img src="docs/imgs/fig4f.png" width=210>
</p>

[Figure 5 FG](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_5_FG_Nature.ipynb)
<p align="left">
  <img src="docs/imgs/fig5fg.png" width=450>
</p>

[ED 5 PQR](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_S5_TU.ipynb)
<p align="left">
  <img src="docs/imgs/ED5pqr.png" width=450>
</p>

[ED 5 ST](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_S5_TU.ipynb)
<p align="left">
  <img src="docs/imgs/ED5st_EarlyTraining.png" width=450>
  <img src="docs/imgs/ED5t_1stCueSession.png" width=450>
</p>

[ED 5 VW](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_S5_TU.ipynb)
<p align="left">
  <img src="docs/imgs/ED5vw.png" width=450>
</p>

[ED 5 XY](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_S5_TU.ipynb)
<p align="left">
  <img src="docs/imgs/ED5xy.png" width=450>
</p>

[ED 12 FG](https://colab.research.google.com/github/HernandoMV/APE_paper/blob/main/docs/figures_notebooks/Figure_S5_TU.ipynb)
<p align="left">
  <img src="docs/imgs/ED12fg.png" width=450>
</p>