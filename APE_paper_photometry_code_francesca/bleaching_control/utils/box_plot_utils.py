import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from utils.plotting_visuals import makes_plots_pretty
from utils.plotting import output_significance_stars_from_pval
from scipy.stats import ttest_ind

import numpy as np
import seaborn as sns


def make_box_plot(df, fig_ax, dx='model', dy='explained variance', ort="v",
                  pal=['#E95F32', '#002F3A', '#F933FF', '#F933FF'], set_ylims=False, label=None, scatter_size=4):
    """
    Creates a box plot with scatter points for data visualization.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        fig_ax (matplotlib.axes._subplots.AxesSubplot): Axes object to draw the plot on.
        dx (str, optional): Column name for the x-axis. Default is 'model'.
        dy (str, optional): Column name for the y-axis. Default is 'explained variance'.
        ort (str, optional): Orientation of the box plot. 'v' for vertical, 'h' for horizontal. Default is 'v'.
        pal (list, optional): List of color codes for the palette. Default palette provided.
        set_ylims (bool, optional): Whether to set y-axis limits. Default is False.
        label (str, optional): Text label to be placed on the plot. Default is None.
        scatter_size (int, optional): Size of the scatter points. Default is 4.

    Returns:
        None
    """
    custom_palette = sns.set_palette(sns.color_palette(pal))
    keys = df[dx].unique()
    for i, key in enumerate(keys):
        data = df[df[dx] == key]
        noise = np.random.normal(0, 0.04, data.shape[0])
        fig_ax.scatter((data[dx].values == key).astype(int) * i + noise - 0.3, data[dy].values, color=pal[i], s=scatter_size,
                       alpha=0.6)

    sns.boxplot(x=dx, y=dy, data=df, palette=custom_palette, width=.3, zorder=10, linewidth=0.1,
                showcaps=True, boxprops={"zorder": 10, 'alpha': .9},
                showfliers=False, whiskerprops={'linewidth': 0.5, "zorder": 10},
                saturation=1, orient=ort, ax=fig_ax,
                medianprops={'color': 'white', 'linewidth': 1})
    if set_ylims:
        fig_ax.set_ylim([-2, np.max(df[dy]) + 2])
    if label:
        fig_ax.text(0.5, 1, label, transform=fig_ax.get_xaxis_transform(), size=8, ha='center')


def plot_and_save_comparison(data_df, ylabel, fig_directory, filename, dx='site', pal = ['#E95F32', '#002F3A'], sig_test=True, save=True):
    """
    Plots a comparison box plot and saves it as a PDF file.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data to plot.
        ylabel (str): Label for the y-axis.
        filename (str): Name of the output PDF file.
        data_dict (dict): Dictionary containing the data for comparison.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=[2, 2.5])
    make_box_plot(data_df, ax, dx=dx, dy=ylabel, pal=pal)

    # Calculate and plot significance stars
    if sig_test:
        VS_data = data_df[data_df[dx] == 'VS']
        TS_data = data_df[data_df[dx] == 'TS']
        p_val = ttest_ind(VS_data[ylabel].values, TS_data[ylabel].values).pvalue
        print(p_val)
        y = data_df[ylabel].max() + 0.04 * data_df[ylabel].max()

        ax.plot([0, 0, 1, 1], [y, y, y, y], c='k', lw=0.5)
        significance_stars = output_significance_stars_from_pval(p_val)
        ax.text(0.5, y + 0.01 * data_df[ylabel].max(), significance_stars, ha='center', fontsize=11)

    makes_plots_pretty(ax)
    plt.tight_layout()
    if save:
        plt.savefig(fig_directory + filename, transparent=True, bbox_inches='tight')
