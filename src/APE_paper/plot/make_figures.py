import numpy as np
import matplotlib.pyplot as plt
import math
from APE_paper.utils import plot_utils
from APE_paper.utils.misc_utils import update_progress


def make_figure_performance_trials_animals_bin(df_to_plot):
    
    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    fig, axs = plt.subplots(math.ceil(num_ans / 3), 3,
                            figsize=(15, num_ans),
                            facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.axis('off')
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal]
        linetp = [df_to_plot[df_to_plot.AnimalID == animal]["TrialIndexBinned200"] + 100,
                  #trick to align as CurrentPastPerformance looks at the past
                  100 * df_to_plot[df_to_plot.AnimalID == animal]["FirstPokeCorrect"]]

        ec = df_to_plot[df_to_plot.AnimalID == animal].ExperimentalGroup.unique()[0]
        ax_title = ec + ': ' + animal

        plot_utils.plot_trials_over_learning(ax, data, linetp, ax_title)

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)

    return fig

