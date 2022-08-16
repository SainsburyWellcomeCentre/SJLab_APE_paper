import numpy as np
import matplotlib.pyplot as plt
import math
from APE_paper.utils import plot_utils
from APE_paper.utils.misc_utils import update_progress
from APE_paper.utils import model_utils


def make_figure_performance_trials_animals_bin(df_to_plot):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    fig, axs = plt.subplots(math.ceil(num_ans / 3), 3,
                            figsize=(15, num_ans),
                            facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis('off')
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal]
        linetp = [df_to_plot[df_to_plot.AnimalID == animal]["TrialIndexBinned200"] + 100,
                  # trick to align as CurrentPastPerformance looks at the past
                  100 * df_to_plot[df_to_plot.AnimalID == animal]["FirstPokeCorrect"]]

        ec = df_to_plot[df_to_plot.AnimalID == animal].ExperimentalGroup.unique()[0]
        ax_title = ec + ': ' + animal

        plot_utils.plot_trials_over_learning(ax, data, linetp, ax_title)
        ax.get_legend().remove()

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)

    return fig


def make_figure_performance_trials_animals_model(df_to_plot, fit_df, der_max_dir):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    x = np.linspace(1, 5000)

    fig, axs = plt.subplots(math.ceil(num_ans / 3), 3,
                            figsize=(15, num_ans),
                            facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis('off')
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID==animal][
                                                      ['CumulativeTrialNumberByProtocol',
                                                       'CurrentPastPerformance100',
                                                       'SessionID']
                                                      ].dropna()
        linetp = [x,
                  model_utils.sigmoid_func_sc(x, *[fit_df[fit_df.AnimalID==animal].maximum_performance.iloc[0],
                                            fit_df[fit_df.AnimalID==animal].slope.iloc[0],
                                            fit_df[fit_df.AnimalID==animal].bias.iloc[0]])]

        ec = df_to_plot[df_to_plot.AnimalID == animal].ExperimentalGroup.unique()[0]
        ax_title = ec + ': ' + animal

        plot_utils.plot_trials_over_learning(ax, data, linetp, ax_title)
        ax.get_legend().remove()
        
        # point to the maximum slope
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        ymin, ymax = ax.get_ybound()
        perc_max_slope = model_utils.sigmoid_func_sc(der_max_dir[animal][0],
                                        *[fit_df[fit_df.AnimalID==animal].maximum_performance.iloc[0],
                                        fit_df[fit_df.AnimalID==animal].slope.iloc[0],
                                        fit_df[fit_df.AnimalID==animal].bias.iloc[0]])
        ax.axvline(fit_df[fit_df.AnimalID==animal].bias.iloc[0], 0,
                   (perc_max_slope - ymin) / (ymax - ymin), linestyle='--', color='k')
        ax.plot([0, fit_df[fit_df.AnimalID==animal].bias.iloc[0]], [perc_max_slope, perc_max_slope], 'k--')

        plt.tight_layout()

        update_progress(counter / num_ans)
        
    update_progress(1)
    return fig


def make_figure_learning_parameters_between_groups(fit_df, parameters_to_show, titles, ylabs,
                                                   pvals, sig_levels, color_palette, hue_order,
                                                   yvals):

    spread = .3

    fig, axs = plt.subplots(ncols=len(parameters_to_show), nrows=1, sharey=False, figsize=(6, 4))
    axs = axs.ravel()
    for i, var in enumerate(parameters_to_show):

        plot_utils.plot_swarm_and_boxplot(fit_df, var, axs[i], hue_order, spread, color_palette)

        axs[i].set_title(titles[i])
        axs[i].set_ylabel(ylabs[i])

    for ax in axs:

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Only show ticks on the left
        #ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks([])
        ax.xaxis.set_visible(False)
        ax.get_legend().remove()

    # add statistics
    for i, ax in enumerate(axs):
        n_ast = sum(pvals[i] < sig_levels)
        plot_utils.add_stats(ax, xvals=(spread / 2, 1 + spread / 2),
                             yval=yvals[i], n_asterisks=n_ast)

    plt.tight_layout()

    return fig


def make_figure_performance_trials_animals_biased_trials(df_to_plot, bias_mask):
    
    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    fig, axs = plt.subplots(math.ceil(num_ans / 3), 3,
                            figsize=(15, num_ans),
                            facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis('off')
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal]

        ec = df_to_plot[df_to_plot.AnimalID == animal].ExperimentalGroup.unique()[0]
        ax_title = ec + ': ' + animal

        plot_utils.plot_trials_over_learning(ax, data, line_to_add=False, axtitle=ax_title,
                                             override_hue='red')
        plot_utils.plot_trials_over_learning(ax, data[bias_mask], line_to_add=False,
                                             axtitle=False,
                                             override_hue='green')

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)
    
    return fig

