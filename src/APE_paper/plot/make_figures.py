import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

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


def make_figure_differences_performance_between_groups(df_to_plot, col_to_plot, hue_order, color_palette):
    data_mean = df_to_plot.groupby(['CumulativeTrialNumberByProtocol','ExperimentalGroup'])[col_to_plot].mean().reset_index()
    st_err_mean = df_to_plot.groupby(['CumulativeTrialNumberByProtocol','ExperimentalGroup'])[col_to_plot].std().reset_index()
    data_mean['low_bound'] = data_mean[col_to_plot] - st_err_mean[col_to_plot]
    data_mean['high_bound'] = data_mean[col_to_plot] + st_err_mean[col_to_plot]

    fig1 = plt.figure(figsize=(8, 4))
    plt.axhline(50, ls='dotted', alpha=0.4, color='k')
    plt.axhline(100, ls='dotted', alpha=0.4, color='k')
    for i,eg in enumerate(hue_order):
        df = data_mean[data_mean.ExperimentalGroup==eg].copy()
        x = df.CumulativeTrialNumberByProtocol
        plt.plot(x, df[col_to_plot], color=color_palette[i], label=eg)
        y1 = df['low_bound']
        y2 = df['high_bound']
        plt.fill_between(x, y1, y2, where=y2 >= y1, color=color_palette[i], alpha=.2, interpolate=False)

    plt.ylabel(col_to_plot)
    plt.xlabel('trial number')
    plt.ylabel('task performance (%)')
    plt.legend(loc=(0.76,0.3), frameon=False)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # remove the legend as the figure has it's own
    ax.get_legend().remove()

    ax.set_xlim((0,5000))

    plt.title('Task learning progression')

    return fig1


def make_figure_differences_performance_significance(real_data_pd, pos_ci, neg_ci):
    fig2 = plt.figure(figsize=(8, 4))
    plt.axhline(0, ls='dotted', alpha=0.4, color='k')
    plt.plot(real_data_pd, color='k', label='observed data')
    plt.plot(pos_ci, linestyle='--', color='gray', label='95% ci')
    plt.plot(neg_ci, linestyle='--', color='gray')
    x = pos_ci.reset_index().TrialIndexBinned
    y1 = pos_ci.reset_index().Performance
    y2 = real_data_pd.reset_index().Performance
    plt.fill_between(x, y1, y2, where=y2 >= y1, facecolor='k', alpha=.2, interpolate=True)
    plt.ylabel('performance difference (%)')
    plt.xlabel('trial number')
    plt.legend(loc=(0.75,0.05), frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim((0,5000))

    return fig2


def make_figure_differences_performance_significance_global(real_data_pd, quants_to_test, shrdf, global_sig, nsh):
    fig = plt.figure(figsize=(16, 4))
    sns.lineplot(data=real_data_pd, color='r')
    for k,q in enumerate(quants_to_test):
        sns.lineplot(data=shrdf.groupby('TrialIndexBinned').quantile(q), color='k')
        sns.lineplot(data=shrdf.groupby('TrialIndexBinned').quantile((1 - q)), color='k')
        print('ci = ', q,
            '\tglobal pval = ',  np.sum(global_sig, axis=0)[k] / nsh,
            '\treal data significant ', any(np.logical_or(real_data_pd > shrdf.groupby('TrialIndexBinned').quantile(q),
            real_data_pd < shrdf.groupby('TrialIndexBinned').quantile(1 - q))))
            
    return fig