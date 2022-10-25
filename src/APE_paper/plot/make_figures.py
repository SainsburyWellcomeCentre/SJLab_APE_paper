import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import random
from scipy import stats

from APE_paper.utils import plot_utils
from APE_paper.utils.misc_utils import update_progress
from APE_paper.utils import model_utils
from APE_paper.utils import custom_functions as cuf



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
        df = data_mean[data_mean.ExperimentalGroup == eg].copy()
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
    y1 = neg_ci.reset_index().Performance
    y2 = real_data_pd.reset_index().Performance
    plt.fill_between(x, y1, y2, where=y1 >= y2, facecolor='k', alpha=.2, interpolate=True)
    plt.ylabel('performance difference (%)')
    plt.xlabel('trial number')
    plt.legend(loc=(0.75, 0.05), frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim((0, 5000))

    return fig2


def make_figure_differences_performance_significance_global(real_data_pd, quants_to_test, shrdf, global_sig, nsh):
    fig = plt.figure(figsize=(16, 4))
    sns.lineplot(data=real_data_pd, color='r')
    for k, q in enumerate(quants_to_test):
        sns.lineplot(data=shrdf.groupby('TrialIndexBinned').quantile(q), color='k')
        sns.lineplot(data=shrdf.groupby('TrialIndexBinned').quantile((1 - q)), color='k')
        print('ci = ', q,
            '\tglobal pval = ',  np.sum(global_sig, axis=0)[k] / nsh,
            '\treal data significant ', any(np.logical_or(real_data_pd > shrdf.groupby('TrialIndexBinned').quantile(q),
            real_data_pd < shrdf.groupby('TrialIndexBinned').quantile(1 - q))))

    return fig


def make_figure_muscimol_sessions_overview(mus_df):
    # plot a summary of all the animals in the dataset
    fig, ax = plt.subplots(len(pd.unique(mus_df.AnimalID)), 1,
                           figsize=(7, 5 * len(pd.unique(mus_df.AnimalID))))
    axs = ax.ravel()
    fig.subplots_adjust(hspace=1.3)
    for i, animal in enumerate(pd.unique(mus_df.AnimalID)):
        aDF = mus_df[mus_df.AnimalID == animal]
        dfToPlot = plot_utils.summary_matrix(aDF)
        axs[i] = plot_utils.summary_plot(dfToPlot, aDF, axs[i], top_labels=['Muscimol'])

    return fig


def make_figure_muscimol_psychometric(PP_array, muscond_text, colorlist):
    fig = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k')
    ax = plt.gca()
    ax.hlines(50, 0, 100, linestyles='dotted' , alpha=0.4)

    for counter, results in enumerate(PP_array):
        predictDif, PsyPer, fakePredictions, predictPer, EB = results
        plot_utils.PlotPsychPerformance(dataDif=PsyPer['Difficulty'], dataPerf=PsyPer['Performance'],
                                        predictDif=predictDif, ax=ax, fakePred=fakePredictions,
                                        realPred=predictPer, label=muscond_text[counter], errorBars=EB,
                                        color=colorlist[counter])
    ax.axis('on')
    # remove some ticks
    ax.tick_params(which='both', top=False, bottom='on', left='on', right=False,
                   labelleft='on', labelbottom='on')

    L = plt.legend(loc='upper left', frameon=False)
    # L.get_texts()[0].set_text('Saline (str tail)')
    # L.get_texts()[1].set_text('Muscimol (str tail)')
    # L.get_texts()[2].set_text('Muscimol (DMS)')

    return fig


def make_figure_optoinhibition_after_learning_batch(random_opto_df):

    jitter = 0.3
    alpha = 1
    spread = jitter * 1.6
    mice_cohorts = ['D1opto', 'D2opto'] 
    colors = ['skyblue', 'olivedrab']
    labels_for_legend = ['D1-Arch', 'D2-Arch']

    fig, axs = plt.subplots(1, len(mice_cohorts), figsize = (4 * len(mice_cohorts), 8), sharey=True)

    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_title(labels_for_legend[i], fontsize=20)
        ax.set_xticks([])
        ax.set_xlim([-jitter*1.2, jitter*3])
        # get rid of the frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14) 
        
        xmin, _ = axs[0].get_xaxis().get_view_interval()
        ax.plot((xmin, xmin), (-60, 40), color='black', linewidth=1) 

    axs[0].set_ylabel('contralateral bias', fontsize=15)

    jit_list = []

    # plot stds
    for session in pd.unique(random_opto_df.SessionID):
        session_idx = random_opto_df.index[random_opto_df.SessionID == session].item()
        cohort = random_opto_df.loc[session_idx].Genotype
        ax = axs[mice_cohorts.index(cohort)]
        st_t_idx = 0
        sh_d = random_opto_df.loc[session_idx].contralateral_bias
        sh_std = random_opto_df.loc[session_idx].bias_std
        imp_jit = random.uniform(-jitter, jitter)
        x_pos = st_t_idx + imp_jit
        jit_list.append(x_pos)

        #stds
        ax.plot([x_pos, x_pos], [sh_d-sh_std, sh_d+sh_std],
                color=colors[mice_cohorts.index(cohort)], linewidth=3, alpha=alpha)

    counter = 0
    # plot means on top
    mean_vals = [[], []]
    sessions_used = [[], []]
    for session in pd.unique(random_opto_df.SessionID):
        session_idx = random_opto_df.index[random_opto_df.SessionID == session].item()
        cohort = random_opto_df.loc[session_idx].Genotype
        ax = axs[mice_cohorts.index(cohort)]
        st_t_idx = 0
        sh_d = random_opto_df.loc[session_idx].contralateral_bias
        imp_jit = random.uniform(-jitter, jitter)
        x_pos = jit_list[counter]
        counter+=1

        #means
        ax.plot(x_pos, sh_d, 'o', ms=14, color='k',
                markerfacecolor=colors[mice_cohorts.index(cohort)])
        #append to list
        mean_vals[mice_cohorts.index(cohort)].append(sh_d)
        sessions_used[mice_cohorts.index(cohort)].append(session)

    # plot mean of means next to it, and random distribution, and pvalue
    pvals = []
    for i, ax in enumerate(axs):
        bp = ax.boxplot(mean_vals[i], positions=[spread], widths=0.07, 
                    patch_artist=True, showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=colors[i], linewidth=3)
        for patch in bp['boxes']:
            patch.set(facecolor='white')

        # random expectation. Mean at 0 by definition. Use the bias_std to sample from
        # do one instance only
        random_means = []
        for session in sessions_used[i]:
            # get x number of a random bias
            sess_std = random_opto_df[random_opto_df.SessionID==session].bias_std.values
            random_means.append(np.random.normal(loc=0.0, scale=sess_std[0], size=100))
        random_means_flat_list = [item for sublist in random_means for item in sublist]
        
        spr_adj = 1.5
        bp = ax.boxplot(random_means_flat_list, positions=[spread*spr_adj], widths=0.07, 
                    patch_artist=True, showfliers=False)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='lightgray', linewidth=3)
        for patch in bp['boxes']:
            patch.set(facecolor='white')


        pvals.append(stats.kruskal(mean_vals[i], random_means_flat_list).pvalue)

    # add pvalues info
    hlocs = [20, -20]
    hadj = [1.2, 1.4]

    for i, ax in enumerate(axs):
        pvaltext = '{0:.7f}'.format(pvals[i])
        ax.text(x=spread*(1 + spr_adj)/2, y=hlocs[i]*hadj[i], s='pval {}'.format(str(pvaltext)),
                horizontalalignment='center', fontsize=14)
        ax.plot([spread, spread*spr_adj], [hlocs[i], hlocs[i]], color='k', linewidth=.5)
        ax.plot([spread, spread], [hlocs[i], hlocs[i]*.8], color='k', linewidth=.5)
        ax.plot([spread*spr_adj, spread*spr_adj], [hlocs[i], hlocs[i]*.8], color='k', linewidth=.5)
        ax.set_xticks([])
                    
    return fig


def make_figure_optoinhibition_after_learning_curves(oal_df, random_opto_df):
    # Plot the data with the error bars for the random sampling, and the custom fitting
    ColorList = ['powderblue', 'plum']
    normal_color = 'gray'
    LabelList = ['left stimulation', 'right stimulation']
    Genotypes = ['D1opto', 'D2opto']
    StimSides = ['Left', 'Right']

    n_cols = 2

    fig, axs = plt.subplots(1, n_cols,
                            figsize=(7 * n_cols, 5),
                            facecolor='w', edgecolor='k')

    axs = axs.ravel()

    for i, ax in enumerate(axs):

        genot = Genotypes[i]

        # select sessions
        g_mask = random_opto_df.Genotype == genot
        s_mask = random_opto_df.stimulated_side.isin(StimSides)

        sessions_list_cleaned = random_opto_df[np.logical_and(g_mask, s_mask)].SessionID
        
        # plot the normal choices and fit
        session_df = oal_df[oal_df['SessionID'].isin(sessions_list_cleaned)]
        df_for_plot = session_df[session_df.OptoStim==0]
        plot_utils.plot_regression(df=df_for_plot, ax=ax,
                                color=normal_color, label='', plot_points=False)
        
        predictDif, PsyPer, _, _, EB = \
        cuf.PP_ProcessExperiment(df_for_plot, 0, error_bars='SessionTime')
        plot_utils.PlotPsychPerformance(dataDif = PsyPer['Difficulty'], dataPerf = PsyPer['Performance'],
                                        predictDif = predictDif, ax = ax, fakePred = None,
                                        realPred = None, color = normal_color, label = 'control trials', errorBars = EB)
        
        
        # plot each side
        for k, stside in enumerate(StimSides):
            s_mask = random_opto_df.stimulated_side == stside
            sessions_list_cleaned = random_opto_df[np.logical_and(g_mask, s_mask)].SessionID

            # plot the normal choices and fit
            session_df = oal_df[oal_df['SessionID'].isin(sessions_list_cleaned)]
            df_for_plot = session_df[session_df.OptoStim==1]
            plot_utils.plot_regression(df=df_for_plot, ax=ax,
                                    color=ColorList[k], label='', plot_points=False)

            predictDif, PsyPer, _, _, EB = \
            cuf.PP_ProcessExperiment(df_for_plot, 0, error_bars='SessionTime')
            plot_utils.PlotPsychPerformance(dataDif = PsyPer['Difficulty'], dataPerf = PsyPer['Performance'],
                                            predictDif = predictDif, ax = ax, fakePred = None,
                                            realPred = None, color = ColorList[k], label = LabelList[k],
                                            errorBars = EB)

        ax.text(.5,1.05, genot, \
                horizontalalignment='center', fontweight='bold', transform=ax.transAxes, fontsize=16)    
        
        ax.axis('on')
        # remove some ticks
        ax.tick_params(which='both', top=False, bottom='on', left='on', right=False,
                    labelleft='on', labelbottom='on')
        if not ax.is_first_col():
            ax.set_ylabel('')
            ax.set_yticks([])
        if not ax.is_last_row():
            ax.set_xlabel('')
            ax.set_xticks([])

        ax.set_ylim(-2., 102.)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)

        # get rid of the frame
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # tick text size
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        
        # reverse x axis ticks
        ax.set_xticklabels([2, 18, 34, 50, 66, 82, 98][::-1])


        ax.set_ylabel('trials reported low (%)' , fontsize=16)
        
        ax.set_xlabel('low tones (%)', fontsize=16)

    plt.tight_layout()

    return fig

