import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from set_global_params import processed_data_path, spreadsheet_path, reproduce_figures_path
from bias_correlation_utils import categorise_da_responses, get_diff_in_proportion_correct, calculate_statistics, \
    calculate_psychometric, logistic
from utils.plotting_visuals import makes_plots_pretty, set_plotting_defaults
from utils.plotting import output_significance_stars_from_pval
import statsmodels.formula.api as smf
import shutil

# set some analysis parameters
quantile_cutoff = .65
slope_threshold = 1.  # slope at PSE of the psychometric function. this measures sensitivity
bias_threshold = .09  # bias of the behaviour during "easy" trials. this measures bias
# we include animals if they have a slope above the threshold AND a bias below the threshold


# Define colors for each data type
set_plotting_defaults(font_size=7)
colors = {'tail':['#76A8DA', '#002F3A'], 'nacc': ['#F9C0AF', '#E95F32']}
# Create a figure and axis for the stats plot
fig, ax = plt.subplots(figsize=(10, 6))
coefficients_df = pd.DataFrame()
std_errors_df = pd.DataFrame()
p_vals = {}

sites = ['nacc', 'tail']
for i, site in enumerate(sites):

    print('Analyzing site: {}'.format(site))
    print('-' * 50)

    # Set up data directory and copy data is it isn't in the right place
    original_dir = os.path.join(processed_data_path, 'psychometric_data')
    data_dir = os.path.join(reproduce_figures_path, 'fig4')
    if not os.path.exists(os.path.join(data_dir, 'all_tail_data_for_paper.csv')):
        shutil.copy(os.path.join(original_dir, 'all_tail_data_for_paper.csv'), os.path.join(data_dir, 'all_tail_data_for_paper.csv'))
    if not os.path.exists(os.path.join(data_dir, 'nacc_data_for_paper.csv')):
        shutil.copy(os.path.join(original_dir, 'nacc_data_for_paper.csv'), os.path.join(data_dir, 'nacc_data_for_paper.csv'))


    file_paths = {'tail': os.path.join(data_dir, 'all_tail_data_for_paper.csv'),
                  'nacc': os.path.join(data_dir, 'nacc_data_for_paper.csv')}
    file_path = file_paths[site]

    # set up results directory
    results_dir = os.path.join(data_dir, 'results\\{}'.format(site))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load data
    all_data = pd.read_csv(file_path)
    df = all_data.copy()
    df = calculate_statistics(df)

    # ---------------------------------------------------------------
    # calculate bias for each mouse
    bias_df = get_diff_in_proportion_correct(df)
    psymetric_df = calculate_psychometric(df)
    psymetric_df.to_csv(os.path.join(results_dir, 'bias_df_{}.csv'.format(site)))
    print(psymetric_df)

    # ---------------------------------------------------------------
    # show the average reward for each mouse as a function of uncertainty
    # No need to filter only for correct trials here, we're including both correct and incorrect
    grouped_by_mouse = df.groupby(['mouse', 'contraSensoryEvidence'])['outcome'].mean().reset_index()

    # Create a line plot for each mouse
    g = sns.FacetGrid(grouped_by_mouse, col="mouse", col_wrap=4, height=4, aspect=1.2)
    g.map(sns.lineplot, 'contraSensoryEvidence', 'outcome')
    g.add_legend()
    #plt.savefig(os.path.join(results_dir, 'outcome_vs_contraSensoryEvidence.png'))

    

    # ---------------------------------------------------------------
    # now show the choices for each mouse as a function of uncertainty

    # No need to filter only for correct trials here, we're including both correct and incorrect
    params_df = psymetric_df[['mouse', 'a', 'b']]
    def plot_with_fitted_psychometric_curve(x, y, color=None, label=None, **kwargs):
        # Extract data from kwargs
        data = kwargs.pop("data")

        # Plotting the actual data
        sns.lineplot(x=x, y=y, data=data, label=label, color=color)

        # Getting logistic parameters for the mouse
        a, b = params_dict[data['mouse'].iloc[0]]

        # Generating the fitted curve values
        x_range = np.linspace(0, 1, 400)
        y_fit = logistic(x_range, a, b)

        # Plotting the fitted curve
        plt.plot(x_range, y_fit, color='red', linewidth=2)

    grouped_by_mouse = df.groupby(['mouse', 'contraSensoryEvidence'])['numericSide'].mean().reset_index()
    params_dict = dict(zip(params_df['mouse'], params_df[['a', 'b']].values.tolist()))

    # Create the FacetGrid
    g = sns.FacetGrid(grouped_by_mouse, col="mouse", col_wrap=4, height=4, aspect=1.2)

    # Mapping the modified plotting function to the FacetGrid
    g.map_dataframe(plot_with_fitted_psychometric_curve, 'contraSensoryEvidence', 'numericSide')

    g.add_legend()
    #plt.savefig(os.path.join(results_dir, 'choice_vs_contraSensoryEvidence_with_curve.png'))
    


    # ---------------------------------------------------------------
    # exclude incorrect trials and categorize DA response size
    # Note, for the nacc the response is aligned to reward, so we can only use correct trials
    # for the tail, the response is aligned to choice, so we use only correct trials where mice should have a bias,
    # but all trials for the ambiguous stimulus (where incorrect / correct is not defined)
    if site == 'tail':
        df = df[~((df['outcome'] == 0) & (df['contraSensoryEvidence'] != 0.5))]
        df = df.reset_index()
    else:
        df = df[df['outcome'] == 1].reset_index(drop=True)

    df['DAresponseSize'] = df.groupby(['mouse'])['norm APE'].apply(categorise_da_responses, cutoff=quantile_cutoff).reset_index(drop=True)

    # ---------------------------------------------------------------
    # exclude mice with low slope and large bias
    mice_to_exclude = []
    for mouse in psymetric_df.mouse[psymetric_df.slope < slope_threshold]:
        mice_to_exclude.append(mouse)
    for mouse in bias_df.mouse[bias_df.difference > bias_threshold]:
        mice_to_exclude.append(mouse)

    print('Excluding mice: {}'.format(np.unique(mice_to_exclude)))
    df = df[~df.mouse.isin(mice_to_exclude)]

    # ---------------------------------------------------------------
    # plot the difference between ipsi and contra for each mouse, contraSensoryEvidence, and daResponseSize
    plt.figure()
    sns.lineplot(data=df, x='nextContraSensoryEvidence', y='nextNumericSide', hue='DAresponseSize', style='numericSide')

    # Add labels and title
    plt.xlabel('Next Contra Sensory Evidence')
    plt.ylabel('Next Numeric Side')
    plt.title('Next Numeric Side vs Next Contra Sensory Evidence')
    plt.legend(title='Dopamine Level / Current Trial Side')
    #plt.savefig(os.path.join(results_dir, 'nextNumericSide_vs_nextContraSensoryEvidence.png'))
    

    # ---------------------------------------------------------------
    # plot the difference between ipsi and contra for each mouse, nextContraSensoryEvidence, and daResponseSize
    # ---------------------------------------------------------------

    # Step 1: Compute the nextNumericSide for each mouse, nextContraSensoryEvidence, daResponseSize, and numericSide
    grouped_mouse = df.groupby(['mouse', 'nextContraSensoryEvidence', 'DAresponseSize', 'numericSide'])['nextNumericSide'].mean().reset_index()

    # Pivot the table to calculate the difference between ipsi and contra for each mouse, nextContraSensoryEvidence, and daResponseSize
    pivot_table = grouped_mouse.pivot_table(index=['mouse', 'nextContraSensoryEvidence', 'DAresponseSize'], columns='numericSide', values='nextNumericSide').reset_index()
    pivot_table['diffNextNumericSide'] = pivot_table[1] - pivot_table[0]  # Assuming 0 represents ipsi and 1 represents contra

    # Drop the original ipsi and contra columns and "flatten" the dataframe
    agg_diff = pivot_table.drop(columns=[0, 1])
    plt.figure()
    # Step 2: Plot the results directly
    sns.lineplot(data=agg_diff, x='nextContraSensoryEvidence', y='diffNextNumericSide', hue='DAresponseSize', ci=68)
    plt.ylabel("Difference in nextNumericSide (Ipsi - Contra)")
    plt.title("Bias dependent on previous trial DA response")
    #plt.savefig(os.path.join(results_dir,'diffNextNumericSide.png'))
    


    # ---------------------------------------------------------------
    # same as above but separate by mouse
    # ---------------------------------------------------------------

    # Compute the nextNumericSide for each mouse, nextContraSensoryEvidence, DAresponseSize, and numericSide
    grouped_mouse = df.groupby(['mouse', 'nextContraSensoryEvidence', 'DAresponseSize', 'numericSide'])[
        'nextNumericSide'].mean().reset_index()

    # Pivot the table to calculate the difference between ipsi and contra for each mouse, nextContraSensoryEvidence, and DAresponseSize
    pivot_table = grouped_mouse.pivot_table(index=['mouse', 'nextContraSensoryEvidence', 'DAresponseSize'],
                                            columns='numericSide', values='nextNumericSide').reset_index()
    pivot_table['diffNextNumericSide'] = pivot_table[1] - pivot_table[0] # Assuming 0 represents ipsi and 1 represents contra

    # Drop the original ipsi and contra columns and "flatten" the dataframe
    agg_diff = pivot_table.drop(columns=[0, 1])

    # Create separate plots for each mouse (uncomment if you want this)
    # for mouse in agg_diff['mouse'].unique():
    #     plt.figure()
    #     mouse_data = agg_diff[agg_diff['mouse'] == mouse]
    #
    #     sns.lineplot(data=mouse_data, x='nextContraSensoryEvidence', y='diffNextNumericSide', hue='DAresponseSize')
    #     plt.title(f"Mouse: {mouse}")
    #     plt.ylabel("Difference in nextNumericSide (Ipsi - Contra)")
        #plt.savefig(os.path.join(results_dir, f'diffNextNumericSide_mouse_{mouse}.png'))


    # ---------------------------------------------------------------
    # now the uncertainty plot (same as above but folded over the middle)
    # ---------------------------------------------------------------
    # Compute the nextNumericSide for each mouse, nextContraSensoryEvidence, DAresponseSize, and numericSide
    grouped_mouse = df.groupby(['mouse', 'nextContraSensoryEvidence', 'DAresponseSize', 'numericSide'])['nextNumericSide'].mean().reset_index()

    # Compute the "mirrored" nextContraSensoryEvidence
    grouped_mouse['mirroredEvidence'] = np.where(grouped_mouse['nextContraSensoryEvidence'] > 0.5, 1 - grouped_mouse['nextContraSensoryEvidence'], grouped_mouse['nextContraSensoryEvidence'])

    # Compute the "uncertainty" by reflecting around the midpoint and round to 3 decimal places
    grouped_mouse['uncertainty'] = grouped_mouse['nextContraSensoryEvidence'].apply(lambda x: 1 - np.around(np.abs(.5 - x), decimals=2))
    # normalize between 0 and 1 and take log
    #grouped_mouse['uncertainty'] = grouped_mouse.uncertainty.apply(lambda x: np.log(x + 1))

    # Pivot the table to calculate the difference between ipsi and contra for each mouse, uncertainty, and DAresponseSize
    pivot_table = grouped_mouse.pivot_table(index=['mouse', 'uncertainty', 'DAresponseSize'], columns='numericSide', values='nextNumericSide').reset_index()
    pivot_table['diffNextNumericSide'] = pivot_table[1] - pivot_table[0]  # Assuming 0 represents ipsi and 1 represents contra

    # Drop the original ipsi and contra columns and "flatten" the dataframe
    agg_diff = pivot_table.drop(columns=[0, 1])

    # Group by uncertainty and DAresponseSize and compute the mean difference
    agg_diff_grouped = agg_diff.groupby(['mouse', 'uncertainty', 'DAresponseSize'])['diffNextNumericSide'].mean().reset_index()
    max_values_per_mouse = agg_diff_grouped.groupby('mouse')['diffNextNumericSide'].transform('max')
    # Normalize 'diffNextNumericSide' by dividing by the maximum value
    agg_diff_grouped['normalizedDiffNextNumericSide'] = agg_diff_grouped['diffNextNumericSide'] / max_values_per_mouse
    # Plot the average difference as a function of uncertainty
    fig, ax = plt.subplots(figsize=(2, 2.3))
    sns.lineplot(data=agg_diff_grouped, x='uncertainty', y='diffNextNumericSide', hue='DAresponseSize', ci=68, palette=colors[site])
    plt.ylabel("Contralateral bias")
    plt.xlabel("Perceptual uncertainty")
    makes_plots_pretty(ax)
    plt.tight_layout()
    ax.legend(frameon=False)
    #plt.savefig(os.path.join(results_dir, 'uncertainty_plot_{}.pdf'.format(site)))

    # Apply logarithm transformation to 'uncertainty'
    agg_diff_grouped['log_uncertainty'] = np.log(agg_diff_grouped['uncertainty'])

    # Plot the average difference as a function of log uncertainty
    fig, ax = plt.subplots(figsize=(2, 2.3))
    sns.lineplot(data=agg_diff_grouped, x='log_uncertainty', y='diffNextNumericSide', hue='DAresponseSize',
                 ci=68, palette=colors[site])
    plt.ylabel("Contralateral bias")
    plt.xlabel("Log Perceptual Uncertainty")
    makes_plots_pretty(ax)
    plt.tight_layout()
    ax.legend(frameon=False)
    #plt.savefig(os.path.join(results_dir, 'log_uncertainty_plot_{}.pdf'.format(site)))
    subfig = 'H' if site == 'tail' else 'J'
    line_plot_csv_file = os.path.join(spreadsheet_path, 'fig4', f'fig4{subfig}_uncertainty_df.csv')
    if not os.path.exists(line_plot_csv_file):
        df_to_save = agg_diff_grouped[[ 'mouse', 'uncertainty', 'diffNextNumericSide', 'DAresponseSize']]
        df_to_save.to_csv(line_plot_csv_file)


    model = smf.ols(formula='diffNextNumericSide ~ log_uncertainty * C(DAresponseSize)', data=agg_diff_grouped)
    results = model.fit()
    print(results.summary())

    DA_pval = results.pvalues['C(DAresponseSize)[T.large]']
    interaction_pval = results.pvalues['log_uncertainty:C(DAresponseSize)[T.large]']
    uncertainty_pval = results.pvalues['log_uncertainty']
    p_vals[site] = [DA_pval, uncertainty_pval, interaction_pval]

    coefficients = results.params.drop("Intercept")
    std_errors = results.bse.drop("Intercept")
    # Add the coefficients and standard errors to the DataFrames with a column for the data type
    coefficients_df[site] = coefficients
    std_errors_df[site] = std_errors

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(2.8, 2.3))
set_plotting_defaults(font_size=6)
# Plot the coefficients as bars with error bars for standard errors
x = np.arange(len(coefficients))  # x-axis positions
width = 0.2  # Width of the bars
site_labels = {'nacc': 'VS', 'tail': 'TS'}

for i, site in enumerate(sites):
    x_positions = x + i * width
    ax.bar(x_positions, coefficients_df[site], width=width, label=site, color=colors[site][1], capsize=4,
           yerr=std_errors_df[site],  # Include error bars
           error_kw={'ecolor': 'lightgray', 'elinewidth': 1, 'capsize': 0})

    # significance stars
    site_p_vals = p_vals[site]
    ax.text(x_positions[0], 0.2, output_significance_stars_from_pval(site_p_vals[0]), ha='center', fontsize=7)
    ax.text(x_positions[1], 0.4, output_significance_stars_from_pval(site_p_vals[1]), ha='center', fontsize=7)
    ax.text(x_positions[2], 0.3, output_significance_stars_from_pval(site_p_vals[2]), ha='center', fontsize=7)
    print('{}: DA size {}, uncertainty {}, interaction {})'.format(site, site_p_vals[0], site_p_vals[1], site_p_vals[2]))

# Customize the plot
ax.set_xlabel("Regressors")
tick_positions = x + (width * (len(sites) - 1)) / 2
tick_labels = [ "DA response", "uncertainty (log)", "Interaction"]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_ylabel("Coefficients")
ax.axhline(0, color='gray', lw=0.5)
# remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), frameon=False,loc='center left', bbox_to_anchor=(1, 0.5))
# Show the plot
plt.tight_layout()
makes_plots_pretty(ax)
plt.tight_layout()
#plt.savefig(os.path.join(results_dir, 'reg_coef_bar_plot.pdf'))
plt.show()