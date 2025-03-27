
import matplotlib.pyplot as plt
import matplotlib
from utils.beginning_and_end_quantification_utils import make_beginning_and_end_comparison_plot


font = {'size': 6}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(2, 1, figsize=[2, 4])
tail_df = make_beginning_and_end_comparison_plot(axs[0], site='tail', colour='#002F3A')
nacc_df = make_beginning_and_end_comparison_plot(axs[1], site='Nacc', colour='#E95F32')

plt.show()