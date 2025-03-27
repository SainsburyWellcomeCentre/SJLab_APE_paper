import numpy as np


def cohen_d(group1, group2):
    # Calculate the means of both groups
    mean1, mean2 = np.mean(group1), np.mean(group2)

    # Calculate the standard deviations of both groups
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Calculate the pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    print("cohen d: ", d)
    return d


def cohen_d_paired(group1, group2):
    # Calculate the differences between each pair
    differences = np.array(group1) - np.array(group2)

    # Calculate the mean of the differences
    mean_diff = np.mean(differences)

    # Calculate the standard deviation of the differences
    std_diff = np.std(differences, ddof=1)

    # Calculate Cohen's d for paired samples
    d = mean_diff / std_diff
    print("cohen d: ", d)
    return d


def cohen_d_one_sample(data, mu0=0):
    # Calculate the sample mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    # Calculate Cohen's d
    d = (mean - mu0) / std_dev
    print("cohen d: ", d)
    return d


