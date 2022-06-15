"""Module containing commonly used tests"""
import itertools
import random
from cgitb import small

import numpy as np
import pandas as pd
from Lab_Analyses.Utilities.data_utilities import get_before_after_means
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate


def ANOVA_1way_posthoc(data_dict, method):
    """Function to perform a one way ANOVA with different posthoc
        tets
        
        INPUT PARAMETERS
            data_dict - dict of data to be analyzed. Each item is a different
                        group. Keys are group names, and values represent
                        the datapoints of each sample within the group
            
            method - str indicating the posthoc test to be performed. See
                    statsmodels.stats.multitest for available methods
        
        OUTPUT PARAMETERS
            f_stat - the f statistic from the one way ANOVA
            
            anova_p - the p-value of the one way ANOVA
            
            results_table - table of the results of the posttest

            results_df - dataframe of the results of the posttest
            
    """

    # Perform the one-way ANOVA
    ### Putting all the data points in a single array
    data_array = np.array(list(data_dict.values()))
    f_stat, anova_p = stats.f_oneway(*data_array)

    # Perform t-test across all groups
    ## get all the possible combinations to be tested
    combos = list(itertools.combinations(data_dict.keys(), 2))
    test_performed = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        test_performed.append(combo[0] + " vs. " + combo[1])
        t, p = stats.ttest_ind(data_dict[combo[0]], data_dict[combo[1]])
        t_vals.append(t)
        raw_pvals.append(p)

    # Perform multiple comparisions corrections
    _, adj_pvals, _, alpha_corrected = multipletests(
        raw_pvals, alpha=0.05, method=method, is_sorted=False, returnsorted=False,
    )
    results_dict = {
        "comparison": test_performed,
        "t stat": t_vals,
        "raw p-vals": raw_pvals,
        "adjusted p-vals": adj_pvals,
    }

    results_df = pd.DataFrame.from_dict(results_dict)
    results_table = tabulate(results_dict, headers="keys", tablefmt="fancy_grid")

    return f_stat, anova_p, results_table, results_df


def response_testing(imaging, ROI_ids, timestamps, window, sampling_rate, method):
    """Function to determine if each ROI displays significant responses during
        a specifid event
        
        INPUT PARAMETERSS
            imaging - array of the imaging data, with each column representing an ROI

            ROI_ids - list of string for each ROI id
            
            timestamps - list or array of timestamps corresponding to the imaging frame 
                         where the event occured
                         
            method - str specifying which method is to be used to determine significance.
                    Currently accept the following:
                        'test' - Perform Wilcoxon signed-rank test
                        'shuffle' - Compares the real differences in activity against a 
                                    shuffled distribution
        
        OUTPUT PARAMETERS
            results_dict - dictionary containing the results for each ROI (keys)
            
            results_df - dataframe containing the results for each ROI (columns)
            
            shuff_diffs - ('shuffle' method only) list containing all the shuffled
                          differences in activity for each ROI
    """

    if method == "test":
        # get the means before and after the event
        befores, afters = get_before_after_means(
            activity=imaging,
            timestamps=timestamps,
            window=window,
            sampling_rate=sampling_rate,
            offset=False,
        )

        # Peform testing and get mean differences in activity
        p_values = []
        rank_values = []
        diffs = []
        for before, after in zip(befores, afters):
            rank, pval = stats.wilcoxon(after, before)
            p_values.append(pval)
            rank_values.append(rank)
            diffs.append(np.mean(after - before))

        # Assess significance
        sig = (np.array(p_values) < 0.01) * 1

        # Put results in dictionary
        results_dict = {}
        for p, r, d, s, ROI in zip(p_values, rank_values, diffs, sig, ROI_ids):
            results_dict[ROI] = {"pvalue": p, "rank": r, "diff": d, "sig": s}

    elif method == "shuffle":
        data = imaging.copy()
        # Set up list variables
        real_diffs = []
        shuff_diffs = []
        bounds = []
        sigs = []
        smallest = sampling_rate  # smallest shift for shuffling
        biggest = 300 * sampling_rate  # Biggest shift for shuffling (5min)

        # Assess each roi individually
        for i in data.shape[1]:
            d = data[:, i]

            # get the real differences
            before, after = get_before_after_means(
                activity=d.reshape(-1, 1),
                timestamps=timestamps,
                window=window,
                sampling_rate=sampling_rate,
                offset=False,
            )
            r_diff = np.mean(after - before)

            # Perform 1000 shuffles
            s_diffs = []
            for j in range(1000):
                n = random.randint(smallest, biggest)
                s_d = np.copy(d)
                shuff_d = np.roll(s_d, n)
                s_before, s_after = get_before_after_means(
                    activity=shuff_d.reshape(-1, 1),
                    timestamps=timestamps,
                    window=window,
                    offset=False,
                )
                s_diffs.append(s_after, s_before)

            # Assess significance
            upper = np.percentile(s_diffs, 99)
            lower = np.percentile(s_diffs, 1)
            if lower <= r_diff <= upper:
                sig = 0
            else:
                sig = 1
            # Store values for each ROI
            real_diffs.append(r_diff)
            shuff_diffs.append(s_diffs)
            bounds.append((upper, lower))
            sigs.append(sig)

        # Store results in dictionary
        results_dict = {}
        for ROI, r, s, b, sig in zip(ROI_ids, real_diffs, shuff_diffs, bounds, sigs):
            results_dict[ROI] = {"diff": r, "shuff_diffs": s, "bounds": b, "sig": sig}

    else:
        return print("Not a valid testing method!!!")

    # Generate dataframe for results
    results_df = pd.DataFrame.from_dict(results_dict, orient="index")
    if "shuff_diff" in results_df.columns:
        results_df = results_df.drop(columns=["shuff_diffs"])

    return results_dict, results_df, shuff_diffs
