"""Module containing commonly used tests"""
import itertools
import random

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate

from Lab_Analyses.Utilities.data_utilities import get_before_after_means


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
    data_array = list(data_dict.values())
    f_stat, anova_p = stats.f_oneway(*data_array)

    # Perform multiple comparisions
    if method != "Tukey":
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
        # Perform corrections
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

    if method == "Tukey":
        # Organize data for input
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        df = pd.melt(df, value_vars=df.columns, var_name="group", value_name="variable")
        df.dropna(inplace=True)
        print(len(df.index))

        # Perform comparisions
        tukey_result = pairwise_tukeyhsd(
            endog=df["variable"], groups=df["group"], alpha=0.05
        )

        results_df = pd.DataFrame(
            data=tukey_result._results_table.data[1:],
            columns=tukey_result._results_table.data[0],
        )
        results_dict = results_df.to_dict("list")

    results_table = tabulate(results_dict, headers="keys", tablefmt="fancy_grid")

    return f_stat, anova_p, results_table, results_df


def ANOVA_2way_posthoc(data_dict, groups_list, variable, method, exclude=None):
    """Function to perform a two way anova with a specified posthoc test
    
        INPUT PARAMETERS 
            data_dict - nested dictionaries of data to be plotted. Outer keys represent
                        the subgroups, while the inner keys represent the main groups
            
            groups_list - list of str specifying the groups. Main group will be first
                         and sub groups second
            
            variable - str specifying what the variable being tested is
            
            method - str specifying the str indicating the posthoc test to be performed. See
                    statsmodels.stats.multitest for available methods
            
            exclude - list of str specifying posthoc tests to ignore
    """
    # First organize the data in order to perform the testing
    dfs = []
    g1_keys = []
    g2_keys = []
    for key, value in data_dict.items():
        g1_keys = list(value.keys())
        g2_keys.append(key)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in value.items()]))
        df = pd.melt(
            df, value_vars=df.columns, var_name=groups_list[0], value_name=variable
        )
        df[groups_list[1]] = key
        dfs.append(df.dropna())
    test_df = pd.concat(dfs)
    formula = f"{variable} ~ C({groups_list[0]}) + C({groups_list[1]}) + C({groups_list[0]}):C({groups_list[1]})"
    # Perform the two-way anova
    model = ols(formula=formula, data=test_df).fit()
    two_way_anova = sm.stats.anova_lm(model, typ=2)

    # Get all the combos
    all_keys = [g1_keys, g2_keys]
    groups = [c for c in itertools.product(*all_keys)]
    test_combos = list(itertools.combinations(groups, 2))

    # Perform multiple comparison
    if method != "Tukey":
        test_performed = []
        t_vals = []
        raw_pvals = []
        for combo in test_combos:
            # Keep track of the comparisons being made
            group_names = []
            for c in combo:
                names = [x.split("_")[0] for x in c]
                names.append("spines")
                group_names.append(" ".join(names))
            test_performed.append(f"{group_names[0]} vs {group_names[1]}")
            # Get the data for each group
            data_1 = test_df[
                (test_df[groups_list[0]] == combo[0][0])
                & (test_df[groups_list[1]] == combo[0][1])
            ]
            data_2 = test_df[
                (test_df[groups_list[0]] == combo[1][0])
                & (test_df[groups_list[1]] == combo[1][1])
            ]
            # Convert data to arrays
            data_1 = np.array(data_1[variable])
            data_2 = np.array(data_2[variable])
            # Perform t-test
            t, p = stats.ttest_ind(data_1, data_2)
            t_vals.append(t)
            raw_pvals.append(p)
        _, adj_pvals, _, alpha_corrected = multipletests(
            raw_pvals, alpha=0.5, method=method, is_sorted=False, returnsorted=False,
        )
        posthoc_dict = {
            "posthoc comparision": test_performed,
            "t stat": t_vals,
            "raw p-vals": raw_pvals,
            "adjusted p-vals": adj_pvals,
        }

    if method == "Tukey":
        # Organize data to put into Tukey
        temp_dfs = []
        for group in groups:
            names = [x.split("_")[0] for x in group]
            names.append("spines")
            group_name = " ".join(names)
            data = test_df[
                (test_df[groups_list[0]] == group[0])
                & (test_df[groups_list[1]] == group[1])
            ]
            data = np.array(data[variable])
            labels = [group_name for i in range(len(data))]
            d = {"group": labels, "variable": data}
            temp_df = pd.DataFrame(data=d)
            temp_dfs.append(temp_df)
        tukey_df = pd.concat(temp_dfs)
        # Perform comparisons
        tukey_result = pairwise_tukeyhsd(
            endog=tukey_df["variable"], groups=tukey_df["group"], alpha=0.05
        )
        posthoc_df = pd.DataFrame(
            data=tukey_result._results_table.data[1:],
            columns=tukey_result._results_table.data[0],
        )
        posthoc_dict = posthoc_df.to_dict("list")

    two_way_anova_table = tabulate(two_way_anova, headers="keys", tablefmt="fancy_grid")
    posthoc_table = tabulate(posthoc_dict, headers="keys", tablefmt="fancy_grid")

    return two_way_anova_table, posthoc_table


def kruskal_wallis_test(data_dict, post_method, adj_method):
    """Function to perform a Kruskal-Wallis test with different posthoc tests
    
        INPUT PARAMETERS
            data_dict - dict of data to be analyzed. Each item is a different group.
                        Keys are group names, and values represent the data points
            
            method - str indicating the posthoc test to be performed. See
                    statsmodels.stats.multitests for available methods

    """
    # Perform the Kruskal-Wallis Test
    ## Put data into an array format
    data_array = list(data_dict.values())
    f_stat, kruskal_p = stats.kruskal(*data_array, nan_policy="omit")

    # Perform multiple comparisons
    combos = list(itertools.combinations(data_dict.keys(), 2))
    if post_method == "Dunn":
        pval_df = sp.posthoc_dunn(data_array, p_adjust=adj_method).to_numpy()
    elif post_method == "Conover":
        pval_df = sp.posthoc_conover(data_array, p_adjust=adj_method).to_numpy()
    test_performed = [f"{c[0]} vs {c[1]}" for c in combos]
    p_combos = list(itertools.combinations(list(range(pval_df.shape[0])), 2))
    adj_pvals = [pval_df[c[0], c[1]] for c in p_combos]

    results_dict = {
        "comparison": test_performed,
        "adjusted p-vals": adj_pvals,
    }

    results_table = tabulate(results_dict, headers="keys", tablefmt="fancy_grid")

    return f_stat, kruskal_p, results_table


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

    return results_dict, results_df
