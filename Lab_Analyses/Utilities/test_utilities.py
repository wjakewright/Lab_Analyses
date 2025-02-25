"""Module containing commonly used tests"""

import itertools
import os
import random
import subprocess

os.environ["R_HOME"] = r"C:\Users\Jake\Documents\Anaconda3\envs\analysis_env\Lib\R"

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.api as sm
from pymer4.models import Lmer
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
                statsmodels.stats.multitest for available methods,
                in additoin to TukeyHSD

    OUTPUT PARAMETERS
        f_stat - the f statistic from the one way ANOVA

        anova_p - the p-value of the one way ANOVA

        results_table - table of the results of the posttest

        results_df - dataframe of the results of the posttest

    """

    # Perform the one-way ANOVA
    ### Putting all the data points in a single array
    data_array = list(data_dict.values())
    data_array = [x[~np.isnan(x)] for x in data_array]
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
            t, p = stats.ttest_ind(
                data_dict[combo[0]], data_dict[combo[1]], nan_policy="omit"
            )
            t_vals.append(t)
            raw_pvals.append(p)
        # Perform corrections
        _, adj_pvals, _, alpha_corrected = multipletests(
            raw_pvals,
            alpha=0.05,
            method=method,
            is_sorted=False,
            returnsorted=False,
        )
        results_dict = {
            "comparison": test_performed,
            "t stat": t_vals,
            "raw p-vals": raw_pvals,
            "adjusted p-vals": adj_pvals,
        }
        results_df = pd.DataFrame.from_dict(results_dict)
        results_df.update(results_df[["raw p-vals"]].applymap("{:.4E}".format))
        results_df.update(results_df[["adjusted p-vals"]].applymap("{:.4E}".format))

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
                statsmodels.stats.multitest for available methods in addition to TukeyHSD

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

    two_way_anova = two_way_anova.rename_axis("Comp").reset_index()
    two_way_anova.update(two_way_anova[["sum_sq"]].applymap("{:.3}".format))
    two_way_anova.update(two_way_anova[["df"]].applymap("{:.3}".format))
    two_way_anova.update(two_way_anova[["F"]].applymap("{:.3}".format))
    two_way_anova.update(two_way_anova[["PR(>F)"]].applymap("{:.3E}".format))

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
                # names.append("spines")
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
            t, p = stats.ttest_ind(data_1, data_2, nan_policy="omit")
            t_vals.append(t)
            raw_pvals.append(p)
        raw_pvals = np.nan_to_num(raw_pvals, nan=1)
        _, adj_pvals, _, alpha_corrected = multipletests(
            raw_pvals,
            alpha=0.5,
            method=method,
            is_sorted=False,
            returnsorted=False,
        )
        posthoc_dict = {
            "posthoc comparision": test_performed,
            "t stat": t_vals,
            "raw p-vals": np.array(raw_pvals),
            "adjusted p-vals": adj_pvals,
        }
        posthoc_df = pd.DataFrame.from_dict(posthoc_dict)
        posthoc_df.update(posthoc_df[["raw p-vals"]].applymap("{:.4E}".format))
        posthoc_df.update(posthoc_df[["adjusted p-vals"]].applymap("{:.4E}".format))

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

    # two_way_anova_table = tabulate(two_way_anova, headers="keys", tablefmt="fancy_grid")
    # posthoc_table = tabulate(posthoc_dict, headers="keys", tablefmt="fancy_grid")
    two_way_anova_table = two_way_anova
    posthoc_table = posthoc_df

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
    nan_data_array = [x[~np.isnan(x)] for x in data_array]
    if post_method == "Dunn":
        pval_df = sp.posthoc_dunn(nan_data_array, p_adjust=adj_method).to_numpy()
    elif post_method == "Conover":
        pval_df = sp.posthoc_conover(nan_data_array, p_adjust=adj_method).to_numpy()
    test_performed = [f"{c[0]} vs {c[1]}" for c in combos]
    p_combos = list(itertools.combinations(list(range(pval_df.shape[0])), 2))
    adj_pvals = [pval_df[c[0], c[1]] for c in p_combos]

    results_dict = {
        "comparison": test_performed,
        "adjusted p-vals": adj_pvals,
    }

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["adjusted p-vals"]].applymap("{:.4E}".format))

    # results_table = tabulate(results_dict, headers="keys", tablefmt="fancy_grid")
    results_table = results_df

    return f_stat, kruskal_p, results_table


def ANOVA_2way_mixed_posthoc(
    data_dict, method, rm_vals=None, compare_type="between", test_type="parametric"
):
    """Function to perform a repeated measures two-way anova with specified posthoc
    tests

    INPUT PARAMETERS
        data_dict - dictionary with each key representing a group that contains a
                    2d array with the data. Each column represents the roi, while
                    rows represent the repeated measures

        method - str specifying the posthoc test to be performed

        rm_vals - list or array containing the values for the repeated measure
                  values. (e.g, 5,10, 15 um). If none, index will be used as the labels

        compare_type - what type of comparisons you wish to perform posthoc. Options
                        are 'between' to compare between groups at each rm, 'within'
                        to compare within groups across rm values, and 'both' which
                        makes all possible comparisons

    """
    groups = list(data_dict.keys())
    # Set up rm_vals
    data_len = list(data_dict.values())[0].shape[0]
    if rm_vals is None:
        rm_vals = list(range(data_len))
    elif len(rm_vals) != data_len:
        print("Inputed RM Values does not match data!!")
        rm_vals = list(range(data_len))

    # Organize the data in the appropriate format
    dfs = []
    sub_count = 1
    for key, value in data_dict.items():
        for v in range(value.shape[1]):
            data = value[:, v]
            g = [key for x in range(len(data))]
            sub = [sub_count for x in range(len(data))]
            temp_dict = {"subject": sub, "data": data, "group": g, "rm_val": rm_vals}
            temp_df = pd.DataFrame(temp_dict)
            dfs.append(temp_df.dropna(axis=0))
            sub_count = sub_count + 1
    test_df = pd.concat(dfs)

    test_df.reset_index(inplace=True, drop=True)

    # Perform the mixed ANOVA
    two_way_mixed_anova = pg.mixed_anova(
        data=test_df, dv="data", between="group", within="rm_val", subject="subject"
    )
    # two_way_mixed_anova = two_way_mixed_anova.to_dict("list")
    # two_way_mixed_anova = tabulate(
    #    two_way_mixed_anova, headers="keys", tablefmt="fancy_grid"
    # )

    # Perform the posthoc tests
    ## Between group comparisions
    if compare_type == "between":
        ## Get combinations of test
        combos = list(itertools.combinations(groups, 2))
        test_performed = []
        rm_point = []
        t_vals = []
        raw_pvals = []
        for combo in combos:
            for rm in rm_vals:
                ## Kep track of comparisons being made
                test_performed.append(combo[0] + " vs. " + combo[1])
                rm_point.append(rm)
                ## Get the data for the groups
                data1 = test_df[
                    (test_df["group"] == combo[0]) & (test_df["rm_val"] == rm)
                ]
                data2 = test_df[
                    (test_df["group"] == combo[1]) & (test_df["rm_val"] == rm)
                ]
                data1 = np.array(data1["data"])
                data2 = np.array(data2["data"])
                ## Perform the t-tests
                if test_type == "parametric":
                    t, p = stats.ttest_ind(data1, data2)
                elif test_type == "nonparametric":
                    t, p = stats.mannwhitneyu(data1, data2)
                t_vals.append(t)
                raw_pvals.append(p)

        ## Correct multiple comparisons
        raw_pvals = np.nan_to_num(raw_pvals, nan=1)
        _, adj_pvals, _, alpha_corrected = multipletests(
            raw_pvals,
            alpha=0.05,
            method=method,
            is_sorted=False,
            returnsorted=False,
        )
        posthoc_dict = {
            "posthoc comparison": test_performed,
            "within point": rm_point,
            "t stat": t_vals,
            "raw p-vals": np.array(raw_pvals),
            "adjusted p-vals": adj_pvals,
        }
        # posthoc_table = tabulate(posthoc_dict, headers="keys", tablefmt="fancy_grid")
        posthoc_table = pd.DataFrame.from_dict(posthoc_dict)
        try:
            two_way_mixed_anova.drop(
                columns=["sphericity", "W-spher", "p-spher", "p-GG-corr"], inplace=True
            )
        except KeyError:
            pass

        # Format outputs
        two_way_mixed_anova.update(two_way_mixed_anova[["SS"]].applymap("{:.4}".format))
        two_way_mixed_anova.update(two_way_mixed_anova[["MS"]].applymap("{:.4}".format))
        two_way_mixed_anova.update(two_way_mixed_anova[["F"]].applymap("{:.4}".format))
        two_way_mixed_anova.update(
            two_way_mixed_anova[["p-unc"]].applymap("{:.3E}".format)
        )
        two_way_mixed_anova.update(
            two_way_mixed_anova[["np2"]].applymap("{:.4}".format)
        )
        two_way_mixed_anova.update(
            two_way_mixed_anova[["eps"]].applymap("{:.4}".format)
        )

        posthoc_table.update(posthoc_table[["t stat"]].applymap("{:.4}".format))
        posthoc_table.update(posthoc_table[["raw p-vals"]].applymap("{:.3E}".format))
        posthoc_table.update(
            posthoc_table[["adjusted p-vals"]].applymap("{:.3E}".format)
        )

    return two_way_mixed_anova, posthoc_dict, posthoc_table


def significant_vs_shuffle(real_values, shuffle_values, alpha, nan_policy="omit"):
    """Function to test if real data is significantly different vs shuffled data

    INPUT PARAMETERS
        real_values - np.array of the the real data

        shuffle_values - 2d np.array of the shuffle data. Each row represents a shuffle

        alpha - float speicfying the significance level

        nan_policy - str specifying how to deal with nan values in shuffled data
                    Accepts "omit" to omit values before determining rank and "zero"
                    to zero the values before determining rank

    OUTPUT PARAMETERS
        ranks - np.array of the ranks of the real values vs the shuffles

        significance - np.array of whether each value is sig greater or smaller
                        vs the shuffles (-1 = smaller, 1 = greater, 0 = no diff)

    """
    UPPER = 100 - (alpha * 10)
    LOWER = alpha * 10

    # setup the outputs
    ranks = []
    significance = []

    # Iterate through each real and shuffle values
    for i, value in enumerate(real_values):
        # Pull shuff data
        shuff = shuffle_values[:, i]
        # Check for nans
        if np.isnan(value):
            ranks.append(np.nan)
            significance.append(np.nan)
            continue
        ## Remove nan values from shuff
        if np.sum(np.isnan(shuff)):
            if nan_policy == "omit":
                shuff = shuff[~np.isnan(shuff)]
            elif nan_policy == "zero":
                shuff[np.isnan(shuff)] = 0
            continue
        # calculate the rank
        rank = stats.percentileofscore(shuff, value)
        # determine significance
        if rank >= UPPER:
            significance.append(1)
        elif rank <= LOWER:
            significance.append(-1)
        else:
            significance.append(0)

    # Convert outputs to arrays
    ranks = np.array(ranks)
    significance = np.array(significance)

    return ranks, significance


def correlate_grouped_data(data_dict, x_vals):
    """Function to correlate multiple groups of data

    INPUT PARAMETERS
        data_dict - dictionary of 2d arrays with each row corresponding to a
                    time point / measurement while each col corresponds to
                    different rois / subjects

        x_vals - np.array of the x values corresponding to each row of the 2d arrays
    """
    corr_dict = {
        "Group": [],
        "r": [],
        "p val": [],
    }
    for key, value in data_dict.items():
        binned_v = []
        binned_p = []
        for i in range(value.shape[0]):
            binned_v.append(value[i, :])
            binned_p.append([x_vals[i] for x in range(value.shape[1])])
        binned_values = np.concatenate(binned_v)
        binned_pos = np.array([y for x in binned_p for y in x])
        binned_non_nan = np.nonzero(~np.isnan(binned_values))[0]
        binned_values = binned_values[binned_non_nan]
        binned_pos = binned_pos[binned_non_nan]
        r, p = stats.spearmanr(binned_pos, binned_values)
        corr_dict["Group"].append(key)
        corr_dict["r"].append(r)
        corr_dict["p val"].append(p)

    corr_df = pd.DataFrame.from_dict(corr_dict)
    corr_df.update(corr_df[["p val"]].applymap("{:.4E}".format))
    corr_df.update(corr_df[["r"]].applymap("{:.4E}".format))

    return corr_dict, corr_df


def test_against_chance(real_array, shuff_matrix):
    """Function to test statistical significance against chance distributions

    INPUT PARAMETERS
        real_array - 1d np.array of the real values

        shuff_matrix - 2d np.array of the shuff values. Each row represents
                        a shuff and each column represents each data point

    """
    real_median = np.nanmedian(real_array)
    shuff_medians = np.nanmedian(shuff_matrix, axis=1)
    # Get fraction following null hypothesis
    frac_below = np.nansum(shuff_medians <= real_median) / len(shuff_medians)
    frac_above = np.nansum(shuff_medians >= real_median) / len(shuff_medians)

    return frac_above, frac_below


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


def one_way_mixed_effects_model(
    data_dict, random_dict, post_method, slopes_intercept=False
):
    """Function to perform mixed effects model statistics for comparisons
    across simple groups (e.g., t-test or one-way ANOVA-like). Takes
    in additional random effect parameter to control for (e.g., animal id)

    INPUT PARAMETERS
        data_dict - dict of data to be analyuzed. Each item is a different group.
                    Keys are group names, and values represent the data points

        random_dict - dict of containing the random variable corresponding to
                     each data sample. Keys are the same group names

        post_method - str indicating the posthoc test to be performed.
                      See statsmodels.stats.multitests for available methods

        slopes_intercept - boolean specifying whether to include random effects
                            for both the intercept and slope. Default is False for
                            only intercept effects

    OUTPUT PARAMETERS
        t_stat - float of the t-stat from the mix effects model test

        p_val - float of the p-val from the mixed effects model test

        results_df - pd.DataFrame of the posthoc test results

    """
    # Organize the data into a DataFrame
    ## Convert from dictionaries
    data_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
    rand_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in random_dict.items()]))
    ## Linearize the data into columns
    data_df = pd.melt(
        data_df, value_vars=data_dict.keys(), var_name="Group", value_name="Data"
    )
    rand_df = pd.melt(
        rand_df, value_vars=random_dict.keys(), var_name="Group", value_name="Rand_Var"
    )
    ## Merge the random variable into the dict
    data_df["Rand_Var"] = rand_df["Rand_Var"]
    ## Hot code group values
    map_dict = {}
    codes = list(range(len(data_dict.keys())))
    for i, key in enumerate(data_dict.keys()):
        map_dict[key] = codes[i]
    data_df = data_df.assign(Group_Coded=data_df["Group"].map(map_dict))
    data_df = data_df.dropna(axis=0, how="any")

    # Perform the mix-effects model test
    ## Define formula
    if slopes_intercept is True:
        formula = "Data ~ Group + (1|Rand_Var) + (Group-1|Rand_Var)"
    else:
        formula = "Data ~ Group + (1|Rand_Var)"
    ## Construct and fit the model
    model = Lmer(formula, data=data_df)
    model.fit(summarize=False, factors={"Group": list(data_dict.keys())}, ordered=True)
    anova = model.anova()
    ## Get t and p values
    t_stat = anova["F-stat"]["Group"]
    p_val = anova["P-val"]["Group"]

    # Perform the posthocs
    combos = list(itertools.combinations(data_dict.keys(), 2))
    test_performed = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        test_performed.append(combo[0] + " vs. " + combo[1])
        t, p = stats.ttest_ind(
            data_dict[combo[0]], data_dict[combo[1]], nan_policy="omit"
        )
        t_vals.append(t)
        raw_pvals.append(p)
    # Perform corrections
    _, adj_pvals, _, alpha_corrected = multipletests(
        raw_pvals,
        alpha=0.05,
        method=post_method,
        is_sorted=False,
        returnsorted=False,
    )
    results_dict = {
        "comparison": test_performed,
        "t stat": t_vals,
        "raw p-vals": raw_pvals,
        "adjusted p-vals": adj_pvals,
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["raw p-vals"]].applymap("{:.4E}".format))
    results_df.update(results_df[["adjusted p-vals"]].applymap("{:.4E}".format))

    return t_stat, p_val, results_df


def two_way_RM_mixed_effects_model(
    data_dict,
    random_dict,
    post_method,
    rm_vals=None,
    compare_type="between",
    slopes_intercept=False,
):
    """Function to perform a linear mixed effects model test between groups across
    different time points (e.g., 2-way RM ANOVA like)

    INPUT PARAMETERS
        data_dict - dictionary with each key representing a group that contains
                    a 2d array with the data. Each column represent each sample, while
                    rows represent the repeated measures

        random_dict - dictionary with a list or 1d array of random variable values for
                      each sample in each group

        post_method - str specifying the posthoc test to perform

        rm_vals - list or array containing the values of the repeated measures.
                (e.g., 5, 10, 15um). If none, index will be used as the labels

        compare_type - what type of comparisons to be performed in posthoc testing.
                        Currently only coded for "between" to compare between
                        groups for each rm

    OUTPUT PARAMETERS
        model_df - DataFrame with the results of the model

        posthoc_df - DataFrame with the results of the posthoc tests

    """
    groups = list(data_dict.keys())
    # Set up the rm_Vals
    data_len = list(data_dict.values())[0].shape[0]
    if rm_vals is None:
        rm_vals = list(range(data_len))
    elif len(rm_vals) != data_len:
        print("Inputed RM Values do not match data!!")
        rm_vals = list(range(data_len))
    rm_vals = np.array(rm_vals).astype(int)
    rm_vals_groups = list(rm_vals.astype(str))

    # Organize the data in the appropriate format
    dfs = []
    for key, value in data_dict.items():
        for v in range(value.shape[1]):
            data = value[:, v]
            random_val = random_dict[key][v]
            g = [key for x in range(len(data))]
            rm = [random_val for x in range(len(data))]
            temp_dict = {
                "Rand_Var": rm,
                "Data": data,
                "Group": g,
                "RM_Vals": rm_vals,
            }
            temp_df = pd.DataFrame(temp_dict)
            dfs.append(temp_df)
    test_df = pd.concat(dfs)

    # Hot code group values
    map_dict = {}
    codes = list(range(len(data_dict.keys())))
    for i, key in enumerate(data_dict.keys()):
        map_dict[key] = codes[i]
    test_df = test_df.assign(Group_Coded=test_df["Group"].map(map_dict))
    # test_df = test_df.dropna(axis=0, how="any")
    test_df = test_df.fillna(0)

    test_df.reset_index(inplace=True, drop=True)

    # Perform the mixed-effects model test
    ## Define the formula
    # formula = "Data ~ Group_Coded + RM_Vals + Group_Coded:RM_Vals + (1|Rand_Var) + (Group_Coded-1|Rand_Var) + (RM_Vals:Group_Coded|Rand_Var)"
    if slopes_intercept == True:
        formula = "Data ~ Group + RM_Vals + Group:RM_Vals + (1|Rand_Var) + (Group - 1|Rand_Var)"
    else:
        formula = "Data ~ Group + RM_Vals + Group:RM_Vals + (1|Rand_Var)"
    # Construct the model and fit the model
    model = Lmer(formula, data=test_df)
    model.fit(
        summarize=False,
        factors={"Group": list(data_dict.keys()), "RM_Vals": rm_vals_groups},
        ordered=True,
    )
    fitted_data = model.fits
    test_df["Fitted Data"] = fitted_data
    # model_df = model.coefs
    model_df = model.anova(force_orthogonal=True)

    model_df.update(model_df[["SS"]].applymap("{:.3}".format))
    model_df.update(model_df[["MS"]].applymap("{:.3}".format))
    model_df.update(model_df[["F-stat"]].applymap("{:.3}".format))
    model_df.update(model_df[["P-val"]].applymap("{:.3E}".format))

    # Perform the posthoc tests
    # Between group comparisions
    if compare_type == "between":
        ## Get combinations of tests
        combos = list(itertools.combinations(groups, 2))
        test_performed = []
        rm_point = []
        t_vals = []
        raw_pvals = []
        for combo in combos:
            for rm in rm_vals:
                ## Keep track of the comparisions being made
                test_performed.append(combo[0] + " vs. " + combo[1])
                rm_point.append(rm)
                ## Get the data for the groups
                data1 = test_df[
                    (test_df["Group"] == combo[0]) & (test_df["RM_Vals"] == rm)
                ]
                data2 = test_df[
                    (test_df["Group"] == combo[1]) & (test_df["RM_Vals"] == rm)
                ]
                data1 = np.array(data1["Data"])
                data2 = np.array(data2["Data"])
                ## Perform the t-tests
                t, p = stats.ttest_ind(data1, data2)
                t_vals.append(t)
                raw_pvals.append(p)
        ## Correct for multiple comparisons
        _, adj_pvals, _, _ = multipletests(
            raw_pvals,
            alpha=0.05,
            method=post_method,
            is_sorted=False,
            returnsorted=False,
        )
        posthoc_dict = {
            "Comparision": test_performed,
            "RM Value": rm_point,
            "t stat": t_vals,
            "raw p-vals": np.array(raw_pvals),
            "adjusted p-vals": adj_pvals,
        }
        posthoc_df = pd.DataFrame.from_dict(posthoc_dict)

    # Format tables
    # for col in model_df.columns:
    #   model_df.update(model_df[[col]].applymap("{:.4}".format))

    # model_df.reset_index(inplace=True)

    posthoc_df.update(posthoc_df[["t stat"]].applymap("{:.4}".format))
    posthoc_df.update(posthoc_df[["raw p-vals"]].applymap("{:.3E}".format))
    posthoc_df.update(posthoc_df[["adjusted p-vals"]].applymap("{:.3E}".format))

    return model_df, posthoc_df


def one_way_ART(data_dict, random_dict, post_method="fdr_bh"):
    """Fuction to perform one-way aligned rank transform statistics
    for comparison across groups (e.g., t-test or one-way ANOVA like).
    Takes additional random effect parameter to control for (e.g., animal id).
    Equivalent to a non-parametric mixed effects model

    INPUT PARAMETERS
        data_dict - dict of data to be analyzed. Each item is a different group
                    Keys are group names, and values represent data points

        random_dict - dict containing the random variable corresponding to each
                        data sample. Keys are the same group names

    OUTPUT PARAMETERS
        t_stat - float of the t-stat from the mixed effects model test

        p_val - float of the p-val from the mixed effects model test

        results_df - pd.Dataframe of the posthoc test results

    """
    # Organize the data into a DataFrame
    ## Convert from dictionaries
    data_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
    rand_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in random_dict.items()]))
    ## Linearize the data into columnes
    data_df = pd.melt(
        data_df, value_vars=data_dict.keys(), var_name="Group", value_name="Data"
    )
    rand_df = pd.melt(
        rand_df, value_vars=random_dict.keys(), var_name="Group", value_name="Rand_Var"
    )
    ## Merge the random variable into the data dict
    data_df["Rand_Var"] = rand_df["Rand_Var"]
    ## Hot code the group values
    map_dict = {}
    codes = list(range(len(data_dict.keys())))
    for i, key in enumerate(data_dict.keys()):
        map_dict[key] = codes[i]
    data_df = data_df.assign(Group_Coded=data_df["Group"].map(map_dict))
    data_df = data_df.dropna(axis=0, how="any")

    # Store the dataframe to be read by R
    data_df.to_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\one_way_ART_input.csv"
    )
    # Run ART through R
    subprocess.run(
        [
            "Rscript",
            r"C:\Users\Jake\Desktop\python_code\Lab_Analyses\Lab_Analyses\Utilities\one_way_ART.r",
        ],
        capture_output=False,
    )

    # Read back in the result
    anova_df = pd.read_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\one_way_ART_anova.csv"
    )

    # Grab relevant values
    t_stat = anova_df.iloc[0]["F"]
    p_val = anova_df.iloc[0]["Pr(>F)"]

    # Perform the posthocs
    combos = list(itertools.combinations(data_dict.keys(), 2))
    test_performed = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        test_performed.append(combo[0] + " vs. " + combo[1])
        combo_1_data = data_dict[combo[0]]
        combo_2_data = data_dict[combo[1]]
        t, p = stats.mannwhitneyu(
            combo_1_data[~np.isnan(combo_1_data)], combo_2_data[~np.isnan(combo_2_data)]
        )
        t_vals.append(t)
        raw_pvals.append(p)
    # Perform corrections
    _, adj_pvals, _, alpha_corrected = multipletests(
        raw_pvals,
        alpha=0.05,
        method=post_method,
        is_sorted=False,
        returnsorted=False,
    )
    results_dict = {
        "comparison": test_performed,
        "t stat": t_vals,
        "raw p-vals": raw_pvals,
        "adjusted p-vals": adj_pvals,
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["raw p-vals"]].applymap("{:.4E}".format))
    results_df.update(results_df[["adjusted p-vals"]].applymap("{:.4E}".format))

    return t_stat, p_val, results_df


def two_way_RM_ART(data_dict, random_dict, rm_vals=None, post_method="fdr_bh"):
    """Function to perform a aligned rank transform corrected linear mixed-effect model
    test between groups across different time points (e.g., 2-way RM ANOVA like)

    INPUT PARAMETERS
        data_dict - dictionary with each key representing a group that contains
                    a 2d array with the data. Each column represents each sample,
                    while rows represent the repeated measures

        random_dict - dictionary with a list or 1d array of the random variable values
                    for each sample in each group

        rm_vals - list or array containing the values of the repeated measures
                (e.g., 5, 10, 15um). If none, index will be used as labels

    OUTPUT PARAMETERS
        anova_df - DataFrame witht the anova results of the model

        posthoc_df - DataFrame with the results of the posthoc tests

    """
    groups = list(data_dict.keys())
    # Set up the RM_Vals
    data_len = list(data_dict.values())[0].shape[0]
    if rm_vals is None:
        rm_vals = list(range(data_len))
    elif len(rm_vals) != data_len:
        print("Inputed RM Values do not match data!!")
        rm_vals = list(range(data_len))
    rm_vals = np.array(rm_vals).astype(int)

    # Organize data into the appropriate format
    dfs = []
    for key, value in data_dict.items():
        for v in range(value.shape[1]):
            data = value[:, v]
            random_val = random_dict[key][v]
            g = [key for x in range(len(data))]
            rm = [random_val for x in range(len(data))]
            temp_dict = {
                "Rand_Var": rm,
                "Data": data,
                "Group": g,
                "RM_Vals": rm_vals,
            }
            temp_df = pd.DataFrame(temp_dict)
            dfs.append(temp_df)
    test_df = pd.concat(dfs)

    # Hot code group values
    map_dict = {}
    codes = list(range(len(data_dict.keys())))
    for i, key in enumerate(data_dict.keys()):
        map_dict[key] = codes[i]
    test_df = test_df.assign(Group_Coded=test_df["Group"].map(map_dict))
    test_df = test_df.fillna(0)

    test_df.reset_index(inplace=True, drop=True)

    # Store the dataframe to be read by R
    test_df.to_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_input.csv"
    )
    # Run ART through R
    subprocess.run(
        [
            "Rscript",
            r"C:\Users\Jake\Desktop\python_code\Lab_Analyses\Lab_Analyses\Utilities\two_way_ART.r",
        ],
        capture_output=False,
    )

    # Read back in the result
    anova_df = pd.read_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_anova.csv"
    )

    # Format the output dataframes
    ## Reformat the anova table
    anova_df.update(anova_df[["F"]].applymap("{:.4}".format))
    anova_df.update(anova_df[["Df.res"]].applymap("{:.4}".format))
    anova_df.update(anova_df[["Pr(>F)"]].applymap("{:.4E}".format))

    ## Get combinations of tests
    combos = list(itertools.combinations(groups, 2))
    test_performed = []
    rm_point = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        for rm in rm_vals:
            ## Keep track of the comparisions being made
            test_performed.append(combo[0] + " vs. " + combo[1])
            rm_point.append(rm)
            ## Get the data for the groups
            data1 = test_df[(test_df["Group"] == combo[0]) & (test_df["RM_Vals"] == rm)]
            data2 = test_df[(test_df["Group"] == combo[1]) & (test_df["RM_Vals"] == rm)]
            data1 = np.array(data1["Data"])
            data2 = np.array(data2["Data"])
            ## Perform the t-tests
            t, p = stats.mannwhitneyu(data1, data2)
            t_vals.append(t)
            raw_pvals.append(p)
    ## Correct for multiple comparisons
    _, adj_pvals, _, _ = multipletests(
        raw_pvals,
        alpha=0.05,
        method=post_method,
        is_sorted=False,
        returnsorted=False,
    )
    posthoc_dict = {
        "Comparision": test_performed,
        "RM Value": rm_point,
        "t stat": t_vals,
        "raw p-vals": np.array(raw_pvals),
        "adjusted p-vals": adj_pvals,
    }
    posthoc_df = pd.DataFrame.from_dict(posthoc_dict)
    posthoc_df.update(posthoc_df[["t stat"]].applymap("{:.4}".format))
    posthoc_df.update(posthoc_df[["raw p-vals"]].applymap("{:.3E}".format))
    posthoc_df.update(posthoc_df[["adjusted p-vals"]].applymap("{:.3E}".format))

    return anova_df, posthoc_df


def two_way_RM_ART_ANOVA(data_dict, rm_vals=None, post_method="fdr_bh"):
    """Function to perform a aligned rank transform corrected Mixed ANOVA
    test between groups across different time points (e.g., 2-way RM ANOVA like)

    INPUT PARAMETERS
        data_dict - dictionary with each key representing a group that contains
                    a 2d array with the data. Each column represents each sample,
                    while rows represent the repeated measures

        random_dict - dictionary with a list or 1d array of the random variable values
                    for each sample in each group

        rm_vals - list or array containing the values of the repeated measures
                (e.g., 5, 10, 15um). If none, index will be used as labels

    OUTPUT PARAMETERS
        anova_df - DataFrame witht the anova results of the model

        posthoc_df - DataFrame with the results of the posthoc tests

    """
    groups = list(data_dict.keys())
    # Set up the RM_Vals
    data_len = list(data_dict.values())[0].shape[0]
    if rm_vals is None:
        rm_vals = list(range(data_len))
    elif len(rm_vals) != data_len:
        print("Inputed RM Values do not match data!!")
        rm_vals = list(range(data_len))
    rm_vals = np.array(rm_vals).astype(int)

    # Organize data into the appropriate format
    dfs = []
    sub_count = 1
    for key, value in data_dict.items():
        for v in range(value.shape[1]):
            data = value[:, v]
            g = [key for x in range(len(data))]
            sub = [sub_count for x in range(len(data))]
            temp_dict = {"Subject": sub, "Data": data, "Group": g, "RM_Vals": rm_vals}
            temp_df = pd.DataFrame(temp_dict)
            dfs.append(temp_df.dropna(axis=0))
            sub_count = sub_count + 1
    test_df = pd.concat(dfs)

    # Hot code group values
    map_dict = {}
    codes = list(range(len(data_dict.keys())))
    for i, key in enumerate(data_dict.keys()):
        map_dict[key] = codes[i]
    test_df = test_df.assign(Group_Coded=test_df["Group"].map(map_dict))
    # test_df = test_df.dropna(axis=0, how="any")
    test_df = test_df.fillna(0)

    test_df.reset_index(inplace=True, drop=True)

    # Store the dataframe to be read by R
    test_df.to_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_ANOVA_input.csv"
    )
    # Run ART through R
    subprocess.run(
        [
            "Rscript",
            r"C:\Users\Jake\Desktop\python_code\Lab_Analyses\Lab_Analyses\Utilities\two_way_ART_ANOVA.r",
        ],
        capture_output=False,
    )

    # Read back in the result
    anova_df = pd.read_csv(
        "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_ANOVA_anova.csv"
    )

    # Format the output dataframes
    ## Reformat the anova table
    anova_df.update(anova_df[["F value"]].applymap("{:.4}".format))
    anova_df.update(anova_df[["Sum Sq"]].applymap("{:.4}".format))
    anova_df.update(anova_df[["Sum Sq.res"]].applymap("{:.4}".format))
    anova_df.update(anova_df[["Pr(>F)"]].applymap("{:.4E}".format))

    ## Get combinations of tests
    combos = list(itertools.combinations(groups, 2))
    test_performed = []
    rm_point = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        for rm in rm_vals:
            ## Keep track of the comparisions being made
            test_performed.append(combo[0] + " vs. " + combo[1])
            rm_point.append(rm)
            ## Get the data for the groups
            data1 = test_df[(test_df["Group"] == combo[0]) & (test_df["RM_Vals"] == rm)]
            data2 = test_df[(test_df["Group"] == combo[1]) & (test_df["RM_Vals"] == rm)]
            data1 = np.array(data1["Data"])
            data2 = np.array(data2["Data"])
            ## Perform the t-tests
            t, p = stats.mannwhitneyu(data1, data2)
            t_vals.append(t)
            raw_pvals.append(p)
    ## Correct for multiple comparisons
    _, adj_pvals, _, _ = multipletests(
        raw_pvals,
        alpha=0.05,
        method=post_method,
        is_sorted=False,
        returnsorted=False,
    )
    posthoc_dict = {
        "Comparision": test_performed,
        "RM Value": rm_point,
        "t stat": t_vals,
        "raw p-vals": np.array(raw_pvals),
        "adjusted p-vals": adj_pvals,
    }
    posthoc_df = pd.DataFrame.from_dict(posthoc_dict)
    posthoc_df.update(posthoc_df[["t stat"]].applymap("{:.4}".format))
    posthoc_df.update(posthoc_df[["raw p-vals"]].applymap("{:.3E}".format))
    posthoc_df.update(posthoc_df[["adjusted p-vals"]].applymap("{:.3E}".format))

    return anova_df, posthoc_df
