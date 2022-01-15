import itertools
import random

import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate


def get_before_after_means(
    activity, timestamps, window, sampling_rate=30, offset=False, single=False
):

    """ Function to get the mean activity before and after a specific
        behavioral event.

        CREATOR
            William (Jake) Wright 10/11/2021

        USAGE
            all_befores, all_afters = get_before_after_mean(activity,timestamps,window,sampling_rate)

        INPUT PARAMETERS
            activity - dataframe of neural activity, with each column
                       corresponding to each ROI. Can also be a single column vector
                       for a single ROI

            timestamps - a list of timestamps corresponding to the imaging
                         frame where each behavioral event occured

            window - list specifying the time before and after the behavioral
                     event you want to assess (e.g. [-2,2] for 2 secs before
                     and after.

            sampling_rate - scaler specifying the image sampling rate. Default
                            is set to 30hz.

            offset - boolean term indicating if the timestamps include the offset of behavioral
                     event and if this should be used to determine the after period. Default is
                     set to false.
            
            single - boolean term indicating if the input data is a single ROI or a dataframe. 
                     Default is set to False to read a dataframe.

        OUTPUT PARAMETERS
            all_befores - a list containing all the before means for each ROI

            all_afters - a list constining all the after means for each ROI """

    if offset == False:
        if len(timestamps[0]) == 1:
            stamp1 = timestamps
            stamp2 = timestamps
        elif len(timestamps[0]) == 2:
            stamps = []
            for i in timestamps:
                stamps.append(i[0])
            stamp1 = stamps
            stamp2 = stamps
        else:
            return print("Too many indicies in each timestamps !!!")
    else:
        stamp1 = []
        stamp2 = []
        for i in timestamps:
            stamp1.append(i[0])
            stamp2.append(i[1])
    before_f = window[0] * sampling_rate
    after_f = window[1] * sampling_rate

    all_befores = []
    all_afters = []
    if single is False:
        for j in activity.columns:
            d = activity[j]
            before_values = []
            after_values = []
            for s1, s2 in zip(stamp1, stamp2):
                before_values.append(np.mean(d[s1 + before_f : s1]))
                after_values.append(np.mean(d[s2 : s2 + after_f]))
            all_befores.append(before_values)
            all_afters.append(after_values)

    else:
        d = activity
        before_values = []
        after_values = []
        for s1, s2 in zip(stamp1, stamp2):
            before_values.append(np.mean(d[s1 + before_f : s1]))
            after_values.append(np.mean(d[s2 : s2 + after_f]))
        all_befores.append(before_values)
        all_afters.append(after_values)

    return all_befores, all_afters


def get_trace_mean_sem(activity, timestamps, window, sampling_rate=30):
    """ Function to get the mean and sem of neural activity around timelocked behavioral
        events.

        CREATOR
            William (Jake) Wright 10/11/2021

        USAGE
            roi_stim_epochs, roi_mean_sems = get_trace_mean_sem(inputs)

        INPUT PARAMETERS
            activity - dataframe of neural activity, with each column
                       corresponding to each ROI

            timestamps - a list of timestamps corresponding to the imaging
                         frame where each behavioral event occured

            window - list specifying the time before and after the behavioral
                     event you want to assess (e.g. [-2,2] for 2 secs before
                     and after.

            sampling_rate - scaler specifying the image sampling rate. Default
                            is set to 30hz.

        OUTPUT PARAMETERS
            roi_stim_epochs - dictionary containing an array for each roi. Each array
                              contains the activity during the window for each behavioral
                              event, with a column for each event

            roi_mean_sems - dictionary containing the activity mean and sem for each ROI.
                            For each ROI key there are two lists, one for the mean activity
                            during the mean, and the other for the sem during the same
                            period. """

    # first get the window size
    before = window[0] * sampling_rate
    after = window[1] * sampling_rate
    win_size = -before + after
    roi_stim_epochs = {}
    for col in activity.columns:
        d = activity[col]
        epochs = np.zeros(win_size).reshape(-1, 1)
        for i in timestamps:
            e = np.array(d[i + before : i + after]).reshape(-1, 1)
            epochs = np.hstack((epochs, e))
        epochs = epochs[:, 1:]
        roi_stim_epochs[col] = epochs

    # Get mean and sem of the traces
    roi_mean_sems = {}
    for key, value in roi_stim_epochs.items():
        m = np.mean(value, axis=1)
        sem = stats.sem(value, axis=1)
        roi_mean_sems[key] = [m, sem]

    return roi_stim_epochs, roi_mean_sems


def z_score(data):
    """ Function to z-score the dat
    
        INPUT PARAMETERS
            data - dataframe of neural data. Each column represents a seperate
                   neuron.
        
        OUTPUT PARAMATERS
            z_data - dataframe of z-scored neural data. Same format as input"""

    cols = data.columns
    z_data = pd.DataFrame()
    for col in cols:
        z_data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)

    return z_data


def ANOVA_1way_bonferroni(data_dict, method):
    """Function to perform a one way ANOVA with posttests
    
        CREATOR
            William (Jake) Wright   11/22/2021
        
        INPUT PARAMETERS
            data_dict  -  dictionary of the data to be analyzed. Each item is 
                          a different group. Keys represent group names, and 
                          values represent data points of the group
            
            method  -  string specifying which posttest to perform. See documentation
                        for statsmodels.stats.multitest for available methods
                        
        OUTPUT PAREMETERS
            f_stat  -  the f statistic from the one way ANOVA

            anova_p  -  p-value of the one way ANOVA

            results_table  -  table of the results of the posttest"""

    # Perform one-way ANOVA
    data_array = np.array(list(data_dict.values()))
    f_stat, anova_p = stats.f_oneway(*data_array)

    # Perform t-test across all groups
    combos = list(itertools.combinations(data_dict.keys(), 2))
    test_performed = []
    t_vals = []
    raw_pvals = []
    for combo in combos:
        test_performed.append(combo[0] + " vs." + combo[1])
        t, p = stats.ttest_ind(data_dict[combo[0]], data_dict[combo[1]])
        t_vals.append(t)
        raw_pvals.append(p)
    # Peform multiple comparisons correction
    # Set up for Bonferroni at the moment
    _, adj_pvals, _, alpha_corrected = multipletests(
        raw_pvals, alpha=0.05, method=method, is_sorted=False, returnsorted=False
    )
    results_dict = {
        "comparison": test_performed,
        "t stat": t_vals,
        "raw p-values": raw_pvals,
        "adjusted p-vals": adj_pvals,
    }

    results_df = pd.DataFrame.from_dict(results_dict)

    results_table = tabulate(results_dict, headers="keys", tablefmt="fancy_grid")

    return f_stat, anova_p, results_table, results_df


def significance_testing(imaging, timestamps, window, sampling_rate, method):
    """Function to determine if each ROI was significantly activated by a specifice event.
    
        INPUT PARAMETERS
            imaging - dataframe of the imaging data, with each column representing
                      an ROI
            
            timestamps - list or array of timestamps corresponding to the imaging
                        frame where event occured

            method - string specifying which method is to be used to test
                    significance. Currnetly coded to accept:
                        
                        'test' - Performs Wilcoxon Signed-Rank Test
                        'shuff' - Compares the real difference in activity against
                                    a shuffled distribution
        
        OUPTPUT PARAMETERS
            'test' method returns two outputs and 'shuff' method returns 3

            results_dict - dictionary containing the results for each ROI (keys)

            results_df - DataFrame containing the results for each ROI (columns)

            shuff_diffs - ('shuff' method only) list containing all the shuffled
                            differences in activity for each ROI
            """
    ROIs = imaging.columns
    shuff_diffs = []
    if method == "test":
        # Get the means before and after event of interest
        befores, afters = get_before_after_means(
            activity=imaging,
            timestamps=timestamps,
            window=window,
            sampling_rate=sampling_rate,
            offset=False,
            single=False,
        )
        pValues = []
        rankValues = []
        diffs = []
        # Perform testing and get mean difference in activity
        for before, after in zip(befores, afters):
            rank, pVal = stats.wilcoxon(after, before)
            pValues.append(pVal)
            rankValues.append(rank)
            diffs.append(np.mean(np.array(after) - np.array(before)))
        # Assess signficance
        sig = (np.array(pValues) < 0.01) * 1

        # Put results in dictionary
        results_dict = {}
        for p, r, d, s, ROI in zip(pValues, rankValues, diffs, sig, ROIs):
            results_dict[ROI] = {"pvalue": p, "rank": r, "diff": d, "sig": s}

    elif method == "shuff":
        data = imaging.copy()
        real_diffs = []
        shuff_diffs = []
        bounds = []
        sigs = []
        smallest = sampling_rate  # smallest shift of data is 1s
        biggest = 300 * sampling_rate  # biggest shift of data is 5min

        # Assess each ROI individually
        for col in data.columns:
            d = data[col]

            # Get the real difference
            before, after = get_before_after_means(
                activity=d,
                timestamps=timestamps,
                window=window,
                sampling_rate=sampling_rate,
                offset=False,
                single=True,
            )
            r_diff = np.mean(np.array(after) - np.array(before))
            # Perform 1000 shuffles
            s_diffs = []
            for i in range(1000):
                n = random.randint(smallest, biggest)
                s_d = np.copy(d)
                shuff_d = np.roll(s_d, n)
                s_before, s_after = get_before_after_means(
                    activity=shuff_d,
                    timestamps=timestamps,
                    window=window,
                    sampling_rate=sampling_rate,
                    offset=False,
                    single=True,
                )
                s_diffs.append(np.mean(np.array(s_after) - np.array(s_before)))
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
        # Put results in dictionary
        results_dict = {}
        for ROI, r, s, b, sig in zip(ROIs, real_diffs, shuff_diffs, bounds, sigs):
            results_dict[ROI] = {"diff": r, "shuff_diffs": s, "bounds": b, "sig": sig}

    else:
        return "Not a valid testing method specified!!!"

    # Generate a dataframe for results
    results_df = pd.DataFrame.from_dict(results_dict, orient="index")
    if "shuff_diff" in results_df.columns:
        results_df = results_df.drop(columns=["shuff_diffs"])

    return results_dict, results_df, shuff_diffs

