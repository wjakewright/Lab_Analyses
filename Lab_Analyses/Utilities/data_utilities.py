"""Module containing various functions that help with data handeling"""

import numpy as np
import pandas as pd
from scipy import stats


def get_before_after_means(
    activity, timestamps, window, sampling_rate, offset=False,
):
    """Function to get the mean activity before and after a specific behavioral
        event
        
        INPUT PARAMETERS
            activity - np.array of neural activity, with each column representing a single
                        ROI. Can also be a single column vector for a single ROI
            
            timestamps - list of timestamps corresponding to the imaging frame where
                         each behavioral event occurred. 
                         
            window - list specifying the time before and after the behavioral event
                    you want to assess (e.g., [-2, 2] for 2 sec before and after)
            
            sampling_rate - scaler specifying the image sampling rate
            
            offset - boolean term indicating if the timestamps include the offset of behavioral
                    event and if this should be used to determine the after period. Default
                    is set to False. 
            
        OUTPUT PARAMETERS
            all_befores - list containing all the before means (np.array) for each ROI
            
            all_afters - list containing all the after means (np.array) for each ROI
    """
    # Organize the timestampse
    if offset == False:
        if len(timestamps[0]) == 1:
            stamp1 = [int(x) for x in timestamps]
            stamp2 = [int(x) for x in timestamps]
        elif len(timestamps[0]) == 2:
            stamps = []
            for i in timestamps:
                stamps.append(int(i[0]))
            stamp1 = stamps
            stamp2 = stamps
        else:
            return print("Too many indicies in each tiemstamp!!!")
    else:
        stamp1 = []
        stamp2 = []
        for i in timestamps:
            stamp1.append(int(i[0]))
            stamp2.append(int(i[1]))

    # Get before and after windows in terms of frames
    before_f = int(window[0] * sampling_rate)
    after_f = int(window[1] * sampling_rate)

    all_befores = []
    all_afters = []
    # Get the before an after values
    for i in range(activity.shape[1]):
        d = activity[:, i]
        before_values = []
        after_values = []
        for s1, s2 in zip(stamp1, stamp2):
            before_values.append(np.mean(d[s1 + before_f : s1]))
            after_values.append(np.mean(d[s2 : s2 + after_f + 1]))
        all_befores.append(np.array(before_values))
        all_afters.append(np.array(after_values))

    return all_befores, all_afters


def get_before_during_means(activity, timestamps, window, sampling_rate):
    """Function to get the before and during behavioral event
    
        INPUT PARAMETERS
            activity - np.array of neural activity, with each column representing ROI
            
            timestamps - list of tuples with the onset and offest of each behavioral event
            
            window - int specifying the timewindow before the behavioral event 
                    to use as the baseline in seconds
                    
            sampling_rate - float specifying what the imaging rate was

        OUTPUT PARAMETERS
            all_befores - list containing all the before means (np.array) for each ROI
            
            all_durings - list containing all the during means (np.array) for each ROI
            
    """
    # Get the before time in frames
    before_f = int(window * sampling_rate)

    all_befores = []
    all_durings = []
    # Get before and after values
    for i in range(activity.shape[1]):
        d = activity[:, i]
        before_values = []
        during_values = []
        for stamp in timestamps:
            before_values.append(np.mean(d[stamp[0] - before_f : stamp[0]]))
            during_values.append(np.mean(d[stamp[0] : stamp[1]]))
        all_befores.append(np.array(before_values))
        all_durings.append(np.array(during_values))

    return all_befores, all_durings


def get_trace_mean_sem(activity, ROI_ids, timestamps, window, sampling_rate):
    """Function to get the mean and sem of neural activity around behavioral events
        
        INPUT PARAMETERS
            activity - np.array of neural activity, with each column for each ROI

            ROI_ids - list of strings with each ROI id
            
            timestamps - list of timestamps corresponding to the imaging frame where
                        each event occured
                        
            window - list specifying the time before and after the behavioral event
                     you want to assess (e.g. [-2,2] for 2 secs before and after
                     
            sampling_rate - scaler specifying the image sampling rate
        
        OUTPUT PARAMETERS
            roi_event_epochs - dictionary containing an array for each ROI. Each array
                              contains the activity during the window for each event, 
                              with a column for each event
            
            roi_mean_sems - dictionary containing the activity mean and sem for each ROI.
                            For each ROI key there are two arrays, one for the mean activity
                            during the event, and the other for the sem during the same period
    """
    # Get the window in terms of frames
    before_f = int(window[0] * sampling_rate)
    after_f = int(window[1] * sampling_rate)
    win_size = -before_f + after_f

    # Get each event epoch
    roi_event_epochs = {}
    for i in range(activity.shape[1]):
        d = activity[:, i]
        roi = ROI_ids[i]
        epochs = np.zeros(win_size).reshape(-1, 1)
        for t in timestamps:
            t = int(t)
            e = d[t + before_f : t + after_f].reshape(-1, 1)
            epochs = np.hstack((epochs, e))
        epochs = epochs[:, 1:]
        roi_event_epochs[roi] = epochs

    # Get the mean and sem of the traces
    roi_mean_sems = {}
    for key, value in roi_event_epochs.items():
        m = np.mean(value, axis=1)
        sem = stats.sem(value, axis=1)
        roi_mean_sems[key] = [m, sem]

    return roi_event_epochs, roi_mean_sems


def z_score(data):
    """Function to z-score the data
    
        INPUT PARAMETERS
            data - np.array of neural data with each column representing a seperate roi
            
        OUTPUT PARAMETERS
            z_data - np.array of z-scored data
    """

    # initialize the output variable
    z_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        z = (data[:, i] - data[:, i].mean()) / data[:, i].std(ddof=0)
        z_data[:, i] = z

    return z_data


def zero_window(data, base_win, sampling_rate):
    """Fucntion to zero neural activity to a specified baseline period for trial
        activity. For plotting purposes
        
        INPUT PARAMETERS
            data - dataframe or array of neural activity, with each column representing a single ROI
            
            base_win - tuple of ints spcifying the period within the trial you wish to 
                        define as the baseline period in terms of seconds. 
            
            sampling_rate - int specifying the imaging rate 
        
        OUTPUT PARAMETERS
            zeroed_data - dataframe of the now zeroed data
    """

    # get baseline period in terms of frames
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)
    zero_b = int(base_win[0] * sampling_rate)
    zero_a = int(base_win[1] * sampling_rate)
    zeroed_data = []
    for col in data.columns:
        zeroed_data.append(
            np.array(data[col].sub(np.nanmedian(data[col].loc[zero_b:zero_a]))).reshape(
                -1, 1
            )
        )
    zeroed_data = pd.DataFrame(np.hstack(zeroed_data))

    return zeroed_data


def diff_sorting(data, length, sampling_rate):
    """Function to sort neurons based on their change in activity
        around a specified event 
        
        INPUT PARAMETERS
            data - array or dataframe of neural activity, with each row
                    representing a single neuron or trial (for a single neuron). 
                    Note data must be centered around the event
            
            length - tuple of ints specifying how long before and after
                     the event to compare (e.g. (2,2) for 2s before vs
                    2s after)
            
            sampling_rate - int of the imaging sampling rate (e.g. 30hz)
            
        OUTPUT PARAMETERS
            sorted_data - dataframe of the sorted neural activity
            
    """
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)
    b_len = int(length[0] * sampling_rate)
    a_len = int(length[1] * sampling_rate)

    before = data.iloc[:, :b_len].mean(axis=1)
    after = data.iloc[:, b_len : b_len + a_len].mean(axis=1)
    differences = []
    for before_i, after_i in zip(before, after):
        differences.append(after_i - before_i)
    data["Diffs"] = differences
    sorted_data = data.sort_values(by="Diffs", ascending=False)
    sorted_data = sorted_data.drop(columns=["Diffs"])

    return sorted_data


def join_dictionaries(dict_list):
    """Function to join a list of dictionaries with the same keys into a single
        contiuous dictionary"""
    new_dict = dict_list[0]
    for d in dict_list[1:]:
        for key, value in d.items():
            if type(value) != list:
                new_dict[key] = np.concatenate((new_dict[key], value))
            else:
                for i, v in enumerate(value):
                    new_dict[key][i] = np.concatenate((new_dict[key][i], v))

    return new_dict
