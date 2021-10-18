import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats


def get_before_after_means(activity, timestamps, window,
                           sampling_rate=30, offset=False, single=False):

    ''' Function to get the mean activity before and after a specific
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

            all_afters - a list constining all the after means for each ROI '''

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
            return print('Too many indicies in each timestamps !!!')
    else:
        stamp1 = []
        stamp2 = []
        for i in timestamps:
            stamp1.append(i[0])
            stamp2.append(i[1])
    before_f = window[0]*sampling_rate
    after_f = window[1]*sampling_rate

    all_befores = []
    all_afters = []
    if single is False:
        for j in activity.columns:
            d = activity[j]
            before_values = []
            after_values = []
            for s1, s2 in zip(stamp1,stamp2):
                before_values.append(np.mean(d[s1+before_f:s1]))
                after_values.append(np.mean(d[s2:s2+after_f]))
            all_befores.append(before_values)
            all_afters.append(after_values)
            
    else:
        d = activity
        before_values = []
        after_values = []
        for s1, s2 in zip(stamp1,stamp2):
            before_values.append(np.mean(d[s1+before_f:s1]))
            after_values.append(np.mean(d[s2:s2+after_f]))
        all_befores.append(before_values)
        all_afters.append(after_values)

    return all_befores, all_afters


def get_trace_mean_sem(activity,timestamps,window,sampling_rate=30):
    ''' Function to get the mean and sem of neural activity around timelocked behavioral
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
                            period. '''


    # first get the window size
    before = window[0]*sampling_rate
    after = window[1]*sampling_rate
    win_size = -before + after
    roi_stim_epochs = {}
    for col in activity.columns:
        d = activity[col]
        epochs = np.zeros(win_size).reshape(-1,1)
        for i in timestamps:
            e = np.array(d[i+before:i+after]).reshape(-1,1)
            epochs = np.hstack((epochs,e))
        epochs = epochs[:,1:]
        roi_stim_epochs[col] = epochs

    # Get mean and sem of the traces
    roi_mean_sems = {}
    for key,value in roi_stim_epochs.items():
        m = np.mean(value,axis=1)
        sem = stats.sem(value,axis=1)
        roi_mean_sems[key] = [m,sem]

    return roi_stim_epochs, roi_mean_sems

def z_score(data):
    ''' Function to z-score the dat
    
        INPUT PARAMETERS
            data - dataframe of neural data. Each column represents a seperate
                   neuron.
        
        OUTPUT PARAMATERS
            z_data - dataframe of z-scored neural data. Same format as input'''
            
    cols = data.columns
    z_data = pd.DataFrame()
    for col in cols:
        z_data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
    
    return z_data
    

