import numpy as np
import pandas as pd
from scipy import stats
from itertools import compress
import random
import utilities as util
import opto_plotting as plotting
from IPython.display import display
from tabulate import tabulate
import seaborn as sns; sns.set()
sns.set_style('ticks')

''' Module containing several classes to analyze optogenetic stimulation data.
    Tests for ROIs significantly activated by stimulation while also providing
    a variety of visualizations
    
    CREATOR 
        William (Jake) Wright 12/02/2021'''

class opto_analysis():
    '''Class to analyze a single optogenetic stimulation session. Tests if ROIs
        are significantly activated by stimulation. Also includes methods to get
        summaried activity and generate plots for visualization
        
        Can be called in isolation to inspect single sessions, but is also utilized
        by other classes for grouped analyses'''

    def __init__(self, imaging_data, behavior_data, sampling_rate=30, 
                 window=[-2,2], vis_window=None, stim_len=1, zscore=False, spines=False):
        ''' __init__ - Initialize opto_analysis Class.
        
            INPUT PARAMETERS
                imaging_data - dictionary of the calcium imaging data output from 
                                the load_mat functions.
                behavior_data - dictionary of the behavior data output from the 
                                load_mat functions.
                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.
                window - list specifying the time before and after opto stim onset you
                        want to analyze. E.g. [-2,2] for 2s before and after. Default
                        set to [-2,2].
                vis_window - same as window, but for visualizing the data only
                            Default is set to none, in which case vis_window will
                            be determined by window and stim_len.
                stim_len - scaler specifying how long opto stim was delivered for.
                            Default is set to 1s.
                zscore - boolean True or False of whether to zscore the data.
                spines - boolean True or False of whether the ROIs are spines.
                
                '''

        # Storing input data
        self.imaging = imaging_data
        self.behavior = behavior_data
        self.sampling_rate = sampling_rate
        self.window = window
        self.before_t = window[0] # before window in time(s)
        self.before_f = window[0]*sampling_rate # before window in frames
        self.after_t = window[1]
        self.after_f = window[1]*sampling_rate
        self.vis_window = vis_window
        if vis_window is not None:
            self.vis_after_f = vis_window[1]*sampling_rate
        self.stim_len = stim_len
        self.stim_len_f = stim_len*sampling_rate
        self.zscore = zscore
        self.spines = spines

        # Organizing imaging and behavior data to be used
        if spines is False:
            ROIs = []
            for i in list(imaging_data['ROIs'][:-1]):
                ROIs.append('Cell ' + str(i))
            dFoF = pd.DataFrame(data=imaging_data['processed_dFoF'].T,columns=ROIs)
        else:
            ROIs = []
            for i in list(imaging_data['Spine_ROIs'][:-1]):
                ROIs.append('Spine ' + str(i))
            dFoF = pd.DataFrame(data=imaging_data['spine_processed_dFoF'].T,columns=ROIs)
        if zscore is True:
            self.dFoF = util.zscore(dFoF)
        else:
            self.dFoF = dFoF
        
        self.ROIs = ROIs

        # Select trials that were imaged
        i_trials = behavior_data['imaged_trials']
        i_trials = i_trials == 1
        self.i_trials = i_trials
        behavior = list(compress(behavior_data['behavior_frames'],i_trials))
        self.behavior = behavior
        itis = []
        for i in behavior:
            itis.append(i['states']['iti2'])
        
        # Check iti intervals are consistent and within imaging period
        longest_window = max([self.after_f+self.stim_len_f, self.vis_after_f])
        for i, _ in enumerate(itis):
            if itis[i][1] + longest_window > len(dFoF):
                itis = itis[0:i-1]
            else:
                pass
        for i, _ in enumerate(itis):
            if (itis[i][1] - itis[i][0]) - self.stim_len_f == 1:
                itis[i] = [itis[i][0],itis[i][1]-1]
            elif (itis[i][1] - itis[i][0]) - self.stim_len_f == -1:
                itis[i] = [itis[i][0],itis[i][1]+1]
            elif (itis[i][1] - itis[i][0]) == self.stim_len_f:
                itis[i] = itis[i]
            else:
                del itis[i]
        self.itis = itis

        # Attributes to be defined
        self.method = None
        self.sig_results_dict = None
        self.sig_results_df = None
        self.all_befores = None
        self.all_afters = None
        self.roi_stim_epochs = None
        self.roi_mean_sems = None

    def opto_before_after_means(self, data=None,single=None):
        '''Method to get mean activity before and after stim'''
        if data is None:
            data = self.dFoF
        else:
            data = data
        if single is None:
            single=False
        else:
            single=single
        all_befores, all_afters = util.get_before_after_means(activity=data,
                                                               timestamps=self.itis,
                                                               window=self.window,
                                                               sampling_rate=self.sampling_rate,
                                                               offset=False,
                                                               single=single)
        self.all_befores = all_befores
        self.all_afters = all_afters
        
        return all_befores, all_afters
    
    def opto_trace_mean_sems(self,data=None):
        '''Method to get the activity of each ROI for each trial as
            well as the mean and sem across all trials.'''
        