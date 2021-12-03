import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
from itertools import compress
import random
import utilities as util
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import opto_plotting as plotting
sns.set_style('ticks')


class population_opto_analysis():
    '''Class to analyze single optogenetic stimulation sessoins. Tests if ROIs
        are significantly activated by stimulation. Also includes methods to 
        get summarized activity and generate plots for visualization'''

    def __init__(self, imaging_data, behavior_data, sampling_rate=30,
                 window = [-2,2], stim_len=1,zscore=False,spines=False):
        '''__init__- Initilize population_opto_analysis Class.

            CREATOR
                William (Jake) Wright 10/7/2021

            INPUT PARAMETERS
                imaging_data - dictionary of the calcium imaging data outpt from
                               the load_mat functions. More than one session is required for
                               power_curve method.

                behavior_data - dictionary of the behavior data output from the
                                load_mat functions.

                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.

                window - list specifying the time before and after the opto stim
                         you want to visualize and assess (e.g. [-2,2] for two sec
                         before and after). Default is set to [-2,2].

                stim_len - scaler specifying how long opto stim was delivered for.
                           Default is set to 1 sec

                zscore - boolean True or False of wheather to zscore the data.
        '''

        # Storing the initial input data
        self.imaging = imaging_data
        self.behavior = behavior_data
        self.sampling_rate = sampling_rate
        self.window = window
        self.before_t = window[0] # before window in time(s)
        self.before_f = window[0]*sampling_rate # before window in frames
        self.after_t = window[1]
        self.after_f = window[1]*sampling_rate
        self.stim_len = stim_len
        self.stim_len_f = stim_len*sampling_rate
        self.zscore = zscore

        # Pulling data from the inputs that will be used
        if spines is False:
            ROIs = []
            for i in list(imaging_data.SpineROIs[:-1]):
                ROIs.append('Cell ' + str(np.floor(i)))
            dFoF = pd.DataFrame(data=imaging_data.Processed_dFoF.T,columns=ROIs)
        else:
            ROIs = []
            for i in list(imaging_data.SpineROIs[:-1]):
                ROIs.append('Spine ' + str(np.floor(i)))
            dFoF = pd.DataFrame(data=imaging_data.Processed_dFoF.T,columns=ROIs)
        if zscore is True:
            self.dFoF = util.z_score(dFoF)
        else:
            self.dFoF = dFoF

        self.ROIs = ROIs
        #self.dFoF = dFoF

        # Select the trials that were imaged
        i_trials = behavior_data.Imaged_Trials
        i_trials = i_trials == 1
        behavior = list(compress(behavior_data.Behavior_Frames,i_trials))
        itis = []
        for i in behavior:
            itis.append(i.states.iti2)

        # Making sure the iti intervals are consistent and within imaging period
        for i in range(len(itis)):
            if itis[i][1] + self.after_f + self.stim_len_f > len(dFoF):
                itis = itis[0:i-1]
            else:
                pass
        for i in range(len(itis)):
            if (itis[i][1] - itis[i][0]) - self.stim_len_f == 1:
                itis[i] = [itis[i][0],itis[i][1]-1]
            elif (itis[i][1] - itis[i][0]) - self.stim_len_f == -1:
                itis[i] = [itis[i][0],itis[i][1]+1]
            elif (itis[i][1] - itis[i][0]) == self.stim_len_f:
                itis[i] = itis[i]
            else:
                del itis[i]
        

        self.i_trials = i_trials
        self.behavior = behavior
        self.itis = itis
        # Attributes to be defined
        self.method = None
        self.sig_results = None
        self.all_befores = None
        self.all_afters = None
        self.roi_stim_epochs = None
        self.roi_mean_sems = None
        self.sig_results_dict = None
        self.sig_results_df = None
    
    def opto_before_after_means(self, data=None,single=None):
        '''Method to get mean activity before and after stim'''
        if data is None:
            data = self.dFoF
        else:
            data = data
        if single is None:
            single = False
        else:
            single = single
        all_befores, all_afters = util.get_before_after_means(activity=data,
                                                              timestamps=self.itis,
                                                              window=self.window,
                                                              sampling_rate=self.sampling_rate,
                                                              offset=False,single=single)
        self.all_befores = all_befores
        self.all_afters = all_afters

        return all_befores, all_afters

    def opto_trace_mean_sems(self,data=None):
        '''Method to get the activity of each ROI for each trial
            as well as the mean and sem across all trials'''
        if data is None:
            data = self.dFoF
        else:
            data = data
  
        new_timestamps = []
        for i in self.itis:
            new_timestamps.append(i[0])
        new_window = [self.window[0],self.window[1]+self.stim_len]
        self.new_window = new_window
        
        roi_stim_epochs,roi_mean_sems = util.get_trace_mean_sem(activity=data,
                                                                timestamps=new_timestamps,
                                                                window=new_window,
                                                                sampling_rate=self.sampling_rate)
        self.roi_stim_epochs = roi_stim_epochs
        self.roi_mean_sems = roi_mean_sems

        return roi_stim_epochs, roi_mean_sems
    
    def significance_testing(self, method):
        ''' Method to determine if each ROI was significantly activated by
            optogenetic stimulation.

            INPUT PARAMETERS
                method - string specifying which method is to be used to test
                         significance. Currently coded to accept:
                         
                             'test' - Performs Wilcoxon Signed-Rank Test
                             'shuff' - compares the real difference in activity
                                       against a shuffled distribution '''
        self.method = method
        if self.all_befores is None:
            before_m, after_m = self.opto_before_after_means()
        else:
            before_m = self.all_befores
            after_m = self.all_afters
        
        # Preform significance testing
        if method == 'test':
            pValues = []
            rankValues = []
            diffs = []
            for before, after in zip(before_m,after_m):
                b = before
                a = after

                # Perform Wilcoxon signed-rank test; significance set at 0.01
                rank, pVal = stats.wilcoxon(a,b)
                pValues.append(pVal)
                rankValues.append(rank)
                diffs.append(np.mean(np.array(a)-np.array(b)))
            # Group in dictionaries
            pValue_dict = dict(zip(self.dFoF.columns,pValues))
            rank_dict = dict(zip(self.dFoF.columns,rankValues))
            diff_dict = dict(zip(self.dFoF.columns,diffs))

            # Assess significance
            sig = (np.array(list(pValue_dict.values())) < 0.01) * 1
            sig_dict = dict(zip(self.dFoF.columns,sig))

            sig_results = {}
            for key,value in pValue_dict.items():
                sig_results[key] = {'pvalue':value,
                                   'rank':rank_dict[key],
                                   'diff':diff_dict[key],
                                   'sig':sig_dict[key]}
        
        elif method == 'shuff':
            data = self.dFoF.copy()
            real_diffs = [] 
            shuff_diffs = [] 
            bounds = [] 
            sigs = []
            smallest = self.sampling_rate # smallest shift is 1 second
            biggest = 300 * self.sampling_rate # biggest shift is 5 min

            # Assess each ROI individually
            for col in data.columns:
                d = self.dFoF[col]
                
                # Get the real difference
                before, after = self.opto_before_after_means(data=d,single=True)
                r_diff = np.mean(np.array(after)-np.array(before))
                # Perform 1000 shuffles
                s_diffs = []
                for i in range(1000):
                    n = random.randint(smallest,biggest)
                    s_d = np.copy(d)
                    shuff_d = np.roll(s_d,n)
                    
                    shuff_before, shuff_after = self.opto_before_after_means(data=shuff_d,
                                                                             single=True)
                    s_diffs.append(np.mean(np.array(shuff_after)-np.array(shuff_before)))
                
                # Assess significance
                upper = np.percentile(s_diffs, 99)
                lower = np.percentile(s_diffs, 1)
                
                if lower <= r_diff <= upper:
                    sig = 0
                else:
                    sig = 1
                
                # Append values for each ROI
                real_diffs.append(r_diff)
                shuff_diffs.append(s_diffs)
                bounds.append((upper,lower))
                sigs.append(sig)
             # Put final results in dictionary
            sig_results = {}
            for col, real, shuff, b, sig in zip(data.columns,real_diffs,
                                                      shuff_diffs,bounds,sigs):
                sig_results[col] = {'real_diff':real,'shuff_diff':shuff,
                                    'bounds':b, 'sig':sig}
        else:
            return ('Not a valid testing method indicated!!!')
        self.sig_results_dict = sig_results

        results_df = pd.DataFrame.from_dict(sig_results,orient='index')
        if 'shuff_diff' in results_df.columns:
            results_df = results_df.drop(columns=['shuff_diff'])
        self.sig_results_df = results_df
        return sig_results
    
    def plot_sess_activity(self,figsize=(7,8), title='default'):
        plotting.plot_session_activity(self.dFoF, self.itis, zscore=self.zscore, figsize=figsize, title=title)
    
    def plot_each_stim(self, figsize=(10,20),title='default'):
        # Get each stimulation epoch for each ROI first
        roi_stim_epochs, _ = self.opto_trace_mean_sems()
        ROIs = self.ROIs
        plotting.plot_each_event(roi_stim_epochs, ROIs, figsize=figsize, title=title)

    def plot_mean_sems(self, figsize=(10,10), col_num=4, main_title='default'):
        # Get the trace mean and sem for each ROI first
        _, roi_mean_sems = self.opto_trace_mean_sems()
        new_window = [self.window[0],self.window[1]+self.stim_len]
        plotting.plot_mean_sem(roi_mean_sems, new_window, self.ROIs, figsize=figsize, col_num=col_num, main_title=main_title)
        
    def plot_heatmap(self, figsize=(4,5), main_title='default', cmap=None):
        if main_title == 'default':
            main_title = 'Mean Opto Activity Heatmap'
        _, roi_mean_sems = self.opto_trace_mean_sems()
        plotting.plot_opto_heatmap(roi_mean_sems, self.zscore, self.sampling_rate, figsize=figsize, main_title=main_title, cmap=cmap)
        
        
    def plot_shuff_dist(self, figsize=(10,10), col_num=4, main_title="default"):
        if self.method == 'test':
            print('Wilcoxon-test not shuffled distribution was performed')
            return
        else:
            pass
        
        if self.sig_results is None:
            sig_results = self.significance_testing(method='shuff')
        else:
            sig_results = self.sig_results
        plotting.plot_shuff_distribution(sig_results, self.ROIs, figsize=figsize, col_num=col_num, main_title=main_title)