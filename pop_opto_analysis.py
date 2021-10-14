import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
from itertools import compress
import random
import utilities as util
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('ticks')


class population_opto_analysis():
    '''Class to determine if neurons are being significantly activated
        by optogenetic stimulation.'''

    def __init__(self, imaging_data, behavior_data, sampling_rate=30,
                 window = [-2,2], stim_len=1, to_plot=False):
        ''' __init__- Initilize population_opto_analysis Class.

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

                to_plot - boolean True or False of weather or not you wish to
                          plot all figures generated.
            


                '''

        # Storing the initial input data
        self.imaging = imaging_data
        self.behavior = behavior_data
        self.sampling_rate = sampling_rate
        self.window = window
        self.to_plot = to_plot
        self.before_t = window[0] # before window in time(s)
        self.before_f = window[0]*sampling_rate # before window in frames
        self.after_t = window[1]
        self.after_f = window[1]*sampling_rate
        self.stim_len = stim_len
        self.stim_len_f = stim_len*sampling_rate
        
        # Pulling data from the inputs that will be used
        ROIs = []
        for i in list(imaging_data['ROIs'][:-1]):
            ROIs.append('ROI ' + str(i))
        dFoF = pd.DataFrame(data=imaging_data['processed_dFoF'].T,columns=ROIs)

        self.ROIs = ROIs
        self.dFoF = dFoF

        # Select on the trials that were imaged
        i_trials = behavior_data['imaged_trials']
        i_trials = i_trials == 1
        behavior = list(compress(behavior_data['behavior_frames'],i_trials))
        itis = []
        for i in behavior:
            itis.append(i['states']['iti2'])

        # Making sure the iti intervals are consistent and within imaging period
        for i in range(len(itis)):
            if itis[i][1] > len(dFoF):
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
        self.method = None
        self.sig_results = None


    def opto_before_after_means(self, data=None,single=None):
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

        return all_befores, all_afters

    def opto_trace_mean_sems(self,data=None):
        if data is None:
            data = self.dFoF
        else:
            data = data
  
        new_timestamps = []
        for i in self.itis:
            new_timestamps.append(i[0])
        new_window = [self.window[0],self.window[1]+self.stim_len]
        
        roi_stim_epochs,roi_mean_sems = util.get_trace_mean_sem(activity=data,
                                                                timestamps=new_timestamps,
                                                                window=new_window,
                                                                sampling_rate=self.sampling_rate)

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

        before_m, after_m = self.opto_before_after_means()

        if method == 'test':
            pValues = []
            rankValues = []
            diffs = []
            for before, after in zip(before_m,after_m):
                b = before
                a = after

                # Perform Wilcoxon signed-rank test; significance set at 0.01
                rank, pVal = stats.wilcoxon(b,a)
                pValues.append(pVal)
                rankValues.append(rank)
                diffs.append(np.mean(np.array(a)-np.array(b)))
                
            pValue_dict = dict(zip(self.dFoF.columns,pValues))
            rank_dict = dict(zip(self.dFoF.columns,rankValues))
            diff_dict = dict(zip(self.dFoF.columns,diffs))

            
            # Assess significance and differences
            sig = []
            for key,value in pValue_dict.items():
                if value < 0.01:
                    sig.append(1)
                else:
                    sig.append(0)
            sig_dict = dict(zip(self.dFoF.columns,sig))
            
            #Put final results in dictionary
            sig_results = {}
            for key,value in pValue_dict.items():
                sig_results[key] = {'pvalue':value,
                                   'rank':rank_dict[key],
                                   'diff':diff_dict[key],
                                   'sig':sig_dict[key]}
            self.sig_results = sig_results
            
        elif method == 'shuff':
            data = self.dFoF.copy()
            real_diffs = [] 
            shuff_diffs = [] 
            bounds = [] 
            sigs = []
            smallest = self.sampling_rate # smallest shift if 1 second
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
                upper = np.percentile(s_diffs, 95)
                lower = np.percentile(s_diffs, 5)
                
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
                                                      shuff_diffs,bounds, sigs):
                sig_results[col] = {'real_diff':real,'shuff_diff':shuff,
                                    'bounds':b, 'sig':sig}
            self.sig_results = sig_results
        
        return sig_results
    
    
    def plot_session_activity(self, title='Session Activity'):
        plt.figure(figsize=(7,8))
        for i, col in enumerate(self.dFoF.columns):
            plt.plot(self.dFoF[col] + i*5, label=col, linewidth=0.5)
        
        for iti in self.itis:
            plt.axvspan(iti[0],iti[1], alpha=0.3, color='red')
        
        plt.tick_params(axis='both', which='both', direction='in')
        plt.xlabel('Imagine Frames')
        plt.ylabel(r'$\Delta$F/F')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.1,1.05))
        plt.show()


    def plot_mean_sem(self):
        
        
    def plot_shuff_results(self):
        print('unfinished')
        