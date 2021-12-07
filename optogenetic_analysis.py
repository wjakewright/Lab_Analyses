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
                imaging_data - object of the calcium imaging data output from 
                                the load_mat functions.
                behavior_data - object of the behavior data output from the 
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

        # Select the trials that were imaged
        i_trials = behavior_data.Imaged_Trials
        i_trials = i_trials == 1
        self.i_trials = i_trials
        behavior = list(compress(behavior_data.Behavior_Frames,i_trials))
        self.behavior = behavior
        itis = []
        for i in behavior:
            itis.append(i.states.iti2)
        
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
        self.results_dict = None
        self.results_df = None
        self.all_befores = None
        self.all_afters = None
        self.roi_stim_epochs = None
        self.roi_stim_epochs_list = None
        self.roi_mean_sems = None
        self.roi_mean_sems_list = None

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
        if data is None:
            data = self.dFoF
        else:
            data = data
        
        new_timestamps = [i[0] for i in self.itis]
        if self.vis_window is None:
            new_window = [self.window[0],self.window[1]+self.stim_len]
        else:
            new_window = self.vis_window
        self.new_window = new_window

        roi_stim_epochs, roi_mean_sems = util.get_trace_mean_sem(activity=data,
                                                                 timestamps=new_timestamps,
                                                                 window=new_window,
                                                                 sampling_rate=self.sampling_rate)
        
        self.roi_stim_epochs = roi_stim_epochs
        self.roi_stim_epochs_list = list(roi_stim_epochs.values())
        self.roi_mean_sems = roi_mean_sems
        self.roi_mean_sems_list = list(roi_mean_sems.values())

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
        
        # Perform significance testing
        # Wilcoxon signed-rank method
        if method == 'test':
            pValues = []
            rankValues = []
            diffs = []
            for before, after in zip(before_m,after_m):
                b = before
                a = after

                ## Perform Wilcoxon signed rank test: Significance set at 0.01
                rank, pVal = stats.wilcoxon(a,b)
                pValues.append(pVal)
                rankValues.append(rank)
                diffs.append(np.mean(np.array(a)-np.array(b)))
            ## Assess significance
            sig = (np.array(pValues) < 0.01) * 1

            results_dict = {}
            for p, r, d, s, ROI in zip(pValues,rankValues,diffs,sig,self.ROIs):
                results_dict[ROI] = {'pvalue': p,
                                     'rank': r,
                                     'diff': d,
                                     'sig': s}
        
        # Shuffle distribution method
        elif method == 'shuff':
            data = self.dFoF.copy()
            real_diffs = []
            shuff_diffs = []
            bounds = []
            sigs = []
            smallest = self.sampling_rate # smallest shift of data is 1s
            biggest = 300 * self.sampling_rate # biggest shift of data is 5min

            # Assess each ROI individually
            for col in data.columns:
                d = data[col]

                # Get the real difference
                before, after = self.opto_before_after_means(data=d,single=True)
                r_diff = np.mean(np.array(after)-np.array(before))
                # Perform 1000 shuffles
                s_diffs = []
                for i in range(1000):
                    n = random.randint(smallest,biggest)
                    s_d = np.copy(d)
                    shuff_d = np.roll(s_d,n)

                    shuff_before, shuff_after = self.opto_before_after_means(data=shuff_d,single=True)
                    s_diffs.append(np.mean(np.array(shuff_after)-np.array(shuff_before)))
                # Assess significance
                ## Significance set at 1st percentile
                upper = np.percentile(s_diffs,99)
                lower = np.percentile(s_diffs,1)

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
            results_dict = {}
            for col, r, s, b, sig in zip(data.columns, real_diffs, shuff_diffs, bounds, sigs):
                results_dict[col] = {'real_diff': r, 'shuff_diffs': s, 'bounds':b, 'sig': sig}
        
        else:
            return ('Not a valid testing method indicated !!!')
        self.results_dict = results_dict
        results_df = pd.DataFrame.from_dict(results_dict,orient='index')
        if 'shuff_diff' in results_df.columns:
            self.shuff_diffs = list(results_df['shuff_diff'])
            results_df = results_df.drop(columns=['shuff_diff'])
        self.results_df = results_df

        return results_dict, results_df
    
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
        plotting.plot_mean_sem(roi_mean_sems, self.new_window, self.ROIs, figsize=figsize, col_num=col_num, main_title=main_title)
        
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
        
        if self.results_dict is None:
            sig_results = self.significance_testing(method='shuff')
        else:
            sig_results = self.results_dict
        plotting.plot_shuff_distribution(sig_results, self.ROIs, figsize=figsize, col_num=col_num, main_title=main_title)

class opto_curve():
    '''Class to generate a power curve for optogenetic stimulation across
        different stimulation power levels. Includes visualization and statistial
        testing between powers. Dependent on opto_analysis Class'''
    
    def __init__(self, imaging_data, behavior_data, powers, method, sampling_rate=30, 
                 window=[-2,2], vis_window=None, stim_len=1, zscore=False, spines=False):
        '''__init__ - Initialize pop_opto_curve Class.

            
            INPUT PARAMETERS
                imaging_data - list of the objects containing the imaging data
                               output from the load_mat functions. Each element in the 
                               list will correspond with each power tested
                
                behavioral_data - list of the objects containing the behavioral data
                                  output from the load_mat functions. Each element in the list
                                  will correspond with each power tested. Must match the 
                                  imaging data in its length
                
                powers - a list of the powers used 
                
                method - string to determine how significant activation is to be assessed
                         currently coded for 'test' and 'shuff'
                
                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.

                window - list specifying the time before and after the opto stim
                         you want to visualize and assess (e.g. [-2,2] for two sec
                         before and after). Default is set to [-2,2].
                
                vis_window - same as window, but for visualizing the data only
                            Default is set to none, in which case vis_window will
                            be determined by window and stim_len.

                stim_len - scaler specifying how long opto stim was delivered for.
                           Default is set to 1 sec'''

        # Storing initial inputs
        self.imaging_data = imaging_data
        self.behavioral_data = behavior_data
        self.powers = powers
        self.method = method
        self.sampling_rate = sampling_rate
        self.window = window
        self.vis_window = vis_window
        self.stim_len = stim_len
        self.zscore = zscore
        self.spines = spines
        self.results_dicts = None
        self.results_dfs = None
        self.shuff_diffs = None
        self.mean_diffs = None
        self.power_diffs = None
        self.power_scatter = None
        self.power_sem = None
        self.percent_sig = None

        self.analyze_opto()

        self.ROIs = self.optos[0].ROIs

    def analyze_opto(self):
        # Getting opto objects using opto_analysis
        optos = []
        for imaging, behavior in zip(self.imaging_data, self.behavioral_data):
            opto = opto_analysis(imaging,behavior,self.sampling_rate,
                                 self.window,self.vis_window,self.stim_len,
                                 self.zscore,self.spines)
            opto.opto_before_after_means()
            opto.opto_trace_mean_sems()
            optos.append(opto)
        results_dicts = []
        results_dfs = []
        shuff_diffs = []
        mean_diffs = []
        for o in optos:
            o.significance_testing(method=self.method)
            results_dicts.append(o.results_dict)
            results_dfs.append(o.results_df)
            shuff_diffs.append(o.shuff_diffs)
            mean_diffs.append(np.array(o.all_afters)-np.array(o.all_befores))
        self.optos = optos
        self.results_dicts = results_dicts
        self.results_dfs = results_dfs
        self.shuff_diffs = shuff_diffs
        self.mean_diffs = mean_diffs
    
    def visualize_individual_sessions(self, sess):
        '''Plot the analysis results for each individual session. Must input the 
            index of which session you wish to visualize (e.g. 0 for first session)'''
        
        session = self.optos[sess]
        sess_name = str(self.powers[sess]) + ' mW'
        plotting.plot_session_activity(session.dFoF, session.itis, zscore=self.zscore, 
                                       figsize=(7,8), title=sess_name + ' Session Activity')
        plotting.plot_each_event(session.roi_stim_epochs,session.ROIs,figsize=(10,20),
                                 title=sess_name + ' Session Timelocked Activity')
        plotting.plot_mean_sem(session.roi_mean_sems,session.new_window, session.ROIs,
                                figsize=(10,10),col_num=4,main_title = sess_name + ' Mean Opto Activity')
        plotting.plot_opto_heatmap(session.roi_mean_sems,session.zscore, session.sampling_rate,
                                    figsize=(4,5),main_title = sess_name +' Mean Opto Heatmap')
        if self.method == 'shuff':
            plotting.plot_shuff_distribution(session.results_dict, session.ROIs, 
                                             figsize=(10,10),col_num=4,main_title = sess_name + ' Shuff Distributions')
        else:
            pass
        display(session.results_dict)

    def get_power_curves(self):
        '''Method to generate the power curve and plot it'''
        power_diffs, power_scatter, power_sem, percent_sig = generate_power_curve(self.mean_diffs,
                                                                                  self.powers,
                                                                                  self.results_dfs,
                                                                                  self.zscore)
        self.power_diffs = power_diffs
        self.power_scatter = power_scatter
        self.power_sem = power_sem
        self.percent_sig = percent_sig

    def display_curve_results(self,method):
        '''Method to display power curve results'''
        display_results(self.mean_diffs,self.power_diffs,self.power_sem,
                        self.percent_sig,self.powers,len(self.ROIs),method)


class group_opto_analysis():
    '''Class to analyze optogenetic stimulation experiments across different mice'''

    def __init__(self, data, powers, method, sampling_rate=30, 
                 window=[-2,2], vis_window=None, stim_len=1, zscore=False, spines=False):
        ''' __init__ - Initialize pop_opto_curve Class.

            
            INPUT PARAMETERS
                data - list of lists containing the paired imaging and behavioral data
                        for the different datasets to be analyzed. E.g [[mouse1],[mouse2]]. 
                        Each dataset is a list of dictionaries containing the paired imaging
                        and behavioral data for each session for that mouse. E.g. mouse1 =
                        [dicitionary1,dictionary2], dictionary1 = {imaging:data, behavior:data}
                
                powers - a list of the powers used 
                
                method - string to determine how significant activation is to be assessed
                         currently coded for 'test' and 'shuff'
                
                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.

                window - list specifying the time before and after the opto stim
                         you want to visualize and assess (e.g. [-2,2] for two sec
                         before and after). Default is set to [-2,2].
                
                vis_window - same as window, but for visualizing the data only
                            Default is set to none, in which case vis_window will
                            be determined by window and stim_len.

                stim_len - scaler specifying how long opto stim was delivered for.
                           Default is set to 1 sec'''

        # Set up the data to be analyzed
        self.data = data
        self.powers = powers
        self.method = method
        self.sampling_rate = sampling_rate
        self.window = window
        self.vis_window = vis_window
        self.stim_len = stim_len
        self.zscore = zscore
        self.spines = spines
        # Analyze each session seperately
        self.curves = None
        self.generate_curves
        # Get the grouped data
        self.group_mean_diffs = None
        self.group_men_sems = None
        self.sig_list = None
        self.power_diffs = None
        self.power_scatter = None
        self.percent_sig = None
        self.group_data()

    def generate_curves(self):
        '''Method to analyze each dataset using the opto_curve Class'''
        curves = []
        for dataset in self.data:
            imaging = []
            behavior = []
            for session in dataset:
                imaging.append(dataset['imaging'])
                behavior.append(dataset['behavior'])
            curve = opto_curve(imaging_data=imaging,behavior_data=behavior,
                                powers=self.powers,method=self.method,sampling_rate=self.sampling_rate,
                                window=self.window,vis_window=self.vis_window,stim_len=self.stim_len,
                                zscore=self.zscore,spines=self.spines)
            curves.append(curve)
        self.curves = curves

    def group_data(self):
        '''Method to group data and results across datasets'''





# Utility functions
def generate_power_curve(diffs, powers, results_df,zscore):
    '''Utility function to generate power curve. 
        Utilized by opto_curve and grouped_opto_analysis'''
    
    # For Figure 1
    power_diffs = []
    power_scatter = pd.DataFrame()
    power_sem = []
    for diff, power in zip(diffs, powers):
        power_diffs.append(np.mean(diff))
        power_scatter[power] = np.array(diff)
        power_sem.append(stats.sem(diff))
    
    # For Figure 2
    percent_sig = []
    for result in results_df:
        percent = (result['sig'].sum()/len(result.index)) * 100
        percent_sig.append(percent)
    
    plotting.plot_power_curve(powers,power_diffs,power_sem,power_scatter,percent_sig,zscore)

    return power_diffs, power_scatter, power_sem, percent_sig

def display_results(mean_diffs,power_diffs,power_sem,percent_sig,powers,n,method):
    '''Utility function to display results in an easily readable manner'''
    all_diffs = {}
    for diff, power in zip(mean_diffs,powers):
        all_diffs[str(power), + ' mW'] = diff
    
    # Summary table
    summary_df = pd.DataFrame()
    for diff, sem, p_sig, power in zip(power_diffs,power_sem,percent_sig,powers):
        summary_df[str(power) + ' mW'] = [diff,sem,p_sig,n]
    summary_df.set_axis(['mean_diff', 'sem_diff', 'percent_sig', 'n'],axis=0,inplace=True)
    summary_table = tabulate(summary_df,headers='keys',tablefmt='fancygrid')

    # Perform one-way anova across different powers
    f_stat, anova_p, results_table = util.ANOVA_1way_bonferroni(all_diffs,method)

    # Display results
    print('One-Way ANOVA results')
    print(f'F statistic: ', f_stat, '\p-value: ',anova_p)
    print('\n')
    print(method + ' Posttest Results')
    print(results_table)
    print('\n')
    print('Summary Statistics')
    print(summary_table)