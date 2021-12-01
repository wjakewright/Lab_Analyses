import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
import pop_opto_curve as curve
import utilities as util
import opto_plotting as plotting
from tabulate import tabulate
import seaborn as sns; sns.set()
sns.set_style('ticks')

class group_opto_analysis():
    '''Class to analyze optogenetic stimulation experiments across different mice'''

    def __init__(self,opto_objs,new_window=None):
        '''__init__ - Initialize group_opto_analysis Class.
        
           CREATOR
                William (Jake) Wright  -  11/22/2021
            
            INPUT PARAMETERS
                opto_objs  -  List of objects output from pop_opto_curve analysis.
                
        '''
        # Store list of objects
        self.opto_objs = opto_objs
        ## Test is objects were analyzed with the same parameters
        obj_powers = [obj.powers for obj in opto_objs]
        obj_powers_t = [tuple(obj.powers) for obj in opto_objs]
        obj_methods = [obj.method for obj in opto_objs]
        obj_zscore = [obj.zscore for obj in opto_objs]
        obj_spines = [obj.spines for obj in opto_objs]
        obj_sampling_rate = [obj.sampling_rate for obj in opto_objs]
        testing_parameters = [obj_powers_t,obj_methods,obj_zscore,obj_spines,obj_sampling_rate]
        for test in testing_parameters:
            if len(set(test)) == 1:
                pass
            else:
                raise ValueError('Objects were not analyzed with the same parameters !!!')
        # Store parameters from the objects
        self.powers = obj_powers[0]
        self.method = obj_methods[0]
        self.zscore = obj_zscore[0]
        self.spines = obj_spines[0]
        self.sampling_rate = obj_sampling_rate[0]
        # Renaming ROIs to correspond with different mice
        ROI_list = [obj.optos[0].ROIs for obj in opto_objs]
        self.ROI_list = ROI_list ## May need for future use
        new_ROIs = []
        new_ROIs_list = [] # Each mouse ROIs in seperate list
        for i, ROIs in enumerate(ROI_list): ## Iterating through each mouse
            ROI_list = []
            for ROI in ROIs: ## Iterating through each ROI
                new_ROIs.append(f'Mouse {i} {ROI}')
                ROI_list.append(f'Mouse {i} {ROI}')
            new_ROIs_list.append(ROI_list)
        self.new_ROIs = new_ROIs
        self.new_ROIs_list = new_ROIs_list

        if new_window is None:
            self.new_window = self.opto_objs[0].optos[0].new_window
        else:
            self.new_window = new_window

        # Getting grouped data
        
        self.group_mean_diffs = None
        self.group_mean_sems = None
        self.sig_list = None
        self.power_diffs = None
        self.power_scatter = None
        self.power_sem = None
        self.percent_sig = None
        self.group_data()
    
    def group_data(self):
        '''Method to group the data across mice. Maintains different power groupings'''

        group_mean_diffs = np.hstack([obj.mean_diffs for obj in self.opto_objs])
        self.group_mean_diffs = group_mean_diffs
        group_mean_sems = np.hstack([obj.sem_diffs for obj in self.opto_objs])
        self.group_mean_sems = group_mean_sems

        self.sig_list = [obj.sig_df for obj in self.opto_objs]
        group_sig_results = []
        for result in zip(*self.sig_list):
            group_result = pd.concat(result,ignore_index=True)
            group_result.set_axis(self.new_ROIs,axis=0)
            group_sig_results.append(group_result)
        self.group_sig_results = group_sig_results

        roi_stim_dics = []
        for ROIs, obj in zip(self.new_ROIs_list,self.opto_objs):
            roi_stims = [opto.roi_stim_epochs for opto in obj.optos]
            new_roi_stims = []
            for roi_stim in roi_stims: #Loop to rename the keys
                new_roi_stim = dict(zip(ROIs, list(roi_stim.values())))
                new_roi_stims.append(new_roi_stim)
            roi_stim_dics.append(new_roi_stims)
        roi_stim_epochs = []
        for dicts in zip(*roi_stim_dics):
            group_dict = {}
            for d in dicts:
                group_dict.update(d)
            roi_stim_epochs.append(group_dict)
        self.roi_stim_epochs = roi_stim_epochs

        roi_mean_dics = []
        for ROIs, obj in zip(self.new_ROIs_list, self.opto_objs):
            roi_means = [opto.roi_mean_sems for opto in obj.optos]
            new_roi_means = []
            for roi_mean in roi_means:
                new_roi_mean = dict(zip(ROIs, list(roi_mean.values())))
                new_roi_means.append(new_roi_mean)
            roi_mean_dics.append(new_roi_means)
        roi_mean_sems = []
        for dicts in zip(*roi_mean_dics):
            group_dict = {}
            for d in dicts:
                group_dict.update(d)
            roi_mean_sems.append(group_dict)
        self.roi_mean_sems = roi_mean_sems


        sig_dics = []
        for ROIs, obj in zip(self.new_ROIs_list, self.opto_objs):
            sigs = obj.significance
            new_sigs = []
            for sig in sigs:
                new_sig = dict(zip(ROIs, list(sig.values())))
                new_sigs.append(new_sig)
            sig_dics.append(new_sigs)
        significance_dicts = []
        for dicts in zip(*sig_dics):
            group_dict = {}
            for d in dicts:
                group_dict.update(d)
            significance_dicts.append(group_dict)
        self.significance_dicts = significance_dicts


    def get_grouped_power_curve(self):
        '''Function to generate the grouped power curve and plot it'''

        # For Figure 1
        power_diffs = []
        power_scatter = pd.DataFrame()
        power_sem = []
        for diff, power in zip(self.group_mean_diffs,self.powers):
            power_diffs.append(np.mean(diff))
            power_scatter[power] = np.array(diff)
            power_sem.append(stats.sem(diff))
        self.power_diffs = power_diffs
        self.power_scatter = power_scatter
        self.power_sem = power_sem
        
        # For Figure 2
        percent_sig = []
        for result in self.group_sig_results:
            percent = (result['sig'].sum()/len(result.index)) * 100 
            percent_sig.append(percent)
        self.percent_sig = percent_sig
        
        plotting.plot_power_curve(self.powers,power_diffs,power_sem,power_scatter,percent_sig,self.zscore)

    def disp_group_results(self,method):
        '''Function to display results in an easily readable manner'''
        if self.power_diffs is None:
            self.get_grouped_power_curve()
        
        all_diffs = {}
        for diff, power in zip(self.group_mean_diffs,self.powers):
            all_diffs[str(power) + ' mW'] = diff
        
        # Generate table summarizing results
        summary_df = pd.DataFrame()
        for diff, sem, p_sig, power in zip(self.power_diffs,self.power_sem,self.percent_sig,self.powers):
            summary_df[str(power) + ' mW'] = [diff, sem, p_sig,len(self.new_ROIs)]
        summary_df.set_axis(['mean_diff', 'sem_diff', 'percent_sig', 'n'],axis=0,inplace=True)
        summary_table = tabulate(summary_df,headers='keys',tablefmt='fancy_grid')

        # Perform one-way anova across different powers
        f_stat, anova_p, results_table = util.ANOVA_1way_bonferroni(all_diffs,method)

        # Display results
        print('One-Way ANOVA results')
        print(f'F statistic: ', f_stat, '\pvalue: ', anova_p)
        print('\n')
        print(method + ' Posttest Results')
        print(results_table)
        print('\n')
        print('Summary Statistics')
        print(summary_table)

    def visulalize_specific_condition(self,sess):
        '''Function to plot the grouped data for a specific stimulation condition.
            Musut input the index of which session you wish to visuzliaze (e.g. 0 for first session)'''
        
        power = self.powers[sess]
        sess_name = str(power) + ' mW'
        for obj in self.opto_objs:
            obj.optos[sess].plot_sess_activity(title=sess_name + ' Session Activity')
        plotting.plot_each_event(self.roi_stim_epochs[sess],self.new_ROIs,figsize=(10,50),
                                 title = sess_name + ' Session Timelocked Activity')
        plotting.plot_mean_sem(self.roi_mean_sems[sess], self.new_window, self.new_ROIs,
                                figsize=(10,20),main_title = sess_name + ' Mean Opto Activity')
        plotting.plot_opto_heatmap(self.roi_mean_sems[sess],self.zscore, self.sampling_rate,
                                    figsize=(4,5),main_title = sess_name +' Mean Opto Heatmap',
                                    cmap=None)
        if self.method == 'shuff':
            plotting.plot_shuff_distribution(self.significance_dicts[sess],self.new_ROIs,
                                             figsize=(10,20),col_num=4,main_title = sess_name + ' Shuff Distributions')
        else:
            pass