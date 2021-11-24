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

    def __init__(self,opto_objs):
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
        for i, ROIs in enumerate(ROI_list): ## Iterating through each mouse
            for ROI in ROIs: ## Iterating through each ROI
                new_ROIs.append(f'Mouse {i} {ROI}')
        self.new_ROIs = new_ROIs

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