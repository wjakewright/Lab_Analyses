# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
import pop_opto_analysis
import utilities as utils
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from IPython.display import display
sns.set_style('ticks')


class pop_opto_curve():
    ''' Class to generate a power curve for optogenetic stimulation across
        different stimulation power levels'''
        
    def __init__(self,imaging_data, behavioral_data, powers, method, sampling_rate=30, window=[-2,2],
                 stim_len=1,zscore=False):
        '''__init__ - Initialize pop_opto_curve Class.
        
            CREATOR
                William (Jake) Wright  -  10/14/2021
            
            INPUT PARAMETERS
                imaging_data - list of the dictionaries containing the imaging data
                               output from the load_mat functions. Each element in the 
                               list will correspond with each power tested
                
                behavioral_data - list of the dictionaries containing the behavioral data
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

                stim_len - scaler specifying how long opto stim was delivered for.
                           Default is set to 1 sec

            
            
                '''
        # Storing initial inputs
        self.imaging_data = imaging_data
        self.behavioral_data = behavioral_data
        self.powers = powers
        self.method = method
        self.sampling_rate = sampling_rate
        self.window = window
        self.stim_len = stim_len
        self.zscore = zscore
        self.optos = None
        self.significance = None
        self.mean_diffs = None
        self.sem_diffs = None
    
    def analyze_opto(self):
        # Getting opto objects using pop_opto_analysis
        optos = []
        
        for imaging,behavior in zip(self.imaging_data,
                                    self.behavioral_data):
            
            opto = pop_opto_analysis.population_opto_analysis(imaging,behavior,self.sampling_rate,
                                                              self.window,self.stim_len,self.zscore)
            optos.append(opto)
        self.optos = optos
        return optos
    
    def get_sig_results(self):
        ## Getting the significance results for each imaging session
        
        # get all the opto objects for each imaging session
        # sets up all the data for each imagine session
        if self.optos is None:
            self.analyze_opto()
        else:
            pass
        results = []
        for o in self.optos:
            results.append(o.significance_testing(method=self.method))
        self.significance = results
        
        return results
            
    
    def get_mean_sems(self):
        ## Getting the mean and sems
        # get all the opto objects for each imaging session
        # will get all the data extracted properly
        if self.optos is None:
            self.analyze_opto()
        else:
            pass
        
        mean_diffs = []
        sem_diffs = []
        # For each imaging session
        for o, power in zip(self.optos,self.powers):
            # Setting up the necessary imaging and behavioral data
            data = o.dFoF
            itis = o.itis
            diffs = []
            sems = []
            # for each roi
            for col in data.columns:
                before, after = utils.get_before_after_means(activity=data[col],
                                                             timestamps=itis,
                                                             window=self.window,
                                                             sampling_rate=self.sampling_rate,
                                                             offset=False,single=True)
                ds = np.array(after) - np.array(before)
                diffs.append(np.mean(ds))
                sems.append(stats.sem(ds))
            mean_diffs.append(diffs)
            sem_diffs.append(sems)
        
        self.mean_diffs = mean_diffs
        self.sem_diffs = sem_diffs
        
        return mean_diffs, sem_diffs
    
    def visualize_individual_sessions(self, sess):
        '''Plot the analysis results for each individual session. Must input the 
            index of which session you wish to visualize (e.g. 0 for first session)'''
            
        # Plot the individual activity of each ROI for each imaging session
        # Uses the plots from pop_opto_analysis
        if self.optos is None:
            self.analyze_opto()
            
        session = self.optos[sess]
        sess_name = str(self.powers[sess]) + ' mW'
        session.plot_session_activity(title = sess_name + ' Session Activity')
        session.plot_mean_sem(main_title = sess_name + ' Mean Opto Activity')
        if self.method == 'shuff':
            session.plot_shuff_dist(main_title = sess_name + ' Shuff Distributions')
        else:
            pass
        display(session.disp_results())
        
    
    def get_power_curves(self):
        ''' Function to generate the power curve and plot it'''
        if self.mean_diffs is None:
            self.get_mean_sems()
        ## For Figure 1
        ## Plotting the mean change in activity 
        power_diffs = []
        power_scatter = pd.DataFrame()
        power_sem = []
        for diff,power in zip(self.mean_diffs,self.powers):
            power_diffs.append(np.mean(diff))
            power_scatter[power] = np.array(diff)
            power_sem.append(stats.sem(diff))
        
        ## For Figure 2
        ## Get percentage of neurons that display significant change
        if self.significance is None:
            self.get_sig_results()
        else:
            pass
        signi = self.significance
        
        percent_sig = []
        for result in signi:
            s = []
            for key,value in result.items():
                s.append(value['sig'])
            percent = (sum(s)/len(s))*100
            percent_sig.append(percent)
        ## Plot figures
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(1,2,1)
        p = list(range(len(self.powers)))
        powers = [str(x) for x in self.powers]
        ax1.errorbar(p,power_diffs,yerr=power_sem,color='red',
                     marker='o',markerfacecolor='red',ecolor='red')
        sns.swarmplot(data=power_scatter,color='red',size=4,alpha=0.2)
        ax1.axhline(y=0,color='black',linestyle='--',linewidth=1)
        ax1.set_title('Mean Change in Activity',fontsize = 12)
        ax1.set_ylabel('$\Delta$F/F')
        ax1.set_xticklabels(labels=powers)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(p,percent_sig,color='red',marker='o',markerfacecolor='red')
        ax2.set_title('Percent Significant',fontsize=12)
        ax2.set_ylabel('Percentage of Neurons')
        ax2.set_xticks(p)
        ax2.set_xticklabels(labels=powers)
        plt.ylim(bottom=0)
        
        fig.add_subplot(111,frame_on=False)
        plt.tick_params(labelcolor='none',bottom=False,left=False)
        plt.xlabel('Power (mW)',labelpad=15)
        fig.tight_layout()
        
    def disp_results(self):
        print('incomplete code')
