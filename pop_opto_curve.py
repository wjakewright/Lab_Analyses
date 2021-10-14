# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sy
from scipy import stats
import pop_opto_analysis
import utilities as utils


class pop_opto_curve():
    ''' Class to generate a power curve for optogenetic stimulation across
        different stimulation power levels'''
        
    def __init__(self,imaging_data, behavioral_data, powers, method, sampling_rate=30, window=[-2,2],
                 stim_len=1):
        '''__init__ - Initialize pop_opto_curve Class.
        
            CREATOR
                William (Jake) Wright
            
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
        self.optos = None
        self.sig_results = None
        self.mean_diffs = None
        self.sem_diffs = None
    
    def analyze_opto(self):
        # Getting opto objects using pop_opto_analysis
        optos = []
        
        for imaging,behavior in zip(self.imaging_data,
                                    self.behavioral_data):
            
            opto = pop_opto_analysis(imaging,behavior,self.sampling_rate,
                                     self.window,self.stim_len,self.to_plot)
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
        sig_results = []
        for o in self.optos:
            sig_results.append(o.significance_testing(method=self.method))
        self.sig_results = sig_results
        
        return sig_results
            
    
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
                before, after = util.get_before_after_means(activity=data[col],
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
    
    def plot_individual_sessions(self):
        # Plot the individual activity of each ROI for each imaging session
        # Uses the plots from pop_opto_analysis
            
    
    def generate_curves(self):
        #
            