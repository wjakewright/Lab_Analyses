from os.path import join as pjoin
import scipy.io as sio
import numpy as np

def load_behavior(fname,fname1=None,path=None):
    '''load_behavior - Function to load the behavior .mat file output
                       from NHanaly_LeverPressBehavior.m function that reads
                       behavior data output from ephus and dispatcher. See
                       corresponding matlab code for more details

        CREATOR
            William (Jake) Wright - 10/7/2021
            
        USAGE
            behavior_dict = load_behavior_mat_file(fname,fname1=None)

        INPUT PARAMETERs
            fname - string of the name of the file you wish to load.

            fname1 - string of the structure in the file you wish to load if it is different
                     than the file name. Optional, with default set to None.

            path - path where your file will be loaded from. Default path is set to
                   'C:\\Users\\Jake\\Desktop\\Processed_Data. Input path string if you
                   wish to change.

        OUTPUT PARAMETERs
            behavior_dict - Dictionary of all the data within the .mat file.
                            Organized in the same manner as theh strcture within
                            the .mat file, but set up with dictionary instead

        ADDITONAL DETAILS
            Waves structure in my .mat files only contain empty structures, so not coding it in
            for now.'''

    ## In case the name of the structure in the file doesn't match what you've
    ## saved the file as. Name of the structure depends on the name assigned
    ## in NHanay_LeverPressBehavior
    if fname1 == None:
        fname1 = fname
    else:
        fname1 = fname1

    ## set the path to load the .mat file
    if path == None:
        p = r'C:\Users\Jake\Desktop\Processed_data'
    else:
        p = path
    mat_fname = pjoin(p,fname)
    try:
        mat_file = sio.loadmat(mat_fname)
    except FileNotFoundError:
        print(fname + ' not found')
        return

    ## Extract data from the .mat file into a dictionary
    behavior_dict = {}
    behavior_dict['lever_active'] = mat_file[fname1]['lever_active'][0,0]
    behavior_dict['lever_force_resample'] = mat_file[fname1]['lever_force_resample'][0,0]
    behavior_dict['lever_force_smooth'] = mat_file[fname1]['lever_force_smooth'][0,0]
    behavior_dict['lever_velocity_envelope_smooth'] = mat_file[fname1]['lever_velocity_envelope_smooth'][0,0]
    behavior_dict['imaged_trials'] = mat_file[fname1]['Imaged_Trials'][0,0]
    behavior_frames = mat_file[fname1]['Behavior_Frames'][0,0]
    behavior_trials = [] ## list with each item being a dictionary for each behavioral trial
    
    for i in range(len(behavior_frames)):
        behavior_d = {} ## dictionary containing dictionaries for states, pokes, and waves

        behavior_states = {} ## dictionary containing all info in the states structure
        states = behavior_frames[i][0][0,0]['states'][0,0]
        behavior_states['state_0'] = states['state_0']
        behavior_states['check_next_trial_ready'] = states['check_next_trial_ready']
        behavior_states['bitcode'] = states['bitcode']
        behavior_states['cue'] = states['cue']
        behavior_states['end_cue_hold'] = states['end_cue_hold']
        behavior_states['reward'] = states['reward']
        behavior_states['end_cue_punish'] = states['end_cue_punish']
        behavior_states['punish'] = states['punish']
        behavior_states['opto_off'] = states['opto_off']
        behavior_states['iti'] = list(states['iti'][0])
        behavior_states['iti2'] = list(states['iti2'][0])
        behavior_states['iti3'] = list(states['iti3'][0])
        behavior_states['starting_state'] = states['starting_state']
        behavior_states['ending_state'] = states['ending_state']
        behavior_d['states'] = behavior_states

        behavior_pokes = {} ## dictionary containing all info in the pokes structure
        pokes = behavior_frames[i][0][0,0]['pokes'][0,0]
        behavior_pokes['C'] = pokes['C']
        behavior_pokes['L'] = pokes['L']
        behavior_pokes['R'] = pokes['R']
        p_starting_states = {}
        p_starting_states['C'] = pokes['starting_state'][0,0]['C']
        p_starting_states['L'] = pokes['starting_state'][0,0]['L']
        p_starting_states['R'] = pokes['starting_state'][0,0]['R']
        behavior_pokes['starting_state'] = p_starting_states
        p_ending_states = {}
        p_ending_states['C'] = pokes['ending_state'][0,0]['C']
        p_ending_states['L'] = pokes['ending_state'][0,0]['L']
        p_ending_states['R'] = pokes['ending_state'][0,0]['R']
        behavior_pokes['ending_state'] = p_ending_states
        behavior_d['pokes'] = behavior_pokes

        behavior_trials.append(behavior_d)

    behavior_dict['behavior_frames'] = behavior_trials
    behavior_dict['frame_times'] = mat_file[fname1]['Frame_Times'][0,0]

    return behavior_dict


def load_soma_imaging(fname,fname1=None,path=None):
    ''' load_soma_imaging - Function to load the .mat file output from SummarizeSomaData.m function, which
                       processes the fluorescence traces extrated via the CaImageViewer matlab GUI. See
                       corresponding matlab code for more details.

        CREATOR
            William (Jake) Wright 10/7/21

        USAGE
            data_dict = load_soma_imaging(fname,fname1=None,path=None)

        INPUT PARAMETERS
            fname - string of the file name you wish to load

            fname1 - string of the name of the structure you wish to load if it is different from
                     the file name. Optional and is Default to None

            path - string of the path where your file will be loaded from. Default path is set to
                   'C:\\Users\\Jake\\Desktop\\Processed_Data. Input path string if you
                   wish to change.

        OUTPUT PARAMETERS
            data_dict - Dictionary of all the data within the .mat file. Oraganized in the same manner
                        as the structure within the .mat file, but formated as a dictionary.

        ADDITIONAL DETAILS
            This code is currently hard coded to load only the SummarySomaData output. If you wish to
            load other imaging data .mat files code will need to be updated if there are different fields
            within the .mat file structure (e.g. fields related to dendrites and polyline). Position
            variables are also not coded in.'''

    
    ## In case the name of the structure in the file doesn't match what you've
    ## saved the file as. Name of the structure depends on the name assigned
    ## in NHanay_LeverPressBehavior
    if fname1 == None:
        fname1 = fname
    else:
        fname1 = fname1

    ## set the path to load the .mat file
    if path == None:
        p = r'C:\UsersJake\Desktop\Processed_data'
    else:
        p = path
    
    mat_fname = pjoin(p,fname)
    try:
        mat_file = sio.loadmat(mat_fname)
    except FileNotFoundError:
        print(fname + ' not found')
        return

    data_dict = {}
    data_dict['baseline_frames'] = mat_file[fname1]['BaselineFrames'][0,0]
    data_dict['Background_Intensity'] = mat_file[fname1]['Background_Intensity'][0,0]
    data_dict['Time'] = mat_file[fname1]['Time'][0,0]
    fluo_intensity = []
    for i in mat_file[fname1]['Fluorescence_Intensity'][0,0]:
        fluo_intensity.append(i[0])
    data_dict['Fluorescence_Intensity'] = fluo_intensity
    tot_intensity = []
    for i in mat_file[fname1]['Total_Intensity'][0,0]:
        tot_intensity.append(i[0][0][0])
    data_dict['Total_Intensity'] = tot_intensity
    pix_num = []
    for i in mat_file[fname1]['Pixel_Number'][0,0]:
        pix_num.append(i[0][0][0])
    data_dict['Pixel_Number'] = pix_num
    fluo_m = []
    for i in mat_file[fname1]['Fluorescence_Measurement'][0,0]:
        fluo_m.append(i[0][0])
    data_dict['Fluorescence_Measurement'] = fluo_m
    # dFoF from CaImageViewer not used. Use the dFoF below.
    #deltaF = []
    #for i in mat_file[fname1]['deltaF'][0,0][0]:
    #    deltaF.append(i[0])
    #data_dict['deltaF'] = deltaF
    #dF_over_F = []
    #for i in mat_file[fname1]['dF_over_F'][0,0][0]:
    #    dF_over_F.append(i[0])
    #data_dict['dF_over_F'] = dF_over_F
    data_dict['Filename'] = mat_file[fname1]['Filename'][0,0][0]
    data_dict['ZoomValue'] = mat_file[fname1]['ZoomValue'][0,0][0][0]
    data_dict['Is_Longitudinal'] = mat_file[fname1]['IsLongitudinal'][0,0][0][0]
    data_dict['ROIs'] = np.floor(mat_file[fname1]['SpineROIs'][0,0][0])
    #data_dict['ROI_text'] = np.floor(mat_file[fname1]['SpineROItext'][0,0][0])
    data_dict['ROI_num'] = mat_file[fname1]['NumberofSpines'][0,0][0][0]
    data_dict['threshold_multiplier'] = mat_file[fname1]['somathresholdmultiplier'][0,0][0][0]
    data_dict['smooth_window'] = mat_file[fname1]['somasmoothwindow'][0,0][0][0]
    data_dict['Cluster_Thresh'] = mat_file[fname1]['ClusterThresh'][0,0][0][0]
    data_dict['SpectralLengthConstant'] = mat_file[fname1]['SpectralLengthConstant'][0,0][0][0]
    data_dict['Imaging_Sensor'] = mat_file[fname1]['ImagingSensor'][0,0][0]
    data_dict['drifting_baseline'] = mat_file[fname1]['somadriftbaseline'][0,0]
    data_dict['dFoF'] = mat_file[fname1]['dFoF'][0,0]
    data_dict['processed_dFoF'] = mat_file[fname1]['Processed_dFoF'][0,0]
    data_dict['floored_events'] = mat_file[fname1]['floored'][0,0]
    data_dict['activity_map'] = mat_file[fname1]['ActivityMap'][0,0]
    data_dict['event_frequency'] = mat_file[fname1]['Frequency'][0,0]
    thresholds = {}
    thresholds['lower_threshold'] = mat_file[fname1]['SomaThresholds'][0,0]['LowerThreshold'][0,0][0]
    thresholds['upper_threshold'] = mat_file[fname1]['SomaThresholds'][0,0]['UpperThreshold'][0,0][0]
    data_dict['thresholds'] = thresholds
    data_dict['threshold_method'] = mat_file[fname1]['ThresholdMethod'][0,0]
    data_dict['mean_event_amp'] = mat_file[fname1]['MeanEventAmp'][0,0][0].reshape(-1,1) ## reshaping to match event_frequency shape
    
    
    return data_dict
    

def merge_imaging_behavior(imaging_dict, behavior_dict):
    ''' Function to merge imaging and behavioral data into single dictionary'''
    data_dict = {'imaging':imaging_dict, 'behavior':behavior_dict}
    return data_dict
  
