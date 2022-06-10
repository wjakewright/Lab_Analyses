"""Module to convert the lever trace into frames to allow for alignment
    with imaging data"""


import numpy as np
from Lab_Analyses.Behavior.read_bit_code import read_bit_code


def lever_to_frames(
    xsg_data,
    lever_active,
    lever_force_resample,
    lever_force_smooth,
    lever_velocity_envelope_smooth,
    dispatcher_data,
):
    """Function to convert and downsample the lever trace to imaging frames
        
        INPUT PARAMETERS
            xsg_data - object containing data from all the xsglog files. Output
                        from load_xsg_continuous function
            
            lever_active - binarized np.array of when the lever is active. Output
                            from parse_lever_movement_continuous
            
            lever_force_resample - np.array of the lever force resampled to 1kHz
                                    Output from parse_lever_movement_continuous
            
            lever_force_smooth  - np.array of the resampled lever force smoothed with
                                  a butterworth filter
                                  Output from parse_lever_movement_continuous
            
            lever_velocity_envelope_smooth - np.array of the lever velocity envelope
                                            calculated with hilbert transformation
                                            and then smoothed
                                            Output from parse_lever_movement_continous
            
            dispatcher_data - object containing the matlab output from dispatcher
            
        OUTPUT PARAMETERS
            lever_active_frames - np.array of lever active in terms of frames

            lever_force_resample_frames - np.array of lever_force_resample in terms of 
                                            frames
            
            lever_force_smooth_frames - np.array of lever_force_smooth in terms of frames

            lever_velocity_envelope_smooth - np.array of lever_velocity_envelope_smooth
                                            in terms of frames

    """

