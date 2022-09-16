"""Module to perform spine co-activity analyses"""

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_activity_timestamps,
    get_coactivity_rate,
    get_dend_spine_traces_and_onsets,
    nearby_spine_conjunctive_events,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import quantify_movment_quality
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    spine_volume_norm_constant,
)
from scipy import stats


def local_spine_coactivity_analysis(
    data,
    movement_epoch=None,
    cluster_dist=10,
    sampling_rate=60,
    zscore=False,
    volume_norm=False,
):
    """Function to analyze local spine coactivity
    
        INPUT PARAMETERS
            data - spine_data object (e.g., Dual_Channel_Spine_Data)
            
            movement_epoch - str specifying if you want to analyze only during specific
                            types of movements. Accepts - 'movement', 'rewarded', 'unrewarded',
                            'learned', and 'nonmovement'. Default is None, analyzing the entire
                            imaging session
            
            cluster_dist - int or float specifying the distance from target spine that will be
                            considered as nearby spines
            
            sampling_rate - int or float specicfying what the imaging rate is
            
            zscore - boolean of whether to zscore dFoF traces for analysis
            
            volume_nomr - boolean of whether or not to normalize activity by spine volume
            
        OUTPUT PARMATERS
            distance_coactivity_rate - np.array of the normalized coactivity for each spine
                                        (columns) over the binned distances (row)

            distance_bins - np.array of the distances data were binned over
            
            local_coactivity_matrix - 2d np.array of the binarized local coactivity 
                                        (columns=spines, rows=time)
            
            local_coactivity_rate - np.array of the event rate of local coactivity events (where
                                    at least one nearby spine is coactive) for each spine
                                    
            spine_fraction_coactive - np.array of the fraction of spine activity events that 
                                      are also coacctive with at least one nearby spine
            
            coactive_spine_num - np.array of the average number of coactive spines across 
                                local coactivity events for each spine
                                
            coactive_spine_volumes - np.array of the average volume of spines coactive during
                                    local coactivity events for each spine
            
            spine_coactive_amplitude - np.array of the average peak amplitude of activity
                                      during local coactive events for each spine
                                      
            nearby_coactive_amplitude - np.array of the peak average summed activity of nearby 
                                        coactive spines during local coactivity events for each spine
            
            spine_coactive_calcium - np.array of the average peak calcium amplitude during
                                    local coactivity events for each spine
            
            nearby_coactive_calcium - np.array of the peak average summed calcium activity of nearby
                                     spines during local coactivity events for each spine
                                     
            spine_coactive_std - np.array of the std around the peak activity amplitude during local
                                coactive events for each spine
            
            nearby_coactive_std - np.array of the std around the summed peak activity amplitude of
                                  nearby coactive spines during local coactive events for each spine
                                  
            spine_coactive_calcium_std - np.array of the std around the peak calcium amplitude
                                         during local coactive events for each spine
                                         
            nearby_coactiive_calcium_std - np.array of the std around the summe dpeak calcium amplitude
                                           of nearby coactive spines during local coactive events
            
            spine_coactive_calcium_auc - np.array of the auc of the average calcium trace during
                                         local coative events for each spine
                                         
            nearby_coactive_calcium_auc - np.array of the auc of the averaged summed calcium trace
                                          of nearby coactive spines during local coactive events
                                          
            spine_coactive_traces - list of 2d np.arrays of activity traces for each local coactivity
                                    event for each spine (columns = events)
            
            nearby_coactive_traces - list of 2d np.arrays of summed activity traces for each local
                                    coactivity event for each spine (columns=events)
                                    
            spine_coactive_calcium_traces - list of 2d np.arrays of calcium traces for each local coactivity
                                            event for each spine (coluumns=events)
            
            nearby_coactive_calcium_traces - list of 2d np.arrays of the summed calcium traces for each
                                            local coactivity event for each spine (columns=events)
    """


def local_coactivity_rate_analysis(
    spine_activity,
    spine_positions,
    flags,
    spine_grouping,
    bin_size=5,
    sampling_rate=60,
):
    """Function to calculate pairwise spine coactivity rate between 
        all spines along the same dendrite. Spine rates are then binned
        based on their distance from the target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces
                            Each column represents each spine
            
            spine_positions - list of the corresponding spine positions along the dendrite
                              for each spine

            flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines on
                             the same dendrite
            
            bin_size - int or float specifying the distance to bin over

        OUTPUT PARAMETERS
            coactivity_matrix - np.array of the normalized coactivity for each spine (columns)
                                over the binned distances (rows)
            
            position_bins - np.array of the distances data were binned over
            
    """
    # Set up the position bins
    MAX_DIST = 30
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # First sort out spine_groping to make sure you can iterate
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    coactivity_matrix = np.zeros((bin_num, spine_activity.shape[1]))

    # find indexes of eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Now iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = spine_activity[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Go through each spine
        for i in range(s_activity.shape[1]):
            current_coactivity = []
            curr_spine = s_activity[:, i]
            # Get coactivity with each other spine
            for j in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if j == i:
                    continue
                # Don't compare eliminated spines
                if curr_el_spines[i] == True:
                    current_coactivity.append(np.nan)
                    continue
                if curr_el_spines[j] == True:
                    current_coactivity.append(np.nan)
                    continue
                test_spine = s_activity[:, j]
                co_rate = calculate_coactivity(curr_spine, test_spine, sampling_rate)
                current_coactivity.append(co_rate)

            # Order by positions
            curr_pos = positions[i]
            pos = [x for idx, x in enumerate(positions) if idx != i]
            # normalize  distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            # Make all distances positive
            relative_pos = np.absolute(relative_pos)
            # Sort coactivity and position based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, current_coactivity))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, current_coactivity))]
            )
            # Bin the data
            binned_coactivity = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            coactivity_matrix[:, spines[i]] = binned_coactivity

    return coactivity_matrix, position_bins


def calculate_coactivity(spine_1, spine_2, sampling_rate):
    """Helper function to calculate spine coactivity rate"""
    duration = len(spine_1) / sampling_rate
    coactivity = spine_1 * spine_2
    events = np.nonzero(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    event_freq = event_num / duration

    # Normalize frequency based on overall activity levels
    spine_1_freq = len(np.nonzero(np.diff(spine_1) == 1)[0]) / duration
    spine_2_freq = len(np.nonzero(np.diff(spine_2) == 1)[0]) / duration
    geo_mean = stats.gmean([spine_1_freq, spine_2_freq])

    coactivity_rate = event_freq / geo_mean

    return coactivity_rate


def bin_by_position(data, positions, bins):
    """Helper function to bin the data by position"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.where((positions > bins[i]) & (positions <= bins[i + 1]))
            binned_data.append(np.nanmean(data[idxs]))
        # else:
        #    idxs = np.where(positions > bins[i])
        #    binned_data.append(np.nanmean(data[idxs]))

    return np.array(binned_data)

