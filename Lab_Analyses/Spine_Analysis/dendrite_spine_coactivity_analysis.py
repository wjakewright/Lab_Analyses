def dendrite_spine_coactivity_analysis(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    dend_activity,
    dend_dFoF,
    spine_groupings,
    spine_flags,
    spine_positions,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to handle the spine-dendrite coaactivity analysis
    
    INPUT PARAMETERS
        spine_activity - 2d np.array of the binarized spine activity
        
        spine_dFoF - 2d np.array of the spine dFoF traces
        
        spine_calcium - 2d np.array of the spine calcium traces
        
        dend_activity - 2d np.array of the dendrite binarized activity
        
        dend_dFoF - 2d np.array of the dendrite dFoF calcium traces
        
        spine_groupings - list of spine groupings 
        
        spine_flags - list containing the spine flags
        
        spine_positions - list or array of the spine positions along the dendrite
        
        activity_window - tuple specifying the time window in sec over which to analyze
        
        cluster_dist - int specifying the distance in um to consider local
        
        sampling_rate - int specifying the imaging sampling rate
        
        volume_norm - tuple list of constants to normalzie spine dFoF and calcium
        
    """

