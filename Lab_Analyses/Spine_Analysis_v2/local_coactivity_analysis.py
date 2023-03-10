import numpy as np

from Lab_Analyses.Spine_Analysis_v2.coactive_vs_noncoactive_event_analysis import (
    coactive_vs_noncoactive_event_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.distance_coactivity_rate_analysis import (
    distance_coactivity_rate_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.nearby_spine_density_analysis import (
    nearby_spine_density_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    load_spine_datasets,
    parse_movement_nonmovement_spines,
)
from Lab_Analyses.Spine_Analysis_v2.spine_volume_normalization import (
    load_norm_constants,
)
from Lab_Analyses.Spine_Analysis_v2.variable_distance_dependence import (
    variable_distance_dependence,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def local_coactivity_analysis(
    mice_list,
    session,
    fov_type="apical",
    activity_window=(-2, 4),
    cluster_dist=5,
    zscore=False,
    volume_norm=True,
    partners=None,
    save_ind=False,
    save_grouped=False,
):
    """Function to handle the analysis of local coactivity between spines
    
        INPUT PARAMETERS
            mice_list - list of str specifying all the mice to be analyzed
            
            session - str specifying the session to be analyzed
            
            fov_type - str specifying whether to analyze apical or basal FOVs
            
            activity_window - tuple specifying the window around which the activity
                              should be analyzed (sec)
            
            cluster_dist - int/float speicfying the distance to be considered local
            
            zscore - boolean of whether or not to zscore the activity for analysis
            
            volume_norm - boolean of whether or not to normalize activity by 
                          spine volume
                          
            partners - str specifying whether or not to contrain analysis to 
                        only 'MRS' or 'nMRS' partners
            
            save_ind - boolean specifying whether to save the data for each FOV
            
            save_grouped - boolean specifying whether to save all of the FOVs grouped together
    """
    # Get normalization constants
    if volume_norm:
        try:
            glu_constants = load_norm_constants(fov_type, zscore, "GluSnFr")
        except FileNotFoundError:
            return "GluSnFr constants have not be generated!!!"
        try:
            ca_constants = load_norm_constants(fov_type, zscore, "Calcium")
        except FileNotFoundError:
            return "Calcium constants have not been generated!!!"

    # Analyze each mouse seperately
    analyzed_data = []
    for mouse in mice_list:
        print("----------------------------------------")
        print(f"- Analyzing {mouse}")
        # Load the datasets
        datasets = load_spine_datasets(mouse, session, fov_type)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
            print(f"-- {FOV}")
            data = dataset[session]
            if volume_norm:
                curr_glu_constants = glu_constants[mouse][FOV]
                curr_ca_constants = ca_constants[mouse][FOV]
                all_constants = (curr_glu_constants, curr_ca_constants)
            else:
                curr_glu_constants = None
                curr_ca_constants = None
                all_constants = None

            # Pull relevant data
            ## Some parameters
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            zoom_factor = data.imaging_parameters["Zoom"]
            ## Spine identifiers
            spine_groupings = data.spine_groupings
            spine_flags = data.spine_flags
            followup_flags = data.followup_flags
            spine_positions = data.spine_positions
            ## Spine Volumes
            raw_spine_volume = data.corrected_spine_volume
            raw_followup_volume = data.corrected_followup_volume
            ## Spine activity
            spine_activity = data.spine_GluSnFr_activity
            spine_dFoF = data.spine_GluSnFr_processed_dFoF
            spine_calcium_dFoF = data.spine_calcium_processed_dFoF
            ## Spine movement-related information
            movement_spines = np.array(data.movement_spines)
            rwd_movement_spines = np.array(data.rwd_movement_spines)
            (
                nonmovement_spines,
                nonrwd_movement_spines,
            ) = parse_movement_nonmovement_spines(movement_spines, rwd_movement_spines)
            ## Dendrite activity
            dendrite_activity = data.dendrite_calcium_activity
            dendrite_dFoF = data.dendrite_calcium_processed_dFoF
            ## Dendrite movement-related information
            movement_dendrites = np.array(data.movement_dendrites)
            rwd_movement_dendrites = np.array(data.rwd_movement_dendrites)
            (
                nonmovement_dendrites,
                nonrwd_movement_dendrites,
            ) = parse_movement_nonmovement_spines(
                movement_dendrites, rwd_movement_dendrites
            )
            ## Dendrite poly roi positions and activity
            poly_dendrite_positions = data.poly_dendrite_positions
            poly_dendrite_dFoF = data.poly_dendrite_calcium_processed_dFoF
            ## Behavioral data
            lever_active = data.lever_active
            lever_force = data.lever_force_smooth
            lever_active_rwd = data.rewarded_movement_binary
            lever_inactive = np.absolute(lever_active - 1)

            # zscore activity if specified
            if zscore:
                spine_dFoF = d_utils.z_score(spine_dFoF)
                spine_calcium_dFoF = d_utils.z_score(spine_calcium_dFoF)
                dendrite_dFoF = d_utils.z_score(dendrite_dFoF)
                poly_dendrite_dFoF = [d_utils.z_score(x) for x in poly_dendrite_dFoF]

            # Get volumes in um
            pix_to_um = zoom_factor / 2
            spine_volumes = (np.sqrt(raw_spine_volume) / pix_to_um) ** 2
            followup_volumes = (np.sqrt(raw_followup_volume) / pix_to_um) ** 2

            # Get spine activity rates
            spine_activity_rate = d_utils.calculate_activity_event_rate(spine_activity)

            # Sort out partner list if specified
            if partners == "MRS":
                partner_list = movement_spines
            elif partners == "nMRS":
                partner_list = nonmovement_spines
            elif partners == "rMRS":
                partner_list == rwd_movement_spines
            else:
                partner_list = None

            # Get distance-dependent coactivity rates
            print(f"---- Calulating distance-dependent coactivity")
            ## All periods
            (
                position_bins,
                distance_coactivity_rate,
                distance_coactivity_rate_norm,
                avg_local_coactivity_rate,
                avg_local_coactivity_rate_norm,
                shuff_local_coactivity_rate,
                shuff_local_coactivity_rate_norm,
                real_vs_shuff_coactivity_diff,
                real_vs_shuff_coactivity_diff_norm,
                coactive_spines,
                coactive_norm_spines,
            ) = distance_coactivity_rate_analysis(
                spine_activity,
                spine_positions,
                spine_flags,
                spine_groupings,
                constrain_matrix=None,
                partner_list=partner_list,
                bin_size=5,
                cluster_dist=cluster_dist,
                sampling_rate=sampling_rate,
                norm_method="mean",
                alpha=0.05,
                iterations=1000,
            )
            ## Movement periods
            (
                _,
                mvmt_distance_coactivity_rate,
                mvmt_distance_coactivity_rate_norm,
                mvmt_avg_local_coactivity_rate,
                mvmt_avg_local_coactivity_rate_norm,
                mvmt_shuff_local_coactivity_rate,
                mvmt_shuff_local_coactivity_rate_norm,
                mvmt_real_vs_shuff_coactivity_diff,
                mvmt_real_vs_shuff_coactivity_diff_norm,
                mvmt_coactive_spines,
                mvmt_coactive_norm_spines,
            ) = distance_coactivity_rate_analysis(
                spine_activity,
                spine_positions,
                spine_flags,
                spine_groupings,
                constrain_matrix=lever_active,
                partner_list=partner_list,
                bin_size=5,
                cluster_dist=cluster_dist,
                sampling_rate=sampling_rate,
                norm_method="mean",
                alpha=0.05,
                iterations=1000,
            )
            ## Nonmovement periods
            (
                _,
                nonmvmt_distance_coactivity_rate,
                nonmvmt_distance_coactivity_rate_norm,
                nonmvmt_avg_local_coactivity_rate,
                nonmvmt_avg_local_coactivity_rate_norm,
                nonmvmt_shuff_local_coactivity_rate,
                nonmvmt_shuff_local_coactivity_rate_norm,
                nonmvmt_real_vs_shuff_coactivity_diff,
                nonmvmt_real_vs_shuff_coactivity_diff_norm,
                nonmvmt_coactive_spines,
                nonmvmt_coactive_norm_spines,
            ) = distance_coactivity_rate_analysis(
                spine_activity,
                spine_positions,
                spine_flags,
                spine_groupings,
                constrain_matrix=lever_inactive,
                partner_list=partner_list,
                bin_size=5,
                cluster_dist=cluster_dist,
                sampling_rate=sampling_rate,
                norm_method="mean",
                alpha=0.05,
                iterations=1000,
            )

            # Assess activity and coactivity of nearby spines
            print(f"---- Assessing properties of nearby spines")
            ## Spine activity
            (
                avg_nearby_spine_rate,
                shuff_nearby_spine_rate,
                spine_activity_rate_distribution,
            ) = variable_distance_dependence(
                spine_activity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            ## All local coactivity
            (
                avg_nearby_coactivity_rate,
                shuff_nearby_coactivity_rate,
                local_coactivity_rate_distribution,
            ) = variable_distance_dependence(
                avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            (
                avg_nearby_coactivity_rate_norm,
                shuff_nearby_coactivity_rate_norm,
                local_coactivity_rate_norm_distribution,
            ) = variable_distance_dependence(
                avg_local_coactivity_rate_norm,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            ## Movement local coactivity
            (
                mvmt_avg_nearby_coactivity_rate,
                mvmt_shuff_nearby_coactivity_rate,
                mvmt_local_coactivity_rate_distribution,
            ) = variable_distance_dependence(
                mvmt_avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            (
                mvmt_avg_nearby_coactivity_rate_norm,
                mvmt_shuff_nearby_coactivity_rate_norm,
                mvmt_local_coactivity_rate_norm_distribution,
            ) = variable_distance_dependence(
                mvmt_avg_local_coactivity_rate_norm,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            ## Nonmovement local coactivity
            (
                nonmvmt_avg_nearby_coactivity_rate,
                nonmvmt_shuff_nearby_coactivity_rate,
                nonmvmt_local_coactivity_rate_distribution,
            ) = variable_distance_dependence(
                nonmvmt_avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            (
                nonmvmt_avg_nearby_coactivity_rate_norm,
                nonmvmt_shuff_nearby_coactivity_rate_norm,
                nonmvmt_local_coactivity_rate_norm_distribution,
            ) = variable_distance_dependence(
                nonmvmt_avg_local_coactivity_rate_norm,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )

            # Assess local MRS density and spine volume
            ## MRS density
            (
                spine_density_distribution,
                MRS_density_distribution,
                avg_local_MRS_density,
                shuff_local_MRS_density,
                _,
            ) = nearby_spine_density_analysis(
                movement_spines,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                iterations=1000,
            )
            ## rMRS density
            (
                _,
                rMRS_density_distribution,
                avg_local_rMRS_density,
                shuff_local_rMRS_density,
                _,
            ) = nearby_spine_density_analysis(
                rwd_movement_spines,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                iterations=1000,
            )
            ## Spine volume
            (
                avg_nearby_spine_volume,
                shuff_nearby_spine_volume,
                nearby_spine_volume_distribution,
            ) = variable_distance_dependence(
                spine_volumes,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )

            # Analyze coactive and noncoactive spine events
            ## All events
            (
                nearby_spine_idxs,
                coactive_binary,
                noncoactive_binary,
                spine_coactive_event_num,
                spine_coactive_traces,
                spine_noncoactive_traces,
                spine_coactive_calcium_traces,
                spine_noncoactive_calcium_traces,
                spine_coactive_amplitude,
                spine_noncoactive_amplitude,
                spine_coactive_calcium_amplitude,
                spine_noncoactive_calcium_amplitude,
                spine_coactive_onset,
                spine_noncoactive_onset,
                fraction_spine_coactive,
                fraction_coactivity_participation,
            ) = coactive_vs_noncoactive_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=None,
                partner_list=partner_list,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )
            ## Movement events
            (
                _,
                mvmt_coactive_binary,
                mvmt_noncoactive_binary,
                mvmt_spine_coactive_event_num,
                mvmt_spine_coactive_traces,
                mvmt_spine_noncoactive_traces,
                mvmt_spine_coactive_calcium_traces,
                mvmt_spine_noncoactive_calcium_traces,
                mvmt_spine_coactive_amplitude,
                mvmt_spine_noncoactive_amplitude,
                mvmt_spine_coactive_calcium_amplitude,
                mvmt_spine_noncoactive_calcium_amplitude,
                mvmt_spine_coactive_onset,
                mvmt_spine_noncoactive_onset,
                mvmt_fraction_spine_coactive,
                mvmt_fraction_coactivity_participation,
            ) = coactive_vs_noncoactive_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=lever_active,
                partner_list=partner_list,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )
            ## Nonmovement events
            (
                _,
                nonmvmt_coactive_binary,
                nonmvmt_noncoactive_binary,
                nonmvmt_spine_coactive_event_num,
                nonmvmt_spine_coactive_traces,
                nonmvmt_spine_noncoactive_traces,
                nonmvmt_spine_coactive_calcium_traces,
                nonmvmt_spine_noncoactive_calcium_traces,
                nonmvmt_spine_coactive_amplitude,
                nonmvmt_spine_noncoactive_amplitude,
                nonmvmt_spine_coactive_calcium_amplitude,
                nonmvmt_spine_noncoactive_calcium_amplitude,
                nonmvmt_spine_coactive_onset,
                nonmvmt_spine_noncoactive_onset,
                nonmvmt_fraction_spine_coactive,
                nonmvmt_fraction_coactivity_participation,
            ) = coactive_vs_noncoactive_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=lever_inactive,
                partner_list=partner_list,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )

