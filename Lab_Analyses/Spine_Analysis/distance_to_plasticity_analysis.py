import numpy as np

from Lab_Analyses.Spine_Analysis.distance_coactivity_rate_analysis import (
    bin_by_position,
)
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)


def distance_to_plasticity_analysis(
    spine_volumes,
    followup_volumes,
    spine_flags,
    followup_flags,
    spine_positions,
    spine_groupings,
    bin_size=5,
):
    """Function to assess distance to spines undergoing plasticity
    
        INPUT PARAMETERS
            spine_volumes - list or array of the estimated spine volume
            
            followup_volumes - list or array of the estimated spine volume
                                for the followup day

            spine_flags - list containing the spine flags

            followup_flags - list containing the spine flags for the followup day

            spine_positions - list or array of the spine positions along the dendrite

            spine_groupings - list of spine groupings

            bin_size - int or float specifying the distance to bin over

    """
    # Set up position bins
    MAX_DIST = 40
    EXCLUDE = "Shaft Spine"
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # Sort out the spine groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Setup output variables
    relative_volume_matrix = np.zeros((bin_num, len(spine_volumes))) * np.nan
    unbinned_relative_volume = []
    enlarge_prob_matrix = np.zeros((bin_num, len(spine_volumes))) * np.nan
    shrunk_prob_matrix = np.zeros((bin_num, len(spine_volumes))) * np.nan

    # Get relative volumes and classify plasticity
    vol_list = [spine_volumes, followup_volumes]
    flag_list = [spine_flags, followup_flags]

    temp_rel_vols, stable_idxs = calculate_volume_change(
        vol_list, flag_list, norm=False, days=None, exclude=EXCLUDE
    )
    temp_rel_vols = np.array(list(temp_rel_vols.values())[-1])
    enlarged, shrunken, _ = classify_plasticity(
        temp_rel_vols, threshold=0.5, norm=False
    )

    # Put them into array matching positions
    relative_volumes = np.zeros(len(spine_positions)) * np.nan
    enlarged_spines = np.zeros(len(spine_positions)) * np.nan
    shrunken_spines = np.zeros(len(spine_positions)) * np.nan
    relative_volumes[stable_idxs] = temp_rel_vols
    enlarged_spines[stable_idxs] = enlarged
    shrunken_spines[stable_idxs] = shrunken

    # Iterate through each spine grouping
    for spines in spine_groupings:
        positions = np.array(spine_positions)[spines]
        curr_volumes = relative_volumes[spines]
        curr_enlarged = enlarged_spines[spines]
        curr_shrunken = shrunken_spines[spines]

        # Iterate through each spine on this dendrite
        for spine in range(len(curr_volumes)):
            partner_volumes = []
            partner_enlarged = []
            partner_shrunken = []

            # Iterate through each partner
            for partner in range(len(curr_volumes)):
                # Don't compare to self
                if partner == spine:
                    continue
                # Don't compare spine exclude from volume analysis
                if np.isnan(curr_volumes[spine]) == True:
                    partner_volumes.append(np.nan)
                    partner_enlarged.append(np.nan)
                    partner_shrunken.append(np.nan)
                    continue
                if np.isnan(curr_volumes[partner]) == True:
                    partner_volumes.append(np.nan)
                    partner_enlarged.append(np.nan)
                    partner_shrunken.append(np.nan)
                # store values for this partner
                partner_volumes.append(curr_volumes[partner])
                partner_enlarged.append(curr_enlarged[partner])
                partner_shrunken.append(curr_shrunken[partner])

            # Order by relative positions
            curr_pos = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # Normalize distances relative to curr position
            relative_pos = np.array(pos) - curr_pos
            relative_pos = np.absolute(relative_pos)
            # Sort by positional distances
            sorted_volumes = np.array(
                [x for _, x in sorted(zip(relative_pos, partner_volumes))]
            )
            sorted_enlarged = np.array(
                [x for _, x in sorted(zip(relative_pos, partner_enlarged))]
            )
            sorted_shrunken = np.array(
                [x for _, x in sorted(zip(relative_pos, partner_shrunken))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, partner_volumes))]
            )
            unbinned_vols = list(zip(sorted_positions, sorted_volumes))
            unbinned_relative_volume.append(unbinned_vols)

            # Bin the data by position
            binned_volumes = bin_by_position(
                sorted_volumes, sorted_positions, position_bins,
            )
            binned_enlarged = bin_by_position(
                sorted_enlarged, sorted_positions, position_bins,
            )
            binned_shrunken = bin_by_position(
                sorted_shrunken, sorted_positions, position_bins,
            )
            relative_volume_matrix[:, spines[spine]] = binned_volumes
            enlarge_prob_matrix[:, spines[spine]] = binned_enlarged
            shrunk_prob_matrix[:, spines[spine]] = binned_shrunken

    return (
        relative_volume_matrix,
        unbinned_relative_volume,
        enlarge_prob_matrix,
        shrunk_prob_matrix,
    )

