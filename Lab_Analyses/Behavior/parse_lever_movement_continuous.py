"""Module to parse the lever force into movement and non-movement epochs"""

import warnings

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities.matlab_smooth import matlab_smooth


def parse_lever_movement_continuous(xsg_data):
    """Function to parse the lever force into movement and nonmovement epochs

    INPUT PARAMETERS
        xsg_data - object containing the data from all the xsglog files. This is
                    output from load_xsg_continuous() function.

    OUTPUT PARAMETERS
        lever_active - binarized np.array indicating when lever is active(1) or inactive (0)

        lever_force_resample - np.array of the lever force resampled to 1kHz

        lever_force_smooth -  np.array of the resampled lever force smoothed with
                              a butterworth filter

        lever_velocity_envelope_smooth - np.array of the lever velocity envelope calculated
                                         with hilbert transformation and then smoothed

    """
    ## Constants
    XSG_SAMPLE_RATE = 10000  ### Should always be 10kHz
    BUTTERWORTH_STOP = 5 / 500  ### fraction of nyquist (cutoff = 10Hz)
    MOVETHRESH = 0.0007
    MOVEMENT_LEEWAY = 150
    GAP_ALLOWANCE = 500  # in ms
    MINIMUM_MOVEMENT_FAST = 0  # Change if you want to get rid of small movements
    THRESH_RUN = 10  # 10ms

    # Resample the lever traces to 1kHz
    lever_force_resample = sysignal.resample_poly(
        xsg_data.channels["Lever"], up=1, down=10
    )
    butter = sysignal.butter(4, BUTTERWORTH_STOP, "low")
    lever_force_smooth = sysignal.filtfilt(
        butter[0],
        butter[1],
        lever_force_resample,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(butter[1]), len(butter[0])) - 1),
    )

    # Get lever velocity and smooth
    lever_velocity_resample = np.insert(np.diff(lever_force_smooth), 0, 0, axis=0)
    lever_velocity_resample_smooth = matlab_smooth(lever_velocity_resample, 5).astype(
        np.float64
    )
    lever_velocity_resample_smooth[
        np.argwhere(np.isnan(lever_velocity_resample)).flatten()
    ] = np.nan

    # Get lever velocity envelope
    lever_velocity_hilbert = sysignal.hilbert(lever_velocity_resample_smooth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lever_velocity_envelope = np.sqrt(
            (lever_velocity_hilbert * np.conj(lever_velocity_hilbert))
        ).astype(np.float64)

    # Change window if you would like to smooth envelope velocity
    lever_velocity_envelope_smooth = matlab_smooth(lever_velocity_envelope, window=1)

    # Define active parts of the lever
    lever_active = (lever_velocity_envelope_smooth > MOVETHRESH).astype(int)

    # Give leeway on both ends of to all movements
    movement_leway_filt = np.ones(MOVEMENT_LEEWAY, dtype=int)
    npad = len(movement_leway_filt) - 1
    lever_active_padded = np.pad(
        lever_active, (npad // 2, npad - npad // 2), mode="constant"
    )
    lever_active = (
        np.convolve(lever_active_padded, movement_leway_filt, "valid")
        .astype(bool)
        .astype(int)
    )

    # Close gaps of a determined size
    (
        lever_active_starts,
        lever_active_stops,
        _,
        lever_active_intermovement_times,
    ) = get_lever_active_points(lever_active)
    lever_active_fill = lever_active_intermovement_times < GAP_ALLOWANCE
    for i in np.nonzero(lever_active_fill)[0]:
        lever_active[lever_active_stops[i] : lever_active_starts[i + 1]] = 1

    # Get rid of very small movements
    minimum_movement = MINIMUM_MOVEMENT_FAST + MOVEMENT_LEEWAY
    (
        lever_active_starts,
        lever_active_stops,
        lever_active_movement_times,
        _,
    ) = get_lever_active_points(lever_active)
    lever_active_erase = lever_active_movement_times < minimum_movement
    for i in np.nonzero(lever_active_erase)[0]:
        lever_active[lever_active_starts[i] : lever_active_stops[i]] = 0

    # Edges of hilbert envelope always go up
    # Eliminate the first/last movements if they're on the edges
    (lever_active_starts, lever_active_stops, _, _,) = get_lever_active_points(
        lever_active
    )
    if lever_active_starts[0] == 0:
        lever_active[0 : lever_active_stops[0] + 1] = 0
    if lever_active_stops[-1] == len(lever_force_resample) - 1:
        lever_active[lever_active_starts[-1] :] = 0

    # Refine the lever active starts and stops
    (lever_active_starts, lever_active_stops, _, _,) = get_lever_active_points(
        lever_active
    )
    inactive_lever_bool = [not x for x in lever_active.astype(bool)]
    noise_cutoff = np.std(
        lever_force_resample[inactive_lever_bool]
        - lever_force_smooth[inactive_lever_bool]
    )
    move_start_values = lever_force_smooth[lever_active_starts]
    move_start_cutoffs = move_start_values + noise_cutoff
    move_stop_values = lever_force_smooth[lever_active_stops]
    move_stop_cutoffs = move_stop_values + noise_cutoff
    splits = list(zip(lever_active_starts, lever_active_stops + 1))
    movement_epochs = [lever_force_resample[x:y] for x, y in splits]

    # Look for traces consecutively past threshold
    movement_start_offsets = []
    for w, x, y, z in zip(
        move_start_values, movement_epochs, move_start_cutoffs, lever_active_starts
    ):
        offset = get_move_start_offset(w, x, y, z, THRESH_RUN)
        movement_start_offsets.append(offset.astype(int))

    movement_stop_offsets = []
    for w, x, y, z in zip(
        move_stop_values, movement_epochs, move_stop_cutoffs, lever_active_stops
    ):
        offset = get_move_stop_offset(w, x, y, z, THRESH_RUN)
        movement_stop_offsets.append(offset.astype(int))

    lever_active[np.concatenate(movement_start_offsets)] = 0
    lever_active[np.concatenate(movement_stop_offsets)] = 0

    return (
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
    )


# -------------------------------------------------------------------------------------
# ---------------------------------HELPER FUNCTIONS------------------------------------
# -------------------------------------------------------------------------------------
def get_move_start_offset(w, x, y, z, thresh_run):
    """Helper function to get movement start offsets"""
    u = (np.absolute(x - w) > np.absolute(y - w)).astype(int)
    v = np.ones(thresh_run)
    npad = len(v) - 1
    u_padded = np.pad(u, (npad // 2, npad - npad // 2), mode="constant")
    conv = np.convolve(u_padded, v, "valid")
    flr = np.floor(thresh_run / 2)
    find = np.nonzero(conv >= thresh_run)[0][0]
    end = find - flr
    result = np.arange(z, z + end + 2)

    return result


def get_move_stop_offset(w, x, y, z, thresh_run):
    """Helper function to get movment stop offsets"""
    u = (np.absolute(x[::-1] - w) > np.absolute(y - w)).astype(int)
    v = np.ones(thresh_run)
    npad = len(v) - 1
    u_padded = np.pad(u, (npad // 2, npad - npad // 2), mode="constant")
    conv = np.convolve(u_padded, v, "valid")
    flr = np.floor(thresh_run / 2)
    find = np.nonzero(conv >= thresh_run)[0][0]
    end = z - find + flr - 1  # weird indexing issue
    result = np.arange(end, z + 1)

    return result


def get_lever_active_points(lever_active):
    """Helper function to get active_lever_switch, active_lever_starts, active_lever_stops"""
    lever_active_switch = np.diff(lever_active, prepend=0, append=0)
    lever_active_starts = np.argwhere(lever_active_switch == 1).flatten()
    lever_active_stops = np.argwhere(lever_active_switch == -1).flatten() - 1
    lever_active_movement_times = (lever_active_stops) - (
        lever_active_starts + 1
    )  # Accounting for index differences
    lever_active_intermovement_times = (lever_active_starts[1:]) - lever_active_stops[
        0:-1
    ]  # Accounting for index differences

    return (
        lever_active_starts,
        lever_active_stops,
        lever_active_movement_times,
        lever_active_intermovement_times,
    )

