"""Module to process lever press behavior from Dispatcher and Ephus outputs

    CREATOR - William (Jake) Wright 1/6/2022"""

import os
import re
from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities.load_mat_files import load_mat
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


# --------------------------------------------------------------------
# -------------------ANALYZE LEVER PRESS BEHAVIOR---------------------
# --------------------------------------------------------------------
def analyze_lever_press_behavior(path, imaged, save=False, save_suffix=None):
    """Function to process lever press behavioral data
    
        INPUT PARAMETERS
            path - string indicating the path where all of the behavior
                    files are located. Should contain all files from
                    dispatcher and ephus

            imaged - boolean True or False indicating if the behavioral session
                    was also imaged
            
            save - boolean True or False to save the data at the end or not
                  Default is set to False

            save_suffix - optional string to be appended at the end of the file name
                          Used to indicate any additional info about the sesson.
                          Default is set to None
                    
        OUTPUT PAREAMTERS
            behavior_data - dataclass object containing the behavior data
                            Contains:
                            dispatcher_data - object containing all the native
                                            data from dispatcher directly loaded 
                                            from matlab

                            xsg_data - xsglog_Data object containing the xsglog data

                            lever_active - binary array indicating when the lever was
                                            being actively moved

                            lever_force_resample - np.array of the lever force resampled to 1kHz
                            
                            lever_force_smooth -  np.array of the resampled lever force smoothed with 
                                                   a butterworth filter

                            lever_velocity_envelope_smooth - np.array of the lever velocity envelope calculated
                                                            with hilbert transformation and then smoothed 
                            
                            behavior_frames - array containing the data for each behavioral trial. 
                                              Data for each trial is stored in an object
                                              
                            imaged_trials - logical array indicating which trials the imaging was 
                                            also performed during
                                            
                            frame_times - array with the time (sec) of each imaging frame
    """
    # Load xsg data
    xsg_data = load_xsg_continuous(path)
    # parse the lever movement
    (
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
    ) = parse_lever_movement_continuous(xsg_data)
    # match behvior to imaging frames
    fnames = os.listdir(path)
    # Get dispatcher filename
    dispatcher_fname = []
    for fname in fnames:
        if "data_@lever2p" in fname:
            dispatcher_fname.append(fname)
    # Make sure only one dispatcher file
    if len(dispatcher_fname) > 1:
        raise Exception(
            "More than one dispatcher file found!!! Move or delete one of the files."
        )
    if imaged is True:
        (
            dispatcher_data,
            behavior_frames,
            imaged_trials,
            frame_times,
        ) = dispatcher_to_frames_continuous(dispatcher_fname[0], path, xsg_data, imaged)
    else:
        dispatcher_data = dispatcher_to_frames_continuous(
            dispatcher_fname[0], path, xsg_data, imaged
        )
        behavior_frames = np.array([])
        imaged_trials = np.array([])
        frame_times = np.array([])

    behavior_data = Behavior_Data(
        dispatcher_data,
        xsg_data,
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
        behavior_frames,
        imaged_trials,
        frame_times,
    )

    if save is True:
        id = re.sarch("[A-Z]{2}[0-9]{3,4}", dispatcher_fname).group()
        date = re.search("[0-9]{6}", dispatcher_fname).group()
        if save_suffix is not None:
            save_name = f"{id}_{date}_lever_behavior_{save_suffix}"
        else:
            save_name = f"{id}_{date}_lever_behavior"
        save_pickle(save_name, behavior_data, path)
    else:
        pass

    return behavior_data


@dataclass
class Behavior_Data:
    """Dataclass for storing the final behavioral data output"""

    dispatcher_data: object
    xsg_data: object
    lever_active: np.ndarray
    lever_force_resample: np.ndarray
    lever_force_smooth: np.ndarray
    lever_velocity_envelope_smooth: np.ndarray
    behavior_frames: np.ndarray
    imaged_trials: np.ndarray
    frame_times: np.ndarray


# -------------------------------------------------------------------
# ----------------------EXTRACT DISPATCHER FRAMES--------------------
# -------------------------------------------------------------------
def dispatcher_to_frames_continuous(file_name, path, xsg_data, imaged):
    """ Function to convert dispatcher behavior data into frames to match 
        with imaging data
        
        INPUT PARAMETERS
            file_name - string containing the file name of the dispatcher file

            path - string containing the path to where the file is loacted
            
            xsg_data - object containing the data from all the xsglog files. This
                       is output from load_xsg_continuous() function

            imaged - boolean true or false if behavioral data was also imaged
        
        OUTPUT PARAMETERS
            dispatcher_data - object containing all the native
                              data from dispatcher directly loade from matlab
            behavior_frames - object containing the behavioral data converted to match
                              imaging frames
            
            imaged_trials - np.array logical of which trials were imaged
            
            frame_times - the time (sec) of each image frame
    
    """
    # Load the structures within the dispatcher .mat file
    mat_saved = load_mat(fname=file_name, fname1="saved", path=path)
    mat_saved_autoset = load_mat(fname=file_name, fname1="saved_autoset", path=path)
    mat_saved_history = load_mat(fname=file_name, fname1="saved_history", path=path)
    dispatcher_data = Dispatcher_Data(mat_saved, mat_saved_autoset, mat_saved_history)
    if imaged is False:
        return dispatcher_data

    bhv_frames = mat_saved_history.ProtocolsSection_parsed_events
    imaged_trials = np.zeros(len(bhv_frames))

    # Get trial offsets in samples - must be hardcoded since not stored in xsg raw files
    xsg_sample_rate = 10000

    # Get frame times (sec) from frame trigger trace
    frame_trace = xsg_data.channels["Frame"]
    frame_times = (
        np.nonzero(
            (frame_trace[1:] > 2.5).astype(int) & (frame_trace[:-1] < 2.5).astype(int)
        )[0]
        + 1
    )
    frame_times = (frame_times + 1) / xsg_sample_rate

    # Get trials in raw samples since started
    trial_channel = xsg_data.channels["Trial_number"]
    curr_trial_list = read_bit_code(trial_channel)

    # Loop through trials and find the offsets
    for idx, curr_trial in enumerate(curr_trial_list[:, 1]):
        # skip if it's the last trial and not completed in behavior
        curr_trial = curr_trial.astype(int) - 1
        if curr_trial >= len(bhv_frames) or curr_trial < 0:
            continue
        # the start time is the rise of the first bitcode
        curr_bhv_start = bhv_frames[curr_trial].states.bitcode[0]
        curr_xsg_bhv_offset = curr_bhv_start - curr_trial_list[idx, 0]
        # Apply the offset to all numbers within the trial
        # Find all fields in overall structure of trial
        curr_fieldnames = bhv_frames[curr_trial]._fieldnames
        # Determine which trials were imaged
        bhv_window = (
            bhv_frames[curr_trial].states.state_0 - curr_xsg_bhv_offset
        )  ## Start-time to stop-time of behavioral trial (sec)
        # Get the frame times within the current behavioral trial window
        a = (frame_times > bhv_window[0, 1]).astype(int)
        b = (frame_times > bhv_window[1, 0]).astype(int)
        imaged_frames = (
            np.round(frame_times[np.nonzero(a & b)] * xsg_sample_rate).astype(int) - 1
        )
        # Extract the voltage signals indicating whether imaging frames were captured during this window
        frame_trace_window = frame_trace[imaged_frames]

        if np.sum(frame_trace_window):
            imaged_trials[curr_trial] = 1
        else:
            imaged_trials[curr_trial] = 0

        for curr_field in curr_fieldnames:
            # get subfields
            curr_field_data = getattr(bhv_frames[curr_trial], curr_field)
            curr_subfields = curr_field_data._fieldnames
            # find which subfields are numeric
            curr_numeric_subfields = [
                x
                for x in curr_subfields
                if type(getattr(curr_field_data, x)) == np.ndarray
            ]
            # subtract offset from numeric fields and convert to frames
            for s_field in curr_numeric_subfields:
                # pull subfield data
                s_field_data = getattr(curr_field_data, s_field)
                # compensate for offset
                curr_bhv_times = s_field_data - curr_xsg_bhv_offset
                # convert to frames (get the closest frame from frame time)
                curr_bhv_frames = np.empty(np.shape(curr_bhv_times)).flatten()
                for index, _ in enumerate(curr_bhv_frames):
                    # get index of closest frame [HL]
                    curr_bhv_frames[index] = np.argmin(
                        np.absolute(frame_times - curr_bhv_times.flatten()[index])
                    )
                # Update the current subfield value in the object
                curr_bhv_frames.reshape(np.shape(curr_bhv_times))
                new_curr_field = setattr(curr_field_data, s_field, curr_bhv_frames)
            # Update the current field value in the object
            setattr(bhv_frames[curr_trial], curr_field, new_curr_field)

    return dispatcher_data, bhv_frames, imaged_trials, frame_times


def read_bit_code(xsg_trial):
    """Helper function to help read the bitcode from Dispatcher
    
        INPUT PARAMETERS
            xsg_trial - np.array of the trial_number located within the xsg file. 
                        Output from load_xsg_continuous as xsg_data.channels['trial_number']
        
        OUTPUT PARAMETERS
            trial_number - 2d np.array. Col 1 contains the time and Col 2 contains the trial number

        NOTES:
            Reads bitcode which has a sync signal followed by 12 bits for the trial number,
            which all have 5ms times with 5ms gaps in between.
            Bitcode is most significant bit first (2048 to 1).
            Total time: 5ms sync, 5ms*12gaps +5ms*12bits = 125ms
            The temporal resolution of linux state machine give >0.1ms loss per 5ms period.
            This causes ~2.6ms to be lost over the course of 24 states

            The start of the trial is defined as the START of the bitcode
    """
    num_bits = 12
    threshold_value = 2

    xsg_sample_rate = 10000
    binary_threshold = (xsg_trial > threshold_value).astype(float)
    shift_binary_threshold = np.insert(binary_threshold[:-1], 0, np.nan)
    # Get raw times for rising edge of signals
    rising_bitcode = np.nonzero(
        (binary_threshold == 1).astype(int) & (shift_binary_threshold == 0).astype(int)
    )[0]

    # Set up the possible bits, 12 values, most significant first
    bit_values = np.arange(num_bits - 1, -1, -1, dtype=int)
    bit_values = 2 ** bit_values

    # Find the sync bitcodes: anything where the differences is larger than the length
    # of the bitcode (16ms - set as 20 ms to be safe)
    bitcode_time_samples = 125 * (xsg_sample_rate / 1000)  ## 125ms
    bitcode_sync = np.nonzero(np.diff(rising_bitcode) > bitcode_time_samples)[0]

    # Find the sync signal from the 2nd bitcode, coz diff. And this is index of rising bitcode [HL]
    # Assume that the first rising edge is a sync signal
    if len(rising_bitcode) == 0:
        trial_number = []
    else:
        # Add first one back and shift back to rising pulse; get the bitcode index in time [HL]
        bitcode_sync = rising_bitcode[np.insert(bitcode_sync + 1, 0, 0)]
        # Initialize trial_number output -col 2 = trial number, col 1 = time
        trial_number = np.zeros((len(bitcode_sync), 2))
        # For each bitcode sync, check each bit and record as hi or low
        for i, curr_bitcode_sync in enumerate(bitcode_sync):
            curr_bitcode = np.zeros(num_bits)
            for curr_bit in np.array(range(num_bits)) + 1:
                # boundaries for bits: between the half of each break
                # (bitcode_sync+5ms+2.5ms = 7.5ms)
                bit_boundary_min = (
                    curr_bitcode_sync
                    + (7.5 * (xsg_sample_rate / 1000))
                    + ((curr_bit - 1) * 10.2 * (xsg_sample_rate / 1000))
                )
                bit_boundary_max = (
                    curr_bitcode_sync
                    + (7.5 * (xsg_sample_rate / 1000))
                    + ((curr_bit) * 10.2 * (xsg_sample_rate / 1000))
                )
                a = rising_bitcode > bit_boundary_min
                b = rising_bitcode < bit_boundary_max
                if any(a & b):
                    curr_bitcode[curr_bit - 1] = 1

            curr_bitcode_trial = np.sum(curr_bitcode * bit_values)
            # trial_number col 2 is trial number
            trial_number[i, 1] = curr_bitcode_trial
            # trial_number col 1 is time (sec)
            trial_number[i, 0] = (bitcode_sync[i] + 1) / xsg_sample_rate

            # Catch rare instance of the xsg file cutting out before the end of the bitcode
            if bit_boundary_max > len(binary_threshold):
                trial_number[i, :] = []

        # Check for anything strange going on and display warning
        if not all(np.diff(trial_number[:2])):
            print("TRIAL NUMBER WARNING: Nonconsecutive trials")

    return trial_number


@dataclass
class Dispatcher_Data:
    """Dataclass to store the native dispatcher data loaded from Matlab"""

    saved: object
    autoset: object
    saved_history: object


# ------------------------------------------------------------------
# ----------------------EXTRACT LEVER DATA--------------------------
# ------------------------------------------------------------------
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
    # Set xsg sampling rate
    xsg_sample_rate = 10000  ## Should always be 10kHz
    # Resample the lever traces to 1kHz
    lever_force_resample = sysignal.resample_poly(
        xsg_data.channels["Lever"], up=1, down=10
    )
    butterworth_stop = 5 / 500  # fraction of nyquist (cutoff = 10 Hz)
    butter = sysignal.butter(4, butterworth_stop, "low")
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
    lever_velocity_envelope = np.sqrt(
        (lever_velocity_hilbert * np.conj(lever_velocity_hilbert))
    ).astype(np.float64)
    # Change window if you would like to smooth envelope velocity
    lever_velocity_envelope_smooth = matlab_smooth(lever_velocity_envelope, 1)

    # Define active parts of the lever
    movethresh = 0.0007
    lever_active = (lever_velocity_envelope_smooth > movethresh).astype(int)
    # Give leeway on both ends to all movements
    movement_leeway = 150  # ms to extend the movment total (half on each end)
    movement_leeway_filt = np.ones(movement_leeway, dtype=int)
    npad = len(movement_leeway_filt) - 1
    lever_active_padded = np.pad(
        lever_active, (npad // 2, npad - npad // 2), mode="constant"
    )
    lever_active = (
        np.convolve(lever_active_padded, movement_leeway_filt, "valid")
        .astype(bool)
        .astype(int)
    )
    # Close gaps of detrmined size
    gap_allowance = 500  # in ms
    (
        lever_active_starts,
        lever_active_stops,
        _,
        lever_active_intermovement_times,
    ) = get_lever_active_points(lever_active)
    lever_active_fill = lever_active_intermovement_times < gap_allowance
    for i in np.nonzero(lever_active_fill)[0]:
        lever_active[lever_active_stops[i] : lever_active_starts[i + 1]] = 1
    # Get rid of small movments
    minimum_movement_fast = 0  ## Not using since small movements seemed real
    minimum_movemet = minimum_movement_fast + movement_leeway
    (
        lever_active_starts,
        lever_active_stops,
        lever_active_movement_times,
        _,
    ) = get_lever_active_points(lever_active)
    lever_active_erase = lever_active_movement_times < minimum_movemet
    for i in np.nonzero(lever_active_erase)[0]:
        lever_active[lever_active_starts[i] : lever_active_stops[i]] = 0

    # Edges of hilbert envelope always goes up
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
    noise = (
        lever_force_resample[np.where(lever_active == 0)]
        - lever_force_smooth[np.where(lever_active == 0)]
    )
    noise_cutoff = np.percentile(noise, 99)
    move_start_values = lever_force_smooth[lever_active_starts]
    move_start_cutoffs = move_start_values + noise_cutoff
    move_stop_values = lever_force_smooth[lever_active_stops]
    move_stop_cutoffs = move_stop_values + noise_cutoff
    splits = list(zip(lever_active_starts, lever_active_stops + 1))
    movement_epochs = [lever_force_resample[x:y] for x, y in splits]
    # Look for trace consecutively past threshold
    thresh_run = 3
    movement_start_offsets = []
    for w, x, y, z in zip(
        move_start_values, movement_epochs, move_start_cutoffs, lever_active_starts
    ):
        offset = get_move_start_offset(w, x, y, z, thresh_run)
        movement_start_offsets.append(offset.astype(int))
    movement_stop_offsets = []
    for w, x, y, z in zip(
        move_stop_values, movement_epochs, move_stop_cutoffs, lever_active_starts
    ):
        offset = get_move_stop_offset(w, x, y, z, thresh_run)
        movement_stop_offsets.append(offset.astype(int))

    lever_active[np.concatenate(movement_start_offsets)] = 0
    lever_active[np.concatenate(movement_stop_offsets)] = 0

    return (
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
    )


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
    result = np.arange(z, z + end + 1)

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
    lever_active_intermovement_times = (
        (lever_active_starts[1:] + 1) - lever_active_stops[0:-1]
    )  # Accounting for index differences

    return (
        lever_active_starts,
        lever_active_stops,
        lever_active_movement_times,
        lever_active_intermovement_times,
    )


def matlab_smooth(data, window):
    """Helper function to replicate the implementation of matlab smooth function
    
        INPUT PARAMETERS
            data - 1d numpy array
            
            window - int. Must be odd value
            
    """
    out0 = np.convolve(data, np.ones(window, dtype=int), "valid") / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(data[: window - 1])[::2] / r
    stop = (np.cumsum(data[:-window:-1])[::2] / r)[::1]

    return np.concatenate((start, out0, stop))


# ---------------------------------------------------------------
# -------------------LOAD XSG.LOG DATA FILES --------------------
# ---------------------------------------------------------------
def load_xsg_continuous(dirname):
    """Function to load xsg files output from ephus
        
        INPUT PARAMETERS 
            dirname - string with the path to directory where files are located

        OUTPUT PARAMETERS
            data - xsglog_data dataclass with attributes:
                        name - str of the file name
                        epoch - str of the epoch name
                        file_info- FileInfo dataclass
                        channels - dictionary with key value pairs for each file name
                                    and file data (np.array)
    """
    # Get all the xsglog file names
    files = [file for file in os.listdir(dirname) if file.endswith(".xsglog")]
    # Get all file names and info
    for file in files:
        fullname = os.path.join(dirname, file)
        fname = file
        name, epoch, _ = parse_xsg_filename(fname)
        byte = os.path.getsize(fullname)
        cdate = os.path.getctime(fullname)
        mdate = os.path.getmtime(fullname)
        file_info = FileInfo(name, dirname, cdate, mdate, byte)

        if file == files[0]:
            name_first = name
            epoch_first = epoch
        else:
            if name_first != name or epoch_first != epoch:
                raise NameError(
                    "Files do not match. Only one epoch is allowed. Delete/move unused files."
                )
    # Create the data object
    data = xsglog_Data(name, epoch, file_info)
    # Load the text file
    fn = name_first + ".txt"
    try:
        with open(os.path.join(dirname, fn), "r") as fid:
            data.txt = fid.read()
    except FileNotFoundError:
        print(f"Could not open {os.path.join(dirname,fn)}")
    # Load and store each xsglog file
    data.channels = {}
    for file in files:
        fn = file
        _, _, channel_name = parse_xsg_filename(fn)
        try:
            with open(os.path.join(dirname, fn), "rb") as fid:
                fdata = np.fromfile(fid, np.float64)
        except FileNotFoundError:
            print(f"Could not open {os.path.join(dirname,fn)}")
        data.channels[channel_name] = fdata

    # Check if all the data files are of the same length
    lens = map(len, data.channels.values())
    if len(set(lens)) != 1:
        raise ValueError("Mismatched data size!!!")

    return data


def parse_xsg_filename(fname):
    """ Function to parse filenames into substrings
    
        INPUT PARAMETERS
            fname - string of the filename to be parsed
            
    """
    name = re.search("[A-Z]{2}[0-9]{4}", fname).group()
    epoch = re.search("[A-Z]{4}[0-9]{4}", fname).group()
    # channel = re.search("_(\w+).xsglog", fname).group()[1:]
    channel = re.search("_(\w+)", fname).group()[1:]

    return name, epoch, channel


@dataclass
class FileInfo:
    """Dataclass to store file info"""

    fname: str
    dir: str
    cdate: str
    mdate: str
    bytes: int


@dataclass
class xsglog_Data:
    """Dataclass for storing the xsglog data"""

    name: str
    epoch: str
    file_info: FileInfo

