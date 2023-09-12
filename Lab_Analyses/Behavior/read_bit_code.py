"""Module to read the bitcode output from Dispatcher to get timing of behavior
    and sync with imaging"""

import numpy as np


def read_bit_code(xsg_trial_num):
    """Function to read bitcode output from dispatcher
    
        INPUT PARAMETERS
            xsg_trial_num - np.array of the trial_number located witin the xsg file.
                            Output from load_xsg_continuous as xsg_data.channels["trial_number"]
        
        OUTPUT PARAMETERS
            trial_number - 2d np.array. Col 1 contains the time and Col 2 contains the trial num
        
        NOTES:
            Reads bitcode which has a sync signal followed by 12 bits for the trial number,
            which all have 5ms times with 5ms gaps between them.
            Bitcode is most significant for bit first (2048 to 1).
            Total time: 5ms sync, 5ms * 12 bits = 125ms
            Temporal resolution of linix state machine gives >0.1 loss per 5ms period.
            This leads to ~2.6ms to be lost over the course of 24 states
            
            The start of theh trial is defined as the START of the bitcode
    """

    # Constants
    NUM_BITS = 12
    THRESHOLD_VALUE = 2
    XSG_SAMPLE_RATE = 10000

    binary_threshold = (xsg_trial_num > THRESHOLD_VALUE).astype(np.float64)
    shift_binary_threshold = np.insert(binary_threshold[:-1], 0, np.nan)

    # Get raw times for rising edge of signals
    rising_bitcode = np.nonzero(
        (binary_threshold == 1).astype(int) & (shift_binary_threshold == 0).astype(int)
    )[0]

    # Set up the possible bits, 12 values, most significant first
    bit_values = np.arange(NUM_BITS - 1, -1, -1, dtype=int)
    bit_values = 2 ** bit_values

    # Find the sync bitcode: anything where the differences is larger than the length
    # of the bitcode (16ms - set as 20ms to be safe)
    bitcode_time_samples = 200 * (XSG_SAMPLE_RATE / 1000)  ## 200ms
    bitcode_sync = np.nonzero(np.diff(rising_bitcode) > bitcode_time_samples)[0]

    # Find the sync signal from the 2nd bitcode, coz diff. And this is the index of the rising bitcode [HL]
    # Assume that the first rising edge is a sync signale
    if len(rising_bitcode) != 0:
        # Add first one back and shift back to rising pulse
        # Get the bitcode index in time [HL]
        bitcode_sync = rising_bitcode[np.insert(bitcode_sync + 1, 0, 0)]

        # Test bitcode parameters to find optimal value for smallest loss of tirals
        trial_number_list = []
        value_range = np.arange(9.0, 11.1, 0.1)  # will iterate through these values
        for value in value_range:
            # Initialize tiral_number output
            # col2 = trial_number, col1 = time
            trial_number = np.zeros((len(bitcode_sync), 2))

            # For each bitcode sync, check each bit and record as hi or low
            for idx, curr_bitcode_sync in enumerate(bitcode_sync):
                curr_bitcode = np.zeros(NUM_BITS)

                for curr_bit in np.array(range(NUM_BITS)) + 1:
                    # boundaries for bits: between the half of each break
                    bit_boundary_min = curr_bitcode_sync + (curr_bit - 0.5) * value * (
                        XSG_SAMPLE_RATE / 1000
                    )
                    bit_boundary_max = curr_bitcode_sync + (curr_bit + 0.5) * value * (
                        XSG_SAMPLE_RATE / 1000
                    )

                    a = rising_bitcode > bit_boundary_min
                    b = rising_bitcode < bit_boundary_max
                    if any(a & b):
                        curr_bitcode[curr_bit - 1] = 1

                curr_bitcode_trial = np.sum(curr_bitcode * bit_values)
                trial_number[idx, 1] = curr_bitcode_trial  ## Trial number
                trial_number[idx, 0] = (
                    bitcode_sync[idx] + 1
                ) / XSG_SAMPLE_RATE  ## Time

                # Catch rare instance of the xsg file cutting out before the end of the bitcode
                if bit_boundary_max > len(binary_threshold):
                    trial_number[idx, :] = np.nan

            # Get rid of duplicated trials
            _, indexes = np.unique(trial_number[:, 1], return_index=True)
            trial_number = trial_number[indexes]
            # Remove nan values
            trial_number = trial_number[~np.isnan(trial_number).any(axis=1)]
            # Append trial number of this test to trial_number_list to then evaluate
            trial_number_list.append(trial_number)

        # Evaluate for loss of trials
        losses = []
        for test in trial_number_list:
            losses.append(len(bitcode_sync) - len(test[:, 1]))
        # Choose the trial_number with the smallest loss
        L = min(i for i in losses if i >= 0)
        trial_number = trial_number_list[losses.index(L)]

        # Check for anything strange going on and display warning
        if not all(np.diff(trial_number[:2])):
            print("TRIAL NUMBER WARNING: Non-consecutive trials")

    else:
        trial_number = []

    return trial_number
