import numpy as np
from scipy.signal import lfilter
from aac_utils import kbd_window, sin_window, mdct, i_mdct


def SSC(frame_T, next_frame_T, prev_frame_type):
    """
    Sequence Segmentation Control - determines frame type based on transient detection.
    
    Args:
        frame_T: Current frame (2048x2) - unused but kept for API consistency
        next_frame_T: Next frame (2048x2) - analyzed for attack detection
        prev_frame_type: Previous frame type ('OLS', 'LSS', 'ESH', 'LPS')
    
    Returns:
        frame_type: Frame type string for current frame
    """

    # High-Pass Filter coefficients
    b = [0.7548, -0.7548]
    a = [1.0, -0.5095]

    # Get frame type decision for each channel separately
    channel_decisions = []
    
    for ch in range(2):

        # Apply high-pass filter
        filtered = lfilter(b, a, next_frame_T[:, ch])
        
        # Extract central region (448:1600)
        center_region = filtered[448:448+1152]
        
        # Compute energies
        s2 = np.zeros(8)
        for l in range(8):
            segment = center_region[l*128:(l+1)*128]
            s2[l] = np.sum(segment ** 2)
        
        # Compute attack values
        ds2 = np.zeros(8)
        for l in range(1, 8):
            mean_prev = np.mean(s2[:l])
            ds2[l] = s2[l] / mean_prev if mean_prev > 1e-10 else 0.0
        
        # Check for transient
        has_attack = False
        for l in range(1, 8):
            if s2[l] > 1e-3 and ds2[l] > 10:
                has_attack = True
                break
        
        # Decide frame type for THIS channel based on prev_frame_type
        if prev_frame_type == "LSS":
            ch_type = "ESH"
        elif prev_frame_type == "LPS":
            ch_type = "OLS"
        elif prev_frame_type == "ESH":
            ch_type = "ESH" if has_attack else "LPS"
        else:  # "OLS" or initialization
            ch_type = "LSS" if has_attack else "OLS"
        
        channel_decisions.append(ch_type)

    # Combine channel decisions
    decision_table = {
        ('OLS', 'OLS'): 'OLS', ('OLS', 'LSS'): 'LSS',
        ('OLS', 'ESH'): 'ESH', ('OLS', 'LPS'): 'LPS',
        ('LSS', 'OLS'): 'LSS', ('LSS', 'LSS'): 'LSS',
        ('LSS', 'ESH'): 'ESH', ('LSS', 'LPS'): 'ESH',
        ('ESH', 'OLS'): 'ESH', ('ESH', 'LSS'): 'ESH',
        ('ESH', 'ESH'): 'ESH', ('ESH', 'LPS'): 'ESH',
        ('LPS', 'OLS'): 'LPS', ('LPS', 'LSS'): 'ESH',
        ('LPS', 'ESH'): 'ESH', ('LPS', 'LPS'): 'LPS',
    }
    
    return decision_table[(channel_decisions[0], channel_decisions[1])]


def filter_bank(frame_T, frame_type, win_type):
    """
    Filterbank - applies MDCT transform to time-domain frame.
    
    Args:
        frame_T: Time-domain frame, NumPy array of shape (2048, 2)
                 2048 samples per channel (stereo)
        frame_type: Frame type string
                    'OLS' = ONLY_LONG_SEQUENCE
                    'LSS' = LONG_START_SEQUENCE  
                    'ESH' = EIGHT_SHORT_SEQUENCE
                    'LPS' = LONG_STOP_SEQUENCE
        win_type: Window type string, 'KBD' or 'SIN'
    
    Returns:
        frame_F: MDCT coefficients, NumPy array of shape (1024, 2) for OLS/LSS/LPS
                 or shape (128, 8, 2) for ESH (8 subframes of 128 coefficients each)
    """

    # Create windows
    if win_type == "KBD":
        w_long = kbd_window(2048)   # Auto-selects a = 6 for N=2048
        w_short = kbd_window(256)   # Auto-selects a = 4 for N=256
    else:  # "SIN"
        w_long = sin_window(2048)
        w_short = sin_window(256)

    # Long frames - use entire frame (2048 samples)
    if frame_type in ["OLS", "LSS", "LPS"]:
            
        # Long frames: output is (1024, 2)
        frame_F = np.zeros((1024, 2))

        # Process each channel separately
        for ch in range(2):

            if frame_type == "OLS":
                # ONLY_LONG_SEQUENCE: symmetric long window
                window = w_long

            elif frame_type == "LSS":
                # LONG_START_SEQUENCE: transition to short
                window = np.concatenate([
                    w_long[:1024],      # Left half long (0:1024)
                    np.ones(448),       # Flat top (1024:1472)
                    w_short[128:],      # Right half short (1472:1600) = w_short[128:256]
                    np.zeros(448)       # Zero padding (1600:2048)
                ])

            else:  # "LPS"
                # LONG_STOP_SEQUENCE: transition from short
                window = np.concatenate([
                    np.zeros(448),      # Zero padding (0:448)
                    w_short[:128],      # Left half short (448:576) = w_short[0:128]
                    np.ones(448),       # Flat top (576:1024)
                    w_long[1024:]       # Right half long (1024:2048)
                ])

            # Apply window and MDCT
            windowed = frame_T[:, ch] * window
            frame_F[:, ch] = mdct(windowed, 2048)

    # Short frames - use central 1152 samples
    else:  # "ESH"
        # EIGHT_SHORT_SEQUENCE: output is (128, 8, 2)
        frame_F = np.zeros((128, 8, 2))

        # Process each channel separately
        for ch in range(2):

            # Ignore 448 samples at beginning and end
            center_samples = frame_T[448:448+1152, ch]

            # Process 8 overlapping subframes (50% overlap)
            for i in range(8):
                start_idx = i * 128
                end_idx = start_idx + 256
                subframe = center_samples[start_idx:end_idx]
                    
                # Apply short window and MDCT
                windowed = subframe * w_short
                frame_F[:, i, ch] = mdct(windowed, 256)

    return frame_F


def i_filter_bank(frame_F, frame_type, win_type):
    """
    Inverse Filterbank - applies IMDCT to reconstruct time-domain frame.
    
    Args:
        frame_F: MDCT coefficients
                 - Shape (1024, 2) for OLS/LSS/LPS frames
                 - Shape (128, 8, 2) for ESH frames
        frame_type: Frame type string ('OLS', 'LSS', 'ESH', 'LPS')
        win_type: Window type string ('KBD' or 'SIN')
    
    Returns:
        frame_T: Reconstructed time-domain frame, NumPy array of shape (2048, 2)
                 Requires overlap-add with previous and next frames for perfect reconstruction
    """

    # Create windows
    if win_type == "KBD":
        w_long = kbd_window(2048)   # Auto-selects a = 6 for N=2048
        w_short = kbd_window(256)   # Auto-selects a = 4 for N=256
    else:  # "SIN"
        w_long = sin_window(2048)
        w_short = sin_window(256)

    # Initialize output frame
    frame_T = np.zeros((2048, 2))

    # Long frames - use entire frame (2048 samples)
    if frame_type in ["OLS", "LSS", "LPS"]:

        # Process each channel separately
        for ch in range(2):

            # Construct window (same as forward filterbank)
            if frame_type == "OLS":
                window = w_long
                
            elif frame_type == "LSS":
                window = np.concatenate([
                    w_long[:1024],
                    np.ones(448),
                    w_short[128:],
                    np.zeros(448)
                ])
                
            else:  # "LPS"
                window = np.concatenate([
                    np.zeros(448),
                    w_short[:128],
                    np.ones(448),
                    w_long[1024:]
                ])
            
            # IMDCT and window
            reconstructed = i_mdct(frame_F[:, ch], 2048)
            frame_T[:, ch] = reconstructed * window
    
    # Short frames - use central 1152 samples
    else:  # "ESH"

        # Process each channel separately
        for ch in range(2):

            # Reconstruct 8 short subframes with overlap-add
            for i in range(8):

                # IMDCT for each subframe
                subframe = i_mdct(frame_F[:, i, ch], 256)
                
                # Apply window
                windowed = subframe * w_short
                
                # Overlap-add at correct position in frame
                start_idx = 448 + i * 128
                end_idx = start_idx + 256
                frame_T[start_idx:end_idx, ch] += windowed
    
    return frame_T
