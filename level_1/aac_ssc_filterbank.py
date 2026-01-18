import numpy as np
from scipy.signal import lfilter
from aac_utils import kbd_window, sin_window, mdct, i_mdct


def SSC(frame_T, next_frame_T, prev_frame_type):

    # High-Pass Filter coefficients
    b = np.array([0.7548, -0.7548])
    a = np.array([1.0, -0.5095])

    # Get frame type decision for each channel separately
    channel_types = []
    
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
            if mean_prev > 0:
                ds2[l] = s2[l] / mean_prev
            else:
                ds2[l] = 0
        
        # Check for transient
        has_attack = False
        for l in range(1, 8):
            if s2[l] > 1e-3 and ds2[l] > 10:
                has_attack = True
                break
        
        # Decide frame type for THIS channel based on prev_frame_type
        if prev_frame_type == "OLS":
            ch_type = "LSS" if has_attack else "OLS"
        elif prev_frame_type == "LSS":
            ch_type = "ESH"
        elif prev_frame_type == "ESH":
            ch_type = "ESH" if has_attack else "LPS"
        elif prev_frame_type == "LPS":
            ch_type = "OLS"
        else:  # Initialization
            ch_type = "LSS" if has_attack else "OLS"
        
        channel_types.append(ch_type)
    
    # Combine channel decisions into final frame type
    type_ch0 = channel_types[0]
    type_ch1 = channel_types[1]

    # Decision table for final frame type
    decision_table = {
        # Channel 0 = ONLY_LONG_SEQUENCE
        ("OLS", "OLS"): "OLS",
        ("OLS", "LSS"): "LSS",
        ("OLS", "ESH"): "ESH",
        ("OLS", "LPS"): "LPS",
        
        # Channel 0 = LONG_START_SEQUENCE
        ("LSS", "OLS"): "LSS",
        ("LSS", "LSS"): "LSS",
        ("LSS", "ESH"): "ESH",
        ("LSS", "LPS"): "ESH",
        
        # Channel 0 = EIGHT_SHORT_SEQUENCE
        ("ESH", "OLS"): "ESH",
        ("ESH", "LSS"): "ESH",
        ("ESH", "ESH"): "ESH",
        ("ESH", "LPS"): "ESH",
        
        # Channel 0 = LONG_STOP_SEQUENCE
        ("LPS", "OLS"): "LPS",
        ("LPS", "LSS"): "ESH",
        ("LPS", "ESH"): "ESH",
        ("LPS", "LPS"): "LPS",
    }
    
    # Get final frame type
    final_type = decision_table[(type_ch0, type_ch1)]
    
    return final_type


# def filter_bank(frame_T, frame_type, win_type):
#     """
#     Filter Bank - Transform from time to frequency domain (MDCT)

#     Parameters:
#     -----------
#     frame_T : ndarray (2048, 2)
#         Frame in time domain (stereo)
#     frame_type : str
#         Frame type ("OLS", "LSS", "ESH", "LPS")
#     win_type : str
#         Window type ("KBD" or "SIN")

#     Returns:
#     --------
#     frame_F : ndarray (1024, 2)
#         MDCT coefficients
#         - For OLS/LSS/LPS: 1024 coefficients per channel
#         - For ESH: 128*8 coefficients (flattened) per channel
#     """

#     # Create windows
#     if win_type == "KBD":
#         w_long = kbd_window(2048, alpha=6)
#         w_short = kbd_window(256, alpha=4)
#     else:  # "SIN"
#         w_long = sin_window(2048)
#         w_short = sin_window(256)

#     frame_F = np.zeros((1024, 2))

#     for ch in range(2):  # For each channel

#         if frame_type in ["OLS", "LSS", "LPS"]:
#             # Long frames - use entire frame (2048 samples)

#             if frame_type == "OLS":
#                 # Symmetric long window
#                 window = w_long

#             elif frame_type == "LSS":
#                 # LONG_START: [left_half_long, 448x1, right_half_short, 448x0]
#                 window = np.concatenate([
#                     w_long[:1024],      # Left half long
#                     np.ones(448),       # Flat top
#                     w_short[128:],      # Right half short
#                     np.zeros(448)       # Zero padding
#                 ])

#             else:  # "LPS"
#                 # LONG_STOP: [448x0, left_half_short, 448x1, right_half_long]
#                 window = np.concatenate([
#                     np.zeros(448),      # Zero padding
#                     w_short[:128],      # Left half short
#                     np.ones(448),       # Flat top
#                     w_long[1024:]       # Right half long
#                 ])

#             # Apply window and MDCT
#             windowed = frame_T[:, ch] * window
#             frame_F[:, ch] = mdct(windowed, 2048)

#         else:  # "ESH" - EIGHT_SHORT_SEQUENCE
#             # Use only central 1152 samples
#             # Ignore 448 samples at beginning and end
#             center_samples = frame_T[448:448+1152, ch]

#             # 8 subframes of 256 samples with 50% overlap
#             subframe_coeffs = np.zeros((128, 8))

#             for i in range(8):
#                 start = i * 128
#                 end = start + 256
#                 subframe = center_samples[start:end]

#                 # Apply short window
#                 windowed = subframe * w_short

#                 # MDCT -> 128 coefficients
#                 subframe_coeffs[:, i] = mdct(windowed, 256)

#             # Flatten: 128*8 = 1024 coefficients
#             frame_F[:, ch] = subframe_coeffs.flatten()

#     return frame_F


# def i_filter_bank(frame_F, frame_type, win_type):
#     """
#     Inverse Filter Bank - Transform from frequency to time domain (IMDCT)

#     Parameters:
#     -----------
#     frame_F : ndarray (1024, 2)
#         MDCT coefficients
#     frame_type : str
#         Frame type ("OLS", "LSS", "ESH", "LPS")
#     win_type : str
#         Window type ("KBD" or "SIN")

#     Returns:
#     --------
#     frame_T : ndarray (2048, 2)
#         Frame in time domain (stereo)
#     """

#     # Create windows
#     if win_type == "KBD":
#         w_long = kbd_window(2048, alpha=6)
#         w_short = kbd_window(256, alpha=4)
#     else:  # "SIN"
#         w_long = sin_window(2048)
#         w_short = sin_window(256)

#     frame_T = np.zeros((2048, 2))

#     for ch in range(2):  # For each channel

#         if frame_type in ["OLS", "LSS", "LPS"]:
#             # Long frames

#             if frame_type == "OLS":
#                 window = w_long

#             elif frame_type == "LSS":
#                 window = np.concatenate([
#                     w_long[:1024],
#                     np.ones(448),
#                     w_short[128:],
#                     np.zeros(448)
#                 ])

#             else:  # "LPS"
#                 window = np.concatenate([
#                     np.zeros(448),
#                     w_short[:128],
#                     np.ones(448),
#                     w_long[1024:]
#                 ])

#             # IMDCT
#             reconstructed = i_mdct(frame_F[:, ch], 2048)

#             # Apply window
#             frame_T[:, ch] = reconstructed * window

#         else:  # "ESH"
#             # Reconstruct from 8 subframes
#             reconstructed = np.zeros(2048)

#             # Reshape: 1024 -> (128, 8)
#             coeffs_reshaped = frame_F[:, ch].reshape(128, 8)

#             # Overlap-add for each subframe
#             for i in range(8):
#                 # IMDCT for subframe
#                 subframe_time = i_mdct(coeffs_reshaped[:, i], 256)

#                 # Apply window
#                 windowed = subframe_time * w_short

#                 # Overlap-add at correct position
#                 start = 448 + i * 128
#                 end = start + 256
#                 reconstructed[start:end] += windowed

#             frame_T[:, ch] = reconstructed

#     return frame_T
