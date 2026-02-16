import numpy as np
from scipy.signal import lfilter


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

    # Final frame type decision based on both channels
    frame_type = decision_table[(channel_decisions[0], channel_decisions[1])]
    
    return frame_type