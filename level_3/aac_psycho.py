import numpy as np
from scipy.fft import fft
from aac_tns import _load_band_tables


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Main psychoacoustic model function for one channel. 
    Computes the Signal-to-Mask Ratio (SMR) for the current frame.
    
    Args:
        frame_T: Current frame (2048,)
        frame_type: 'OLS', 'LSS', 'ESH', or 'LPS'
        frame_T_prev_1: Previous frame (2048,)
        frame_T_prev_2: 2 frames back (2048,)
    
    Returns:
        SMR: Signal-to-Mask Ratio
            - (69,) for long frames (OLS/LSS/LPS)
            - (42, 8) for ESH frames
    """
    tables = _load_band_tables()

    # Handle first two frames where previous frames are not available
    if frame_T_prev_1 is None:
        frame_T_prev_1 = np.zeros_like(frame_T)
    if frame_T_prev_2 is None:
        frame_T_prev_2 = np.zeros_like(frame_T)
    
    if frame_type == "ESH":
        # Short frames: 8 subframes, use Table B.2.1.9.b
        bands = tables['B219b']
        SMR = np.zeros((42, 8))

        # Process each subframe
        for i in range(8):
            # Extract 256-sample subframe from center region
            start = 448 + i * 128
            subframe = frame_T[start:start+256]
            subframe_prev_1 = frame_T_prev_1[start:start+256]
            subframe_prev_2 = frame_T_prev_2[start:start+256]

            # Compute SMR for this subframe
            SMR[:, i] = _process_frame(subframe, subframe_prev_1, subframe_prev_2, bands)

    else:  # OLS/LSS/LPS
        # Long frames: use Table B.2.1.9.a
        bands = tables['B219a']
        SMR = _process_frame(frame_T, frame_T_prev_1, frame_T_prev_2, bands)
    
    return SMR


def _process_frame(frame, frame_prev_1, frame_prev_2, bands):
    """
    Process one frame (or subframe) through psychoacoustic model.
    
    Args:
        frame: Current frame samples (2048,) or (256,)
        frame_prev1, frame_prev2: Previous frames (same shape)
        bands: Critical band table (B219a or B219b) - 13 bands
    
    Returns:
        SMR: Signal-to-Mask Ratio per scalefactor band
            - Length 69 for long frames (N=2048)
            - Length 42 for short frames (N=256)
    """
    N = len(frame)
    num_bands = len(bands)  # 13 critical bands
    
    # Determine number of scalefactor bands
    if N == 2048:
        num_sf_bands = 69  # Long frames
    else:  # N == 256
        num_sf_bands = 42  # Short frames

    # NOTE: Step 1 (Spreading Function) is defined as a formula and implemented later in Step 6
    
    # Step 2: Hann window and FFT
    n = np.arange(N)
    hann = 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)
    
    # Current frame
    windowed = frame * hann
    spectrum = fft(windowed)
    r = np.abs(spectrum[:N//2])      # Magnitude (1024 or 128 bins)
    f = np.angle(spectrum[:N//2])    # Phase
    
    # Previous frame 1
    windowed_prev_1 = frame_prev_1 * hann
    spectrum_prev_1 = fft(windowed_prev_1)
    r_prev_1 = np.abs(spectrum_prev_1[:N//2])
    f_prev_1 = np.angle(spectrum_prev_1[:N//2])
    
    # Previous frame 2
    windowed_prev_2 = frame_prev_2 * hann
    spectrum_prev_2 = fft(windowed_prev_2)
    r_prev_2 = np.abs(spectrum_prev_2[:N//2])
    f_prev_2 = np.angle(spectrum_prev_2[:N//2])
    
    # Step 3: Predictability
    r_pred = 2 * r_prev_1 - r_prev_2
    f_pred = 2 * f_prev_1 - f_prev_2
    
    # Step 4: Compute predictability measure c[w]
    c_pred = np.zeros(N//2)
    for w in range(N//2):
        real_diff = r[w] * np.cos(f[w]) - r_pred[w] * np.cos(f_pred[w])
        imag_diff = r[w] * np.sin(f[w]) - r_pred[w] * np.sin(f_pred[w])
        numerator = np.sqrt(real_diff**2 + imag_diff**2)
        denominator = r[w] + np.abs(r_pred[w])
        c_pred[w] = numerator / denominator if denominator > 1e-10 else 1.0
    
    # Steps 5: Critical band energy and spreading
    e = np.zeros(num_bands)  # Energy per critical band
    c = np.zeros(num_bands)  # Predictability measure per critical band
    
    for b in range(num_bands):
        w_low = int(bands[b, 1])  # Lower FFT bin
        w_high = min(int(bands[b, 2]), N//2)  # Upper FFT bin

        if w_low < N//2:
            e[b] = np.sum(r[w_low:w_high]**2)
            c[b] = np.sum(c_pred[w_low:w_high] * r[w_low:w_high]**2)
    
    # Step 6: Apply spreading function
    bval = bands[:, 4]  # Bark values for each band

    ecb = np.zeros(num_bands)
    ct = np.zeros(num_bands)
    cb = np.zeros(num_bands)
    en = np.zeros(num_bands)
    
    for b in range(num_bands):
        spread_sum = 0.0 # Normalization factor for spreading

        for bb in range(num_bands):
            # Spreading function
            if bb >= b:
                tmpx = 3.0 * (bval[b] - bval[bb])
            else:
                tmpx = 1.5 * (bval[b] - bval[bb])
            
            tmpz = 8.0 * min((tmpx - 0.5)**2 - 2*(tmpx - 0.5), 0)
            tmpy = 15.811389 + 7.5*(tmpx + 0.474) - 17.5*np.sqrt(1.0 + (tmpx + 0.474)**2)
            
            if tmpy < -100:
                sf = 0.0
            else:
                sf = 10**(tmpz + tmpy / 10.0)
            
            ecb[b] += e[bb] * sf
            ct[b] += c[bb] * sf
            spread_sum += sf

        # Normalized predictability measure
        cb[b] = ct[b] / ecb[b] if ecb[b] > 1e-10 else 1.0

        # Normalized band energy
        en[b] = ecb[b] / spread_sum if spread_sum > 1e-10 else 1.0

    # Step 7: Compute tonality index t[b]
    tb = np.clip(-0.299 - 0.43 * np.log(cb + 1e-10), 0, 1)  # Add small value to avoid log(0)
    
    # Steps 8: Compute SNR
    NMT = 6.0   # Noise Masking Threshold in dB
    TMN = 18.0  # Tone Masking Noise in dB
    SNR = tb * TMN + (1 - tb) * NMT

    # Step 9: Convert dB to energy ratio
    bc = 10**(- SNR / 10.0)

    # Step 10: Compute energy masking threshold
    nb = en * bc
    
    # Step 11: Compute scalefactor bands
    qsthr = bands[:, 5]
    qthr = np.finfo(float).eps * (N / 2) * 10**(qsthr / 10.0)

    # Apply maximum between masking threshold and absolute hearing threshold
    npart = np.maximum(nb, qthr)
    
    # Steps 12: Compute SMR
    SMR = 10 * np.log10(e / (npart + 1e-10))  # Add small value to avoid log(0)

    # Map from 13 critical bands to scalefactor bands (69 or 42)
    SMR = np.interp(np.linspace(0, num_bands - 1, num_sf_bands), np.arange(num_bands), SMR)

    # NOTE: Step 13 (computing T[b] thresholds) is implemented in aac_quantizer.py
    
    return SMR