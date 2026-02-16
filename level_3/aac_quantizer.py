import numpy as np
from aac_tns import _load_band_tables

# Constants
MAGIC_NUMBER = 0.4054
MAX_SCALEFACTOR = 60

def aac_quantizer(frame_F, frame_type, SMR):
    """
    Quantizes MDCT coefficients for one channel.
    
    Args:
        frame_F: MDCT after TNS for one channel
            - (1024,) for long frames
            - (128, 8) for ESH
        frame_type: 'OLS', 'LSS', 'ESH', or 'LPS'
        SMR: Signal-to-Mask Ratio for one channel (in dB)
            - (69,) for long frames
            - (42, 8) for ESH
    
    Returns:
        S: Quantized coefficients (1024,) or (128, 8)
        sfc: Scalefactors per band (NB,) or (NB, 8)
        G: Global gain (1,) or (8,)
    """
    
    # Constants
    MAGIC_NUMBER = 0.4054
    
    # Determine parameters based on frame type
    if frame_type == "ESH":
        # Short frames: 8 subframes of 128 coefficients each
        num_subframes = 8
        num_sf_bands = 42
        S = np.zeros((128, 8))
        sfc = np.zeros((num_sf_bands, 8))
        G = np.zeros(8)
        
        # Process each subframe independently
        for i in range(8):
            X = frame_F[:, i]  # (128,)
            SMR_sub = SMR[:, i]  # (42,)
            
            # Quantize this subframe
            S[:, i], sfc[:, i], G[i] = _quantize_subframe(
                X, SMR_sub, num_sf_bands, MAGIC_NUMBER
            )
    else:
        # Long frames: single frame of 1024 coefficients
        num_sf_bands = 69
        X = frame_F  # (1024,)
        SMR_frame = SMR.flatten()  # (69,)
        
        # Quantize
        S, sfc, G_val = _quantize_subframe(
            X, SMR_frame, num_sf_bands, MAGIC_NUMBER
        )
        G = np.array([G_val])
    
    return S, sfc, G


def _quantize_subframe(X, SMR, num_sf_bands, MAGIC_NUMBER):
    """
    Quantize one subframe or full frame.
    
    Args:
        X: MDCT coefficients (1024,) or (128,)
        SMR: SMR per scalefactor band (69,) or (42,)
        num_sf_bands: Number of scalefactor bands
        MAGIC_NUMBER: Quantization constant (0.4054)
    
    Returns:
        S: Quantized coefficients
        sfc: Scalefactors
        G: Global gain
    """
    N = len(X)
    
    # Step 1: Compute Magic Number (MQ)
    max_coef = np.max(np.abs(X))
    if max_coef > 0:
        MQ = (16.0 / 3.0) * np.log2(max(max_coef**(3.0/4.0), 1e-10))
    else:
        MQ = 0
    MQ = np.clip(MQ, -2**(13-1), 2**(13-1) - 1)  # Clamp to valid range
    
    # Step 2: Initialize global gain and scalefactors
    G = 0
    sfc = np.zeros(num_sf_bands)
    
    # TODO: Proper scalefactor band boundaries needed
    # For now, distribute coefficients evenly across bands
    coefs_per_band = N // num_sf_bands
    
    # Step 3: Simple quantization (without iterative adjustment)
    # Quantization formula: S[k] = sign(X[k]) * int(|X[k]|^(3/4) * 2^(1/4) / MN)
    scale = 2**(1.0/4.0) / MAGIC_NUMBER
    
    S = np.zeros(N)
    for k in range(N):
        if X[k] != 0:
            # Apply quantization
            abs_val = np.abs(X[k])
            quantized = int(abs_val**(3.0/4.0) * scale)
            S[k] = np.sign(X[k]) * quantized
    
    # DPCM encoding of scalefactors (difference coding)
    # sfc[b] = λ[b] - λ[b-1], with sfc[0] = λ[0] - G
    # For simplicity, keep scalefactors at 0 for now
    
    return S, sfc, G


def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse quantizer - reconstructs MDCT coefficients.
    
    Args:
        S: Quantized coefficients (1024,) or (128, 8)
        sfc: Scalefactors (NB,) or (NB, 8)
        G: Global gain (1,) or (8,)
        frame_type: 'OLS', 'LSS', 'ESH', or 'LPS'
    
    Returns:
        frame_F: Reconstructed MDCT coefficients (1024,) or (128, 8)
    """
    
    # Constants
    MAGIC_NUMBER = 0.4054
    
    if frame_type == "ESH":
        # Short frames: 8 subframes
        frame_F = np.zeros((128, 8))
        
        for i in range(8):
            frame_F[:, i] = _dequantize_subframe(
                S[:, i], sfc[:, i], G[i], MAGIC_NUMBER
            )
    else:
        # Long frames
        frame_F = _dequantize_subframe(
            S, sfc, G[0], MAGIC_NUMBER
        )
    
    return frame_F


def _dequantize_subframe(S, sfc, G, MAGIC_NUMBER):
    """
    Dequantize one subframe.
    
    Args:
        S: Quantized coefficients (N,)
        sfc: Scalefactors
        G: Global gain
        MAGIC_NUMBER: Quantization constant
    
    Returns:
        X: Reconstructed MDCT coefficients
    """
    N = len(S)
    
    # Inverse quantization: X[k] = sign(S[k]) * |S[k]|^(4/3) * 2^(1/4) * MN
    scale = 2**(1.0/4.0) * MAGIC_NUMBER
    
    X = np.zeros(N)
    for k in range(N):
        if S[k] != 0:
            abs_val = np.abs(S[k])
            X[k] = np.sign(S[k]) * (abs_val**(4.0/3.0)) * scale
    
    # Apply scalefactors (TODO: proper scalefactor band mapping)
    # For now, reconstruction without scalefactor adjustment
    
    return X

