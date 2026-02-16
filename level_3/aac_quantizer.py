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
        S: Quantized coefficients (1024,) no matter the frame type
        sfc: Scalefactors per band (69,) for long frames or (42, 8) for ESH
        G: Global gain (8,) for ESH or scalar for long frames
    """
    # Load band tables (cached after first call)
    tables = _load_band_tables()
    
    if frame_type == "ESH":
        # Short frames: 8 subframes, use Table B.2.1.9.b
        bands = tables['B219b']
        S_list = []
        sfc = np.zeros((42, 8))
        G = np.zeros(8)
        
        # Process each subframe independently
        for i in range(8):
            S_sub, sfc[:, i], G[i] = _quantize_subframe(frame_F[:, i], SMR[:, i], bands)
            S_list.append(S_sub)  # 128 quantized coefficients
        
        # Concatenate all 8 subframes into single vector
        S = np.concatenate(S_list)
        
    else:  # OLS/LSS/LPS
        # Long frames: use Table B.2.1.9.a
        bands = tables['B219a']
        S, sfc, G = _quantize_subframe(frame_F, SMR, bands)

        # Correct output formats
        G = np.array([G])
        sfc = sfc.astype(int)
    
    return S, sfc, G


def _quantize_subframe(X, SMR, bands):
    """
    Quantize one subframe or full frame for ONE channel.
    
    Implements Steps 13 (threshold computation) and full quantization algorithm
    from Section 2.6 of the specification.
    
    Args:
        X: MDCT coefficients - (1024,) for long or (128,) for short
        SMR: Signal-to-Mask Ratio - (69,) for long or (42,) for short
        bands: Band table (B219a for long, B219b for short)
    
    Returns:
        S: Quantized coefficients (1024,) or (128,) 
        sfc: DPCM-encoded scale factors - (69,) or (42,)
        G: Global gain (scalar)
    """
    N = len(X)
    num_bands = len(bands)
    
    # P[b] = sum of squared MDCT coefficients in band b
    P = np.zeros(num_bands)
    for b in range(num_bands):
        w_low = int(bands[b, 1])
        w_high = min(int(bands[b, 2]), N)
        if w_low < N and w_high > w_low:
            P[b] = np.sum(X[w_low:w_high] ** 2)
    
    # Compute energy thresholds T[b] from SMR (convert from dB)
    SMR_linear = 10 ** (SMR / 10.0)
    T = P / (SMR_linear + 1e-10)  # Avoid division by zero
    
    # Step 1: Calculate initial scale factor estimate for all bands
    max_X = np.max(np.abs(X))
    if max_X < 1e-10:  # Essentially silence
        return np.zeros(N, dtype=int), np.zeros(num_bands), 0.0
    
    MQ = 8191  # Maximum quantization levels
    alpha_hat = (16/3) * np.log2(max(max_X ** 0.75, 1e-10) / MQ)

    # Initialize all scale factors with this value
    alpha = np.full(num_bands, alpha_hat)

    # Initialize empty arrays
    S = np.zeros(N, dtype=int)
    X_hat = np.zeros(N)

    # Initial quantization
    for b in range(num_bands):
        w_low = int(bands[b, 1])
        w_high = min(int(bands[b, 2]), N)
        if w_low < N:
            S[w_low:w_high] = _quantize(X[w_low:w_high], alpha[b])
            X_hat[w_low:w_high] = _dequantize(S[w_low:w_high], alpha[b])

    # Step 2: Iterative optimization loop
    max_iterations = 100  # Safety limit

    for iteration in range(max_iterations):
        # Compute quantization error power per band
        Pe = np.zeros(num_bands)

        for b in range(num_bands):
            w_low = int(bands[b, 1])
            w_high = min(int(bands[b, 2]), N)
            if w_low < N and w_high > w_low:
                Pe[b] = np.sum((X[w_low:w_high] - X_hat[w_low:w_high]) ** 2)
        
        # Check which bands can be quantized more coarsely
        bands_to_increase = (Pe < T)  # Error below threshold
        
        # Also check maximum difference constraint
        if num_bands > 1:
            max_diff = np.max(np.abs(np.diff(alpha)))
            if max_diff > MAX_SCALEFACTOR:
                break  # Stop if constraint violated
        
        # If no bands can be increased, we're done
        if not np.any(bands_to_increase):
            break
        
        # Increase alpha for bands below threshold
        alpha[bands_to_increase] += 1
        
        # Re-quantize with new alpha values
        for b in np.where(bands_to_increase)[0]:
            w_low = int(bands[b, 1])
            w_high = min(int(bands[b, 2]), N)
            if w_low < N:
                # Apply band-specific alpha
                S[w_low:w_high] = _quantize(X[w_low:w_high], alpha[b])
                X_hat[w_low:w_high] = _dequantize(S[w_low:w_high], alpha[b])
    
    # Global gain is the first scale factor
    G = float(alpha[0])

    # DPCM encoding: sfc[b] = alpha[b] - alpha[b-1]
    sfc = np.zeros(num_bands)
    sfc[0] = alpha[0]
    for b in range(1, num_bands):
        sfc[b] = alpha[b] - alpha[b-1]  # Differential encoding
    
    return S, sfc, G


def _quantize(X, alpha):
    """Apply non-uniform quantization"""

    S = np.sign(X) * ((np.abs(X) * 2**(-alpha/4))**(3/4) + MAGIC_NUMBER).astype(int)

    return S

def _dequantize(S, alpha):
    """Reconstruct from quantized values"""

    X_hat = np.sign(S) * (np.abs(S)**(4/3)) * 2**(alpha/4)

    return X_hat


def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse quantizer - reconstructs MDCT coefficients from quantized symbols.
    
    Args:
        S: Quantized coefficients (1024,)
        sfc: Scale factors per band (69,) for long frames or (42, 8) for ESH
        G: Global gain scalar for long frames or (8,) for ESH
        frame_type: 'OLS', 'LSS', 'ESH', or 'LPS'
    
    Returns:
        frame_F: Reconstructed MDCT coefficients
            - (128, 8) for ESH
            - (1024,) for long frames
    """
    # Load band tables
    tables = _load_band_tables()
    
    if frame_type == "ESH":
        # Short frames: 8 subframes, use Table B.2.1.9.b
        bands = tables['B219b']
        frame_F = np.zeros((128, 8))
        
        # Process each subframe independently
        for i in range(8):
            # Extract subframe coefficients (128 coefficients per subframe)
            S_sub = S[i*128:(i+1)*128]
            frame_F[:, i] = _dequantize_subframe(S_sub, sfc[:, i], G[i], bands)
        
    else:  # OLS/LSS/LPS
        # Long frames: use Table B.2.1.9.a
        bands = tables['B219a']
        frame_F = _dequantize_subframe(S, sfc, G[0], bands)
    
    return frame_F


def _dequantize_subframe(S, sfc, G, bands):
    """
    Dequantize one subframe or full frame.
    
    Args:
        S: Quantized coefficients (1024,) or (128,)
        sfc: DPCM-encoded scale factors (69,) or (42,)
        G: Global gain (scalar)
        bands: Band table (B219a for long, B219b for short)
    
    Returns:
        X_hat: Reconstructed MDCT coefficients (1024,) or (128,)
    """
    N = len(S)
    num_bands = len(bands)
    
    # Step 1: Reconstruct alpha from DPCM-encoded sfc
    alpha = np.zeros(num_bands)
    alpha[0] = G  # First scale factor is the global gain
    
    for b in range(1, num_bands):
        alpha[b] = alpha[b-1] + sfc[b]  # Reverse DPCM
    
    # Step 2: Dequantize band by band
    X_hat = np.zeros(N)
    
    for b in range(num_bands):
        w_low = int(bands[b, 1])
        w_high = min(int(bands[b, 2]), N)

        if w_low < N and w_high > w_low:
            # Apply dequantization
            X_hat[w_low:w_high] = _dequantize(S[w_low:w_high], alpha[b])
    
    return X_hat


