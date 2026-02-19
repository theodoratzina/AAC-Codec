import numpy as np
import os
from scipy.io import loadmat

# Global cache for band tables
_BAND_TABLES = None


def _load_band_tables():
    """Load band tables once and cache them."""
    global _BAND_TABLES

    if _BAND_TABLES is None:
        table_path = os.path.join(os.path.dirname(__file__),  '..', 'material', 'TableB219.mat')
        table_data = loadmat(table_path)

        _BAND_TABLES = {
            'B219a': table_data['B219a'],  # Long frames
            'B219b': table_data['B219b']   # Short frames
        }
    return _BAND_TABLES


def tns(frame_F_in, frame_type):
    """
    Temporal Noise Shaping (TNS) for AAC encoder.
    
    Args:
        frame_F_in: MDCT coefficients for one channel
                    - Shape (1024,) for OLS/LSS/LPS
                    - Shape (128, 8) for ESH
        frame_type: Frame type string ('OLS', 'LSS', 'ESH', 'LPS')
    
    Returns:
        frame_F_out: TNS-filtered MDCT coefficients (same shape as input)
        tns_coeffs: Quantized LP coefficients
                    - Shape (4,) for long frames
                    - Shape (4, 8) for ESH (one set per subframe)
    """
    # Load tables (cached after first call)
    tables = _load_band_tables()

    if frame_type == "ESH":
        # Short frames: 8 subframes, use Table B.2.1.9.b
        bands = tables['B219b']
        frame_F_out = np.zeros((128, 8))
        tns_coeffs = np.zeros((4, 8))
        
        # Process each subframe
        for i in range(8):
            X = frame_F_in[:, i]
            frame_F_out[:, i], tns_coeffs[:, i] = _process_tns(X, bands)

    else:  # OLS/LSS/LPS
        # Long frames: use Table B.2.1.9.a
        bands = tables['B219a']
        X = frame_F_in
        frame_F_out, tns_coeffs = _process_tns(X, bands)
    
    return frame_F_out, tns_coeffs


def _process_tns(X, bands):
    """
    Apply Temporal Noise Shaping to a single window.
    
    Applies 4th-order linear prediction in frequency domain to reduce
    pre-echo artifacts around transients.
    
    Args:
        X: MDCT coefficients, shape (1024,) or (128,)
        bands: Band table (B219a for long, B219b for short)
    
    Returns:
        X_filtered: TNS-filtered coefficients (or X unchanged if failed)
        a_quantized: Quantized LP coefficients [4,] (or zeros if failed)
    """
    N = len(X)
    num_bands = len(bands)
    
    # Step 1: Normalize by band energy
    # Compute energy per band
    P = np.zeros(num_bands)
    for j in range(num_bands):
        w_low = int(bands[j, 1])
        w_high = min(int(bands[j, 2]), N)
        if w_low < N:
            P[j] = np.sum(X[w_low:w_high + 1] ** 2)
    
    # Create Sw array for normalization
    Sw = np.zeros(N)
    for j in range(num_bands):
        w_low = int(bands[j, 1])
        w_high = min(int(bands[j, 2]), N)
        if w_low < N:
            Sw[w_low:w_high + 1] = np.sqrt(P[j]) if P[j] > 0 else 1.0
    
    # Smooth Sw to avoid sharp transitions
    for k in range(N-2, -1, -1):
        Sw[k] = (Sw[k] + Sw[k+1]) / 2.0
    for k in range(1, N):
        Sw[k] = (Sw[k] + Sw[k-1]) / 2.0
    
    # Normalize
    Xw = X / np.where(Sw > 1e-10, Sw, 1.0)
    
    # Step 2: Compute LP coefficients (autocorrelation method)
    p = 4
    r = np.array([np.sum(Xw[lag:] * Xw[:-lag]) if lag > 0 else np.sum(Xw**2) 
                  for lag in range(p+1)])
    
    R = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
    
    try:
        a = np.linalg.solve(R, r[1:p+1])
    except np.linalg.LinAlgError:
        return X.copy(), np.zeros(4)
    
    # Step 3: Quantize
    a_quantized = np.clip(np.round(a / 0.1) * 0.1, -0.75, 0.75)
    
    # Step 4: Check stability
    # Inverse filter is stable if all poles are inside unit circle |z| < 1
    coeffs = np.concatenate([-a_quantized[::-1], [1]])
    roots = np.polynomial.polynomial.Polynomial(coeffs).roots()
    if not np.all(np.abs(roots) < 1.0):
        return X.copy(), np.zeros(4)
    
    # Step 5: Apply FIR filter
    X_filtered = np.zeros(N)
    for k in range(N):
        X_filtered[k] = X[k]
        for l in range(1, min(k+1, p+1)):
            X_filtered[k] -= a_quantized[l-1] * X[k-l]
    
    return X_filtered, a_quantized


def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Inverse Temporal Noise Shaping (TNS) for AAC decoder.
    
    Args:
        frame_F_in: TNS-filtered MDCT coefficients
                    - Shape (1024,) for OLS/LSS/LPS
                    - Shape (128, 8) for ESH
        frame_type: Frame type string ('OLS', 'LSS', 'ESH', 'LPS')
        tns_coeffs: Quantized LP coefficients from encoder
                    - Shape (4,) for long frames
                    - Shape (4, 8) for ESH
    
    Returns:
        frame_F_out: Reconstructed MDCT coefficients (same shape as input)
    """
    if frame_type == "ESH":
        # Short frames: 8 subframes, use Table B.2.1.9.b
        frame_F_out = np.zeros((128, 8))

        # Process each subframe independently
        for i in range(8):
            frame_F_out[:, i] = _inverse_tns(frame_F_in[:, i], tns_coeffs[:, i])

    else:  # OLS/LSS/LPS
        # Long frames: use Table B.2.1.9.a
        frame_F_out = _inverse_tns(frame_F_in, tns_coeffs)
    
    return frame_F_out


def _inverse_tns(X, a):
    """
    Apply inverse TNS filter (IIR).
    
    H_inv(z) = 1 / (1 - a1*z^-1 - a2*z^-2 - a3*z^-3 - a4*z^-4)
    
    Args:
        X: Filtered coefficients, shape (N,)
        a: LP coefficients, shape (4,)
    
    Returns:
        X_reconstructed: Original coefficients
    """
    N = len(X)
    p = len(a)
    X_reconstructed = np.zeros(N)
    
    # Apply the inverse filter using the quantized coefficients
    for k in range(N):
        X_reconstructed[k] = X[k]
        for l in range(1, min(k+1, p+1)):
            X_reconstructed[k] += a[l-1] * X_reconstructed[k-l]
    
    return X_reconstructed
