import numpy as np
from scipy.signal.windows import kaiser


# Global caches
_mdct_cache = {}
_i_mdct_cache = {}


# Create Kaiser-Bessel-Derived (KBD) window
def kbd_window(N, alpha=None):
    """
    Kaiser-Bessel-Derived (KBD) window for MDCT.
    
    Args:
        N: Window length in samples (2048 for long frames, 256 for short frames)
        alpha: Shape parameter (optional)
               - If None, uses AAC standard: 6 for N=2048, 4 for N=256
               - Larger alpha = sharper frequency selectivity
    
    Returns:
        window: NumPy array of shape (N,) containing KBD window values
    """
    
    if alpha is None:
        alpha = 6 if N == 2048 else 4  # AAC standard values

    # Generate Kaiser window
    w = kaiser(N // 2 + 1, np.pi * alpha)
    
    # Cumulative sum for KBD
    w_cumsum = np.cumsum(w)
    w_sum = w_cumsum[-1]   # Total sum
    
    # Left half of KBD window
    kbd_left = np.sqrt(w_cumsum[:-1] / w_sum)   # First N//2 elements
    
    # Right half of KBD window
    kbd_right = kbd_left[::-1]   # Reverse the left half
    
    return np.concatenate([kbd_left, kbd_right])


# Create sinusoid window
def sin_window(N):
    """
    Sinusoid window for MDCT.
    
    Args:
        N: Window length in samples (2048 for long frames, 256 for short frames)
    
    Returns:
        window: NumPy array of shape (N,) containing sine window values
    """
    
    n = np.arange(N)
    sin = np.sin((np.pi / N) * (n + 0.5))

    return sin


# Modified Discrete Cosine Transform (MDCT)
def mdct(x, N):
    """
    Modified Discrete Cosine Transform (MDCT).
    Transforms time-domain signal to frequency-domain coefficients.
    Uses matrix caching for computational efficiency.
    
    Args:
        x: Windowed time-domain signal, NumPy array of shape (N,)
        N: Transform size (2048 for long frames, 256 for short frames)
    
    Returns:
        X: MDCT coefficients, NumPy array of shape (N/2,)
           - 1024 coefficients for N=2048
           - 128 coefficients for N=256
    """
    
    global _mdct_cache
    
    # Check if cosine matrix for this N is already cached
    if N not in _mdct_cache:
    
        n0 = (N / 2 + 1) / 2
        
        # Create index arrays
        k = np.arange(N // 2).reshape(-1, 1)   # Column vector
        n = np.arange(N)                       # Row vector

        # Compute cosine matrix and cache it
        cos_matrix = np.cos((2 * np.pi / N) * (n + n0) * (k + 0.5))
        _mdct_cache[N] = cos_matrix
    
    # Retrieve cached matrix
    cos_matrix = _mdct_cache[N]
    
    # Compute MDCT
    X = 2 * np.dot(cos_matrix, x)
    
    return X


# Inverse Modified Discrete Cosine Transform (IMDCT)
def i_mdct(X, N):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT).
    Transforms frequency-domain coefficients back to time-domain signal.
    Uses matrix caching for computational efficiency.
    
    Args:
        X: MDCT coefficients, NumPy array of shape (N/2,)
        N: Transform size (2048 for long frames, 256 for short frames)
    
    Returns:
        x: Reconstructed time-domain signal, NumPy array of shape (N,)
           Still requires windowing and overlap-add for final reconstruction
    """
    
    global _i_mdct_cache
    
    # Check if cosine matrix is already cached
    if N not in _i_mdct_cache:

        n0 = (N / 2 + 1) / 2
        
        # Create index arrays
        n = np.arange(N).reshape(-1, 1)   # Column vector
        k = np.arange(N // 2)             # Row vector
        
        # Compute cosine matrix and cache it
        cos_matrix = np.cos((2 * np.pi / N) * (n + n0) * (k + 0.5))
        _i_mdct_cache[N] = cos_matrix
    
    # Retrieve cached matrix
    cos_matrix = _i_mdct_cache[N]
    
    # Compute IMDCT
    x = (2 / N) * np.dot(cos_matrix, X)
    
    return x
