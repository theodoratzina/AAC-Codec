import numpy as np
from scipy.signal.windows import kaiser

# Global caches
_mdct_cache = {}
_i_mdct_cache = {}


# Create Kaiser-Bessel-Derived (KBD) window
def kbd_window(N, alpha):
    
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
    
    n = np.arange(N)
    sin = np.sin((np.pi / N) * (n + 0.5))

    return sin


# Modified Discrete Cosine Transform (MDCT)
def mdct(x, N):
    
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
    
    X = 2 * np.dot(cos_matrix, x)
    
    return X


# Inverse Modified Discrete Cosine Transform (IMDCT)
def i_mdct(X, N):
    
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
    
    x = (2 / N) * np.dot(cos_matrix, X)
    
    return x
