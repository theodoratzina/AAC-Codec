import numpy as np
import soundfile as sf
import time
from aac_ssc import SSC
from aac_filterbank import filter_bank, i_filter_bank
from aac_tns import tns, i_tns


def aac_coder_2(filename_in):
    """
    Level 2 AAC encoder with TNS.
    
    Args:
        filename_in: Path to input WAV file (48 kHz stereo)
    
    Returns:
        aac_seq_2: List of K dictionaries, one per frame, containing:
                    - 'frame_type': 'OLS', 'LSS', 'ESH', or 'LPS'
                    - 'win_type': 'KBD' or 'SIN'
                    - 'chl': dict with 'frame_F' (MDCT after TNS) and 'tns_coeffs'
                    - 'chr': dict with 'frame_F' (MDCT after TNS) and 'tns_coeffs'
    """
    # Read audio file
    audio, fs = sf.read(filename_in)
    
    # Convert mono to stereo if needed
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    # Verify sampling rate and channels
    assert fs == 48000, f"Sample rate must be 48kHz, got {fs}Hz"
    assert audio.shape[1] == 2, f"Audio must be stereo, got {audio.shape[1]} channels"
    
    # AAC parameters
    win_type = "KBD"
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Pad with 1024 zeros at beginning and end for proper MDCT overlap
    audio = np.pad(audio, ((hop_size, hop_size), (0, 0)), mode='constant')
    
    # Calculate number of frames (K)
    num_samples = len(audio)
    K = (num_samples - frame_size) // hop_size + 1
    
    # Pad if needed for last frame
    min_length = (K - 1) * hop_size + frame_size
    if num_samples < min_length:
        audio = np.pad(audio, ((0, min_length - num_samples), (0, 0)), mode='constant')
    
    # Add one extra frame for SSC lookahead
    audio = np.pad(audio, ((0, frame_size), (0, 0)), mode='constant')
    
    # Encoded sequence list
    aac_seq_2 = []
    prev_frame_type = "OLS"
    
    # Encode each frame
    for i in range(K):
        # Extract current frame (with 50% overlap)
        start = i * hop_size
        frame_T = audio[start:start + frame_size, :]
        
        # Extract next frame for SSC
        next_start = (i + 1) * hop_size
        next_frame_T = audio[next_start:next_start + frame_size, :]
        
        # Sequence Segmentation Control
        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        
        # Filter Bank (MDCT)
        frame_F = filter_bank(frame_T, frame_type, win_type)
        
        # Apply TNS to each channel
        if frame_type == "ESH":
            # ESH: frame_F has shape (128, 8, 2)
            frame_F_left_tns, tns_coeffs_left = tns(frame_F[:, :, 0], frame_type)
            frame_F_right_tns, tns_coeffs_right = tns(frame_F[:, :, 1], frame_type)
            
            # Store encoded frame
            aac_seq_2.append({
                'frame_type': frame_type,
                'win_type': win_type,
                'chl': {'frame_F': frame_F_left_tns,
                        'tns_coeffs': tns_coeffs_left},
                'chr': {'frame_F': frame_F_right_tns,
                        'tns_coeffs': tns_coeffs_right}
            })

        else:
            # OLS/LSS/LPS: frame_F has shape (1024, 2)
            frame_F_left_tns, tns_coeffs_left = tns(frame_F[:, 0], frame_type)
            frame_F_right_tns, tns_coeffs_right = tns(frame_F[:, 1], frame_type)
            
            # Store encoded frame
            aac_seq_2.append({
                'frame_type': frame_type,
                'win_type': win_type,
                'chl': {'frame_F': frame_F_left_tns,
                        'tns_coeffs': tns_coeffs_left},
                'chr': {'frame_F': frame_F_right_tns,
                        'tns_coeffs': tns_coeffs_right}
            })
        
        # Update previous frame type
        prev_frame_type = frame_type
    
    return aac_seq_2


def i_aac_coder_2(aac_seq_2, filename_out):
    """
    Level 2 AAC decoder with inverse TNS.
    
    Args:
        aac_seq_2: Encoded sequence from aac_coder_2
        filename_out: Output WAV file path
    
    Returns:
        x: Decoded audio signal (numpy array, shape: (num_samples, 2))
    """
    # Get number of frames
    K = len(aac_seq_2)
    
    # AAC parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Calculate output length
    output_length = (K - 1) * hop_size + frame_size
    
    # Initialize output buffer
    x = np.zeros((output_length, 2))
    
    # Decode each frame
    for i in range(K):
        # Extract frame info
        frame_type = aac_seq_2[i]['frame_type']
        win_type = aac_seq_2[i]['win_type']
        
        # Get TNS coefficients and filtered MDCT
        if frame_type == "ESH":
            # ESH: Reconstruct (128, 8, 2)
            frame_F_left = aac_seq_2[i]['chl']['frame_F']      # (128, 8)
            frame_F_right = aac_seq_2[i]['chr']['frame_F']     # (128, 8)
            tns_coeffs_left = aac_seq_2[i]['chl']['tns_coeffs']   # (4, 8)
            tns_coeffs_right = aac_seq_2[i]['chr']['tns_coeffs']  # (4, 8)
            
            # Apply inverse TNS
            frame_F_left_orig = i_tns(frame_F_left, frame_type, tns_coeffs_left)
            frame_F_right_orig = i_tns(frame_F_right, frame_type, tns_coeffs_right)
            
            # Stack to create (128, 8, 2)
            frame_F = np.stack([frame_F_left_orig, frame_F_right_orig], axis=2)
            
        else:
            # OLS/LSS/LPS: Reconstruct (1024, 2)
            frame_F_left = aac_seq_2[i]['chl']['frame_F']      # (1024,)
            frame_F_right = aac_seq_2[i]['chr']['frame_F']     # (1024,)
            tns_coeffs_left = aac_seq_2[i]['chl']['tns_coeffs']   # (4,)
            tns_coeffs_right = aac_seq_2[i]['chr']['tns_coeffs']  # (4,)
            
            # Apply inverse TNS
            frame_F_left_orig = i_tns(frame_F_left, frame_type, tns_coeffs_left)
            frame_F_right_orig = i_tns(frame_F_right, frame_type, tns_coeffs_right)
            
            # Stack to create (1024, 2)
            frame_F = np.column_stack([frame_F_left_orig, frame_F_right_orig])
        
        # Apply inverse filterbank (IMDCT)
        frame_T = i_filter_bank(frame_F, frame_type, win_type)
        
        # Overlap-add into output buffer
        start = i * hop_size
        end = start + frame_size
        x[start:end, :] += frame_T
    
    # Remove the 1024 samples of padding from start and end
    x = x[hop_size:-hop_size, :]
    
    # Write to WAV file (48 kHz stereo)
    sf.write(filename_out, x, 48000)
    
    return x


def demo_aac_2(filename_in, filename_out):
    """
    Demonstrates Level 2 AAC encoder/decoder with TNS and computes SNR.
    
    Args:
        filename_in: Input WAV file (48 kHz stereo)
        filename_out: Output WAV file (48 kHz stereo)
    
    Returns:
        SNR: Signal-to-Noise Ratio in dB
    """
    # Load original audio
    original, fs = sf.read(filename_in)
    
    # Ensure 2D shape for comparison (mono or stereo)
    if original.ndim == 1:
        original = original.reshape(-1, 1)
    
    # Encode
    print("Encoding...")
    start_time = time.time()
    aac_seq_2 = aac_coder_2(filename_in)
    encode_time = time.time() - start_time
    print(f"Encoded {len(aac_seq_2)} frames in {encode_time:.2f} seconds")
    
    # Decode
    print("Decoding...")
    start_time = time.time()
    decoded = i_aac_coder_2(aac_seq_2, filename_out)
    decode_time = time.time() - start_time
    print(f"Decoded {len(aac_seq_2)} frames in {decode_time:.2f} seconds")
    
    # If original was mono, compare only first channel
    if original.shape[1] == 1 and decoded.shape[1] == 2:
        decoded = decoded[:, :1]
    
    # Match lengths
    min_length = min(len(original), len(decoded))
    original = original[:min_length, :]
    decoded = decoded[:min_length, :]
    
    # Calculate SNR
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - decoded) ** 2)
    
    if noise_power == 0:
        SNR = float('inf')
    else:
        SNR = 10 * np.log10(signal_power / noise_power)
    
    return SNR
