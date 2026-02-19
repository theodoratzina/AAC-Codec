import numpy as np
import soundfile as sf
import scipy.io as sio
import time
from aac_ssc import SSC
from aac_filterbank import filter_bank, i_filter_bank
from aac_tns import tns, i_tns, _load_band_tables
from aac_psycho import psycho
from aac_quantizer import aac_quantizer, i_aac_quantizer
from huff_utils import load_LUT, encode_huff, decode_huff


def aac_coder_3(filename_in, filename_aac_coded):
    """
    Level 3 AAC encoder with Psychoacoustic model, Quantization, and Huffman coding.

    Args:
        filename_in: Path to input WAV file (48 kHz stereo)
        filename_aac_coded: Path to output .mat file for encoded data

    Returns:
        aac_seq_3: List of K dictionaries, one per frame, containing:
                    - 'frame_type': 'OLS', 'LSS', 'ESH', or 'LPS'
                    - 'win_type': 'KBD' or 'SIN'
                    - 'chl': dict with 'tns_coeffs', 'T', 'G', 'sfc', 'stream', 'codebook'
                    - 'chr': dict with 'tns_coeffs', 'T', 'G', 'sfc', 'stream', 'codebook'
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
    win_type = "KBD"  # Window type (can be "KBD" or "SIN")
    frame_size = 2048
    hop_size = 1024  # 50% overlap

    # Load Huffman codebooks
    huff_LUT_list = load_LUT()

    # Load band tables
    tables = _load_band_tables()

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
    aac_seq_3 = []
    prev_frame_type = "OLS"

    # Frame history for psychoacoustic model (per channel)
    frame_T_prev_1 = [None, None]  # Previous frame for each channel
    frame_T_prev_2 = [None, None]  # 2 frames back for each channel

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

        # FilterBank
        frame_F = filter_bank(frame_T, frame_type, win_type)

        # Process each channel
        channel_data = []
        for ch in range(2):
            # Extract channel-specific MDCT coefficients and band table
            if frame_type == "ESH":
                frame_F_ch = frame_F[:, :, ch]  # (128, 8)
                bands = tables['B219b']
            else:
                frame_F_ch = frame_F[:, ch]  # (1024,)
                bands = tables['B219a']

            # Psychoacoustic Model
            SMR = psycho(frame_T[:, ch], frame_type, 
                        frame_T_prev_1[ch], frame_T_prev_2[ch])

            # Update frame history
            frame_T_prev_2[ch] = frame_T_prev_1[ch]
            frame_T_prev_1[ch] = frame_T[:, ch].copy()

            # Temporal Noise Shaping (TNS)
            frame_F_tns, tns_coeffs = tns(frame_F_ch, frame_type)

            # Compute masking thresholds T
            T = _compute_masking_thresholds(frame_F_tns, SMR, bands, frame_type)

            # Quantization
            S, sfc, G = aac_quantizer(frame_F_tns, frame_type, SMR)

            # Huffman encoding for MDCT coefficients
            stream, codebook = encode_huff(S, huff_LUT_list)

            # Huffman encoding for scale factors (always use codebook 11)
            if frame_type == "ESH":
                # ESH: Encode each subframe's scale factors separately
                sfc_encoded = []
                sfc_codebook = []
                for sub in range(8):
                    sfc_sub, codebook_sub = encode_huff(sfc[1:, sub], huff_LUT_list)
                    sfc_encoded.append(sfc_sub)
                    sfc_codebook.append(codebook_sub)

            else:
                # Long frames: Single scale factor vector
                sfc_encoded, sfc_codebook = encode_huff(sfc[1:], huff_LUT_list)

            # Store channel data
            channel_data.append({
                'tns_coeffs': tns_coeffs,
                'T': T,
                'G': G,
                'sfc': sfc_encoded,
                'sfc_codebook': sfc_codebook,
                'stream': stream,
                'codebook': codebook
            })

        # Store encoded frame
        aac_seq_3.append({
            'frame_type': frame_type,
            'win_type': win_type,
            'chl': channel_data[0],
            'chr': channel_data[1]
        })

        # Update previous frame type
        prev_frame_type = frame_type

    # Save to .mat file
    sio.savemat(filename_aac_coded, {'aac_seq_3': aac_seq_3})

    return aac_seq_3


def _compute_masking_thresholds(frame_F_tns, SMR, bands, frame_type):
    """
    Compute masking thresholds T from MDCT coefficients and SMR.

    Args:
        frame_F_tns: MDCT after TNS (1024,) for long frames or (128, 8) for ESH
        SMR: Signal-to-Mask Ratio in dB (69,) for long frames or (42, 8) for ESH
        bands: Band table (B219a for long, B219b for short)
        frame_type: 'OLS', 'LSS', 'ESH', or 'LPS'

    Returns:
        T: Masking thresholds (Nb,)
            - (69,) for long frames
            - (42,) for ESH (averaged across 8 subframes)
    """
    num_bands = len(bands)

    if frame_type == "ESH":
        # ESH: Compute T for each subframe, then average
        T_subframes = np.zeros((42, 8))

        for sub in range(8):
            N = len(frame_F_tns[:, sub])

            # P[b] = sum of squared MDCT coefficients in band b
            P = np.zeros(num_bands)
            for b in range(num_bands):
                w_low = int(bands[b, 1])
                w_high = min(int(bands[b, 2]), N)
                if w_low < N and w_high > w_low:
                    P[b] = np.sum(frame_F_tns[w_low:w_high, sub] ** 2)

            # Compute energy thresholds T[b] from SMR (convert from dB)
            SMR_linear = 10 ** (SMR[:, sub] / 10.0)
            T_subframes[:, sub] = P / (SMR_linear + 1e-10)

        # Average across 8 subframes
        T = np.mean(T_subframes, axis=1)  # Shape: (42,)

    else:
        # Long frames
        N = len(frame_F_tns)

        # P[b] = sum of squared MDCT coefficients in band b
        P = np.zeros(num_bands)
        for b in range(num_bands):
            w_low = int(bands[b, 1])
            w_high = min(int(bands[b, 2]), N)
            if w_low < N and w_high > w_low:
                P[b] = np.sum(frame_F_tns[w_low:w_high] ** 2)

        # Compute energy thresholds T[b] from SMR (convert from dB)
        SMR_linear = 10 ** (SMR / 10.0)
        T = P / (SMR_linear + 1e-10)  # Shape: (69,)

    return T


def i_aac_coder_3(aac_seq_3, filename_out):
    """
    Level 3 AAC decoder with inverse operations.

    Args:
        aac_seq_3: Encoded sequence from aac_coder_3
        filename_out: Output WAV file path

    Returns:
        x: Decoded audio signal (numpy array, shape: (num_samples, 2))
    """
    # Get number of frames
    K = len(aac_seq_3)

    # AAC parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap

    # Load Huffman codebooks
    huff_LUT_list = load_LUT()

    # Calculate output length
    output_length = (K - 1) * hop_size + frame_size

    # Initialize output buffer
    x = np.zeros((output_length, 2))

    # Decode each frame
    for i in range(K):
        # Extract frame info
        frame_type = aac_seq_3[i]['frame_type']
        win_type = aac_seq_3[i]['win_type']

        # Process each channel
        frame_F_channels = []
        for ch_name in ['chl', 'chr']:
            ch_data = aac_seq_3[i][ch_name]

            # Huffman decoding for MDCT coefficients
            codebook = ch_data['codebook']
            stream = ch_data['stream']
            S = np.array(decode_huff(stream, huff_LUT_list[codebook]))

            # Global gain
            G = ch_data['G']

            # Huffman decoding for scale factors
            if frame_type == "ESH":
                # ESH: Decode each subframe's scale factors
                sfc = np.zeros((42, 8))
                for sub in range(8):
                    sfc_codebook = ch_data['sfc_codebook'][sub]
                    sfc[0, sub] = G[sub]  # Insert global gain at index 0

                    if sfc_codebook != 0:
                        decoded = np.array(decode_huff(ch_data['sfc'][sub], huff_LUT_list[sfc_codebook]))
                        # Insert the 41 decoded differences starting from index 1
                        sfc[1:, sub] = decoded[:41]
            else:
                # Long frames: Single scale factor vector
                sfc = np.zeros(69)
                sfc_codebook = ch_data['sfc_codebook']
                
                # Handle G depending on whether it was saved as a scalar or a 1D array
                sfc[0] = G[0] if isinstance(G, (list, np.ndarray)) else G

                if sfc_codebook != 0:
                    decoded = np.array(decode_huff(ch_data['sfc'], huff_LUT_list[sfc_codebook]))
                    # Insert the 68 decoded differences starting from index 1
                    sfc[1:] = decoded[:68]

            # Inverse Quantization
            frame_F_dequantized = i_aac_quantizer(S, sfc, G, frame_type)

            # Inverse TNS
            tns_coeffs = ch_data['tns_coeffs']
            frame_F_ch = i_tns(frame_F_dequantized, frame_type, tns_coeffs)

            frame_F_channels.append(frame_F_ch)

        # Reconstruct stereo MDCT coefficients
        if frame_type == "ESH":
            # ESH: Stack to create (128, 8, 2)
            frame_F = np.stack([frame_F_channels[0], frame_F_channels[1]], axis=2)
        else:
            # Long frames: Stack to create (1024, 2)
            frame_F = np.column_stack([frame_F_channels[0], frame_F_channels[1]])

        # Apply inverse filterbank (IMDCT) - Called once for both channels
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


def demo_aac_3(filename_in, filename_out, filename_aac_coded):
    """
    Demonstrates Level 3 AAC encoder/decoder and computes metrics.

    Args:
        filename_in: Input WAV file (48 kHz stereo)
        filename_out: Output WAV file (48 kHz stereo)
        filename_aac_coded: Output .mat file for encoded data

    Returns:
        SNR: Signal-to-Noise Ratio in dB
        bitrate: Average bitrate in bits/sec
        compression: Compression ratio
    """
    # Load original audio
    original, fs = sf.read(filename_in)

     # Get actual bit depth from file info
    info = sf.info(filename_in)
    
    # Extract bit depth from subtype (e.g., 'PCM_16' → 16)
    if 'PCM_16' in info.subtype:
        bit_depth = 16
    elif 'PCM_24' in info.subtype:
        bit_depth = 24
    elif 'PCM_32' in info.subtype or 'FLOAT' in info.subtype:
        bit_depth = 32
    else:
        bit_depth = 16  # Default assumption
    
    # Ensure 2D shape for comparison (mono or stereo)
    if original.ndim == 1:
        original = original.reshape(-1, 1)

    # Encode
    print("Encoding...")
    start_time = time.time()
    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    encode_time = time.time() - start_time
    print(f"Encoded {len(aac_seq_3)} frames in {encode_time:.2f} seconds")

    # Calculate bitrate and compression
    total_bits = 0
    for frame in aac_seq_3:
        # Count bits from Huffman streams
        total_bits += len(frame['chl']['stream'])  # MDCT left
        total_bits += len(frame['chr']['stream'])  # MDCT right

        # Count bits from scale factors
        if isinstance(frame['chl']['sfc'], list):
            # ESH: Multiple subframes
            for sfc_sub in frame['chl']['sfc']:
                total_bits += len(sfc_sub)
            for sfc_sub in frame['chr']['sfc']:
                total_bits += len(sfc_sub)
        else:
            # Long frames
            total_bits += len(frame['chl']['sfc'])
            total_bits += len(frame['chr']['sfc'])

    duration = len(original) / fs
    bitrate = total_bits / duration

    # Original bitrate: 48000 Hz * 2 channels * bit_depth bits/sample
    original_bitrate = fs * 2 * bit_depth
    compression = original_bitrate / bitrate

    # Decode
    print("Decoding...")
    start_time = time.time()
    decoded = i_aac_coder_3(aac_seq_3, filename_out)
    decode_time = time.time() - start_time
    print(f"Decoded {len(aac_seq_3)} frames in {decode_time:.2f} seconds")

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

    print(f"SNR: {SNR:.2f} dB")
    print(f"Bitrate: {bitrate:.0f} bits/sec ({bitrate/1000:.1f} kbps)")
    print(f"Compression: {compression:.2f}x")

    return SNR, bitrate, compression