# """
# AAC Codec - Level 1
# Includes: Encoder, Decoder, Demo
# """

# import numpy as np
# import soundfile as sf
# from aac_ssc_filterbank import SSC, filter_bank, i_filter_bank


# def aac_coder_1(filename_in):
#     """
#     AAC Encoder Level 1

#     Encodes a WAV file using:
#     - Sequence Segmentation Control (SSC)
#     - Filter Bank (MDCT)

#     Parameters:
#     -----------
#     filename_in : str
#         Input WAV filename (stereo, 48kHz)

#     Returns:
#     --------
#     aac_seq_1 : list
#         List of encoded frames. Each element contains:
#         - frame_type: "OLS", "LSS", "ESH", or "LPS"
#         - win_type: "KBD" or "SIN"
#         - chl["frame_F"]: MDCT coefficients for left channel
#         - chr["frame_F"]: MDCT coefficients for right channel
#     """

#     # Read audio file
#     audio, fs = sf.read(filename_in)

#     # Convert to stereo if mono
#     if len(audio.shape) == 1:
#         audio = np.column_stack([audio, audio])

#     # Checks
#     assert audio.shape[1] == 2, "Audio must be stereo (2 channels)"
#     assert fs == 48000, f"Sample rate must be 48kHz, got {fs}Hz"

#     # Frame parameters
#     frame_size = 2048
#     hop_size = 1024  # 50% overlap

#     num_samples = len(audio)
#     num_frames = (num_samples - frame_size) // hop_size + 1

#     # Padding if file is too short
#     if num_samples < frame_size:
#         audio = np.pad(audio, ((0, frame_size - num_samples), (0, 0)), mode='constant')
#         num_frames = 1

#     aac_seq_1 = []
#     win_type = "KBD"  # Use KBD windows
#     prev_frame_type = "OLS"  # Initialization

#     for i in range(num_frames):
#         # Extract current frame
#         start = i * hop_size
#         end = start + frame_size

#         if end > len(audio):
#             # Padding for last frame
#             frame_T = np.pad(audio[start:], ((0, end - len(audio)), (0, 0)), mode='constant')
#         else:
#             frame_T = audio[start:end]

#         # Extract next frame for SSC
#         next_start = (i + 1) * hop_size
#         next_end = next_start + frame_size

#         if next_end > len(audio):
#             # Padding or use current if no next frame exists
#             if next_start < len(audio):
#                 next_frame_T = np.pad(audio[next_start:], 
#                                       ((0, max(0, next_end - len(audio))), (0, 0)), 
#                                       mode='constant')
#             else:
#                 next_frame_T = frame_T  # Use current at the end
#         else:
#             next_frame_T = audio[next_start:next_end]

#         # Sequence Segmentation Control
#         frame_type = SSC(frame_T, next_frame_T, prev_frame_type)

#         # Filter Bank (MDCT)
#         frame_F = filter_bank(frame_T, frame_type, win_type)

#         # Store frame
#         # Reshape for consistency: (128, 8) for ESH, (1024, 1) for others
#         frame_data = {
#             "frame_type": frame_type,
#             "win_type": win_type,
#             "chl": {
#                 "frame_F": frame_F[:, 0].reshape(-1, 8 if frame_type == "ESH" else 1)
#             },
#             "chr": {
#                 "frame_F": frame_F[:, 1].reshape(-1, 8 if frame_type == "ESH" else 1)
#             }
#         }

#         aac_seq_1.append(frame_data)
#         prev_frame_type = frame_type

#     return aac_seq_1


# def i_aac_coder_1(aac_seq_1, filename_out):
#     """
#     AAC Decoder Level 1

#     Decodes a sequence of frames and saves to WAV.

#     Parameters:
#     -----------
#     aac_seq_1 : list
#         Encoded sequence from aac_coder_1
#     filename_out : str
#         Output WAV filename

#     Returns:
#     --------
#     x : ndarray
#         Decoded signal (stereo)
#     """

#     num_frames = len(aac_seq_1)
#     hop_size = 1024

#     # Compute output size
#     # Each frame contributes hop_size samples + last frame
#     output_length = hop_size * num_frames + hop_size
#     reconstructed = np.zeros((output_length, 2))

#     for i, frame_data in enumerate(aac_seq_1):
#         frame_type = frame_data["frame_type"]
#         win_type = frame_data["win_type"]

#         # Reconstruct frame_F from both channels
#         frame_F_left = frame_data["chl"]["frame_F"].flatten()
#         frame_F_right = frame_data["chr"]["frame_F"].flatten()
#         frame_F = np.column_stack([frame_F_left, frame_F_right])

#         # Inverse Filter Bank (IMDCT)
#         frame_T = i_filter_bank(frame_F, frame_type, win_type)

#         # Overlap-add
#         start = i * hop_size
#         end = start + 2048

#         if end > len(reconstructed):
#             # Extend buffer if needed
#             reconstructed = np.pad(reconstructed, 
#                                   ((0, end - len(reconstructed)), (0, 0)), 
#                                   mode='constant')

#         reconstructed[start:end] += frame_T

#     # Save to WAV file
#     sf.write(filename_out, reconstructed, 48000)

#     return reconstructed


# def demo_aac_1(filename_in, filename_out):
#     """
#     Demo function for Level 1

#     Encodes and decodes a WAV file,
#     computes SNR and displays statistics.

#     Parameters:
#     -----------
#     filename_in : str
#         Input WAV file
#     filename_out : str
#         Output WAV file

#     Returns:
#     --------
#     SNR : float
#         Signal-to-Noise Ratio in dB
#     """

#     print("=" * 60)
#     print("AAC LEVEL 1 - DEMO")
#     print("=" * 60)

#     # Encoding
#     print(f"\n[1/3] Encoding: {filename_in}")
#     aac_seq_1 = aac_coder_1(filename_in)
#     print(f"      ✓ Encoded {len(aac_seq_1)} frames")

#     # Frame type statistics
#     frame_types = [frame["frame_type"] for frame in aac_seq_1]
#     type_counts = {
#         "OLS": frame_types.count("OLS"),
#         "LSS": frame_types.count("LSS"),
#         "ESH": frame_types.count("ESH"),
#         "LPS": frame_types.count("LPS")
#     }
#     print(f"      Frame types: OLS={type_counts['OLS']}, LSS={type_counts['LSS']}, "
#           f"ESH={type_counts['ESH']}, LPS={type_counts['LPS']}")

#     # Decoding
#     print(f"\n[2/3] Decoding: {filename_out}")
#     reconstructed = i_aac_coder_1(aac_seq_1, filename_out)
#     print(f"      ✓ Decoded {len(reconstructed)} samples")

#     # SNR Calculation
#     print(f"\n[3/3] Computing SNR...")
#     original, fs = sf.read(filename_in)

#     # Convert to stereo if mono
#     if len(original.shape) == 1:
#         original = np.column_stack([original, original])

#     # Same length for comparison
#     min_len = min(len(original), len(reconstructed))
#     original = original[:min_len]
#     reconstructed = reconstructed[:min_len]

#     # SNR computation
#     signal_power = np.mean(original ** 2)
#     noise_power = np.mean((original - reconstructed) ** 2)

#     if noise_power > 0:
#         SNR = 10 * np.log10(signal_power / noise_power)
#     else:
#         SNR = np.inf

#     print(f"      ✓ SNR: {SNR:.2f} dB")

#     print("\n" + "=" * 60)
#     print("COMPLETED!")
#     print("=" * 60)

#     return SNR


# if __name__ == "__main__":
#     # Usage example
#     import sys

#     if len(sys.argv) == 3:
#         input_file = sys.argv[1]
#         output_file = sys.argv[2]
#         demo_aac_1(input_file, output_file)
#     else:
#         print("Usage: python aac_codec.py <input.wav> <output.wav>")
#         print("\nExample:")
#         print("  python aac_codec.py input.wav output.wav")
