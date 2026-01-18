# """
# Test script for AAC Level 1
# Creates a test audio file and encodes/decodes it
# """

# import numpy as np
# import soundfile as sf
# from aac_codec import demo_aac_1, aac_coder_1, i_aac_coder_1


# def create_test_audio(filename, duration=2.0, fs=48000):
#     """
#     Creates a test audio file

#     Parameters:
#     -----------
#     filename : str
#         Output filename
#     duration : float
#         Duration in seconds
#     fs : int
#         Sample rate
#     """
#     t = np.linspace(0, duration, int(fs * duration))

#     # Create complex signal:
#     # 1. Base frequency 440 Hz (A4)
#     # 2. Sinusoidal modulation
#     # 3. Transient in the middle

#     # Left channel: 440 Hz with modulation
#     left = 0.3 * np.sin(2 * np.pi * 440 * t)
#     left += 0.1 * np.sin(2 * np.pi * 880 * t)  # Harmonic
#     left *= (1 + 0.3 * np.sin(2 * np.pi * 2 * t))  # Modulation

#     # Right channel: 554 Hz (C#5) with different modulation
#     right = 0.3 * np.sin(2 * np.pi * 554 * t)
#     right += 0.1 * np.sin(2 * np.pi * 1108 * t)
#     right *= (1 + 0.3 * np.sin(2 * np.pi * 3 * t))

#     # Add transient in the middle (to trigger ESH frames)
#     transient_pos = int(len(t) / 2)
#     transient_width = int(fs * 0.01)  # 10ms
#     transient = np.zeros(len(t))
#     transient[transient_pos:transient_pos + transient_width] = 0.5

#     left += transient
#     right += transient

#     # Fade in/out
#     fade_len = int(fs * 0.1)  # 100ms
#     fade_in = np.linspace(0, 1, fade_len)
#     fade_out = np.linspace(1, 0, fade_len)

#     left[:fade_len] *= fade_in
#     left[-fade_len:] *= fade_out
#     right[:fade_len] *= fade_in
#     right[-fade_len:] *= fade_out

#     # Stereo array
#     audio = np.column_stack([left, right])

#     # Normalize
#     audio = audio / np.max(np.abs(audio)) * 0.9

#     # Save
#     sf.write(filename, audio, fs)
#     print(f"✓ Created test audio: {filename}")
#     print(f"  Duration: {duration}s")
#     print(f"  Sample rate: {fs} Hz")
#     print(f"  Channels: 2 (stereo)")


# def test_basic():
#     """Basic encoder/decoder test"""
#     print("\n" + "="*60)
#     print("TEST 1: Basic Encoding/Decoding")
#     print("="*60)

#     # Create test audio
#     create_test_audio("test_input.wav", duration=2.0)

#     # Demo (encoding + decoding + SNR)
#     SNR = demo_aac_1("test_input.wav", "test_output.wav")

#     return SNR


# def test_frame_analysis():
#     """Frame type analysis"""
#     print("\n" + "="*60)
#     print("TEST 2: Frame Type Analysis")
#     print("="*60)

#     # Encoding
#     print("\nEncoding...")
#     aac_seq = aac_coder_1("test_input.wav")

#     # Frame type analysis
#     print(f"\nTotal frames: {len(aac_seq)}")
#     print("\nFrame analysis:")

#     for i, frame in enumerate(aac_seq):
#         frame_type = frame["frame_type"]
#         win_type = frame["win_type"]
#         shape_l = frame["chl"]["frame_F"].shape
#         shape_r = frame["chr"]["frame_F"].shape

#         print(f"  Frame {i:2d}: {frame_type:3s} | {win_type:3s} | "
#               f"L:{shape_l} R:{shape_r}")

#     # Statistics
#     frame_types = [f["frame_type"] for f in aac_seq]
#     print("\nFrame type statistics:")
#     for ftype in ["OLS", "LSS", "ESH", "LPS"]:
#         count = frame_types.count(ftype)
#         if count > 0:
#             print(f"  {ftype}: {count} frames ({100*count/len(aac_seq):.1f}%)")


# def test_mdct_reconstruction():
#     """Test MDCT/IMDCT accuracy"""
#     print("\n" + "="*60)
#     print("TEST 3: MDCT/IMDCT Accuracy")
#     print("="*60)

#     from aac_utils import create_kbd_window, mdct, imdct

#     # Test for different sizes
#     for N in [256, 2048]:
#         # Create random signal
#         signal = np.random.randn(N)

#         # Window
#         window = create_kbd_window(N, alpha=6 if N == 2048 else 4)

#         # MDCT -> IMDCT
#         windowed = signal * window
#         coeffs = mdct(windowed, N)
#         reconstructed = imdct(coeffs, N) * window

#         # Error
#         error = np.max(np.abs(signal * window - reconstructed))

#         print(f"  N={N:4d}: Max error = {error:.2e}")


# def test_all():
#     """Run all tests"""
#     print("\n" + "#"*60)
#     print("# AAC LEVEL 1 - TEST SUITE")
#     print("#"*60)

#     # Test 1: Basic functionality
#     SNR = test_basic()

#     # Test 2: Frame analysis
#     test_frame_analysis()

#     # Test 3: MDCT accuracy
#     test_mdct_reconstruction()

#     print("\n" + "#"*60)
#     print("# ALL TESTS COMPLETED!")
#     print("#"*60)
#     print(f"\nFinal SNR: {SNR:.2f} dB")

#     if SNR > 80:
#         print("✓ EXCELLENT reconstruction quality!")
#     elif SNR > 50:
#         print("✓ GOOD reconstruction quality")
#     else:
#         print("⚠ Low SNR - check implementation")


# if __name__ == "__main__":
#     test_all()
