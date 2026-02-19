import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from aac_codec_3 import demo_aac_3


def analyze_frame_types(aac_seq_3):
    """Analyze frame type distribution."""

    frame_types = [frame['frame_type'] for frame in aac_seq_3]
    counts = {
        'OLS': frame_types.count('OLS'),
        'LSS': frame_types.count('LSS'),
        'ESH': frame_types.count('ESH'),
        'LPS': frame_types.count('LPS')
    }
    return counts


def make_clickable(file_path, display_text=None):
    """Wraps text in an ANSI OSC 8 escape sequence to make it a clickable file link."""
    
    if display_text is None:
        display_text = os.path.basename(file_path)
        
    # Safely convert Windows backslashes to standard file:/// URLs
    file_uri = Path(os.path.abspath(file_path)).as_uri()
    
    return f"\033]8;;{file_uri}\a{display_text}\033]8;;\a"


def plot_masking_threshold(frame_data, frame_index=50):
    """Plots the masking threshold T against the MDCT coefficients for a single random frame."""

    ch_data = frame_data[frame_index]['chl']
    T = ch_data['T']
    
    # NOTE: To plot this perfectly, would load TableB219a and expand T values to match 1024 frequency bins.
    
    plt.figure(figsize=(10, 5))
    plt.plot(10 * np.log10(T + 1e-10), color='red', label='Masking Threshold T(b)', drawstyle='steps-post')
    plt.title(f"Psychoacoustic Masking Threshold (Random Frame {frame_index})")
    plt.xlabel("Scalefactor Band")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    """Main function to run the AAC Level 3 encoder/decoder demo."""

    # Path to the audio file
    input_file = os.path.join(os.path.dirname(__file__), '..', 'material', 'LicorDeCalandraca.wav')

    # Output file paths
    output_file = os.path.join(os.path.dirname(__file__), 'output_level3.wav')
    coded_file = os.path.join(os.path.dirname(__file__), 'aac_coded_level3.mat')

    print("="*60)
    print("AAC Level 3 Encoder/Decoder (Full Codec)")
    print("="*60)

    # Run the demo
    SNR, bitrate, compression = demo_aac_3(input_file, output_file, coded_file)

    # Get audio information
    audio, fs = sf.read(input_file)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    duration = len(audio) / fs

    # Load the already-encoded data
    mat_data = loadmat(coded_file, simplify_cells=True)
    aac_seq_3 = mat_data['aac_seq_3']

    # Count frame types
    frame_counts = analyze_frame_types(aac_seq_3)

    # Calculate original bitrate (for reference)
    original_bitrate = fs * 2 * 16  # Assuming 16-bit stereo

    # Summary
    print("\n" + "✓ Encoding/Decoding Complete!" + "\n")
    print(f"Input file: {make_clickable(input_file)}")
    print(f"Output file: {make_clickable(output_file)}")
    print(f"Coded file: {make_clickable(coded_file)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {fs} Hz")
    print(f"Original bitrate: {original_bitrate:.0f} bits/sec ({original_bitrate/1000:.1f} kbps)")
    print(f"Total frames: {len(aac_seq_3)}")
    print(f"Frame types: OLS={frame_counts['OLS']}, LSS={frame_counts['LSS']}, ESH={frame_counts['ESH']}, LPS={frame_counts['LPS']}")

    # Compression Statistics and audio Quality
    print("\n" + "Compression Statistics and Audio Quality:")
    print("-"*60)
    print(f"SNR: {SNR:.2f} dB")
    print(f"Compressed Bitrate: {bitrate:.0f} bits/sec ({bitrate/1000:.1f} kbps)")
    print(f"Compression ratio: {compression:.2f}:1" + "\n")

    # Interpret the result (adjusted for lossy compression)
    if SNR > 60:
        print("Quality: Excellent")
    elif SNR > 40:
        print("Quality: Very Good")
    elif SNR > 30:
        print("Quality: Good")
    elif SNR > 20:
        print("Quality: Fair")
    else:
        print("Quality: Poor")

    print("="*60)

    plot_masking_threshold(aac_seq_3)


if __name__ == "__main__":
    main()