import os
import soundfile as sf
from aac_codec_3 import demo_aac_3, aac_coder_3


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


def main():
    """Main function to run the AAC Level 3 encoder/decoder demo."""

    # Path to the audio file
    input_file = os.path.join(os.path.dirname(__file__), '..', 'material', 'LicorDeCalandraca.wav')

    # Output file paths
    output_file = os.path.join(os.path.dirname(__file__), 'output_level3.wav')
    coded_file = os.path.join(os.path.dirname(__file__), 'aac_coded_level3.mat')

    print("="*60)
    print("AAC Level 3 Encoder/Decoder Test (Full Codec)")
    print("="*60)

    # Run the demo
    SNR, bitrate, compression = demo_aac_3(input_file, output_file, coded_file)

    # Get audio information
    audio, fs = sf.read(input_file)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    duration = len(audio) / fs

    # Analyze frame types and compression
    aac_seq_3 = aac_coder_3(input_file, coded_file)
    frame_counts = analyze_frame_types(aac_seq_3)

    # Summary
    print("\n" + "✓ Encoding/Decoding Complete!" + "\n")
    print(f"Input file: {os.path.basename(input_file)}")
    print(f"Output file: {os.path.basename(output_file)}")
    print(f"Coded file: {os.path.basename(coded_file)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {fs} Hz")
    print(f"Total frames: {len(aac_seq_3)}")
    print(f"Frame types: OLS={frame_counts['OLS']}, LSS={frame_counts['LSS']}, ESH={frame_counts['ESH']}, LPS={frame_counts['LPS']}")

    # Compression Statistics
    print("\n" + "Compression Statistics:" + "\n")
    original_bitrate = fs * 2 * 16  # Assuming 16-bit stereo
    print(f"Original bitrate: {original_bitrate/1000:.1f} kbps ({fs/1000:.0f} kHz, 16-bit stereo)")
    print(f"Compressed bitrate: {bitrate/1000:.1f} kbps")
    print(f"Compression ratio: {compression:.2f}x")

    # Audio Quality
    print("\n" + "Audio Quality:" + "\n")
    print(f"SNR: {SNR:.2f} dB")

    # Interpret the result (adjusted for lossy compression)
    if SNR > 60:
        print("Quality: Excellent (transparent)")
    elif SNR > 40:
        print("Quality: Very Good")
    elif SNR > 30:
        print("Quality: Good")
    elif SNR > 20:
        print("Quality: Fair")
    else:
        print("Quality: Poor")

    print("="*60)


if __name__ == "__main__":
    main()