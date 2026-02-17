import os
import soundfile as sf
from aac_codec_1 import demo_aac_1, aac_coder_1


def analyze_frame_types(aac_seq_1):
    """Analyze frame type distribution."""

    frame_types = [frame['frame_type'] for frame in aac_seq_1]
    counts = {
        'OLS': frame_types.count('OLS'),
        'LSS': frame_types.count('LSS'),
        'ESH': frame_types.count('ESH'),
        'LPS': frame_types.count('LPS')
    }
    return counts


def main():
    """Main function to run the AAC Level 1 encoder/decoder demo."""

    # Path to the audio file
    input_file = os.path.join(os.path.dirname(__file__), '..', 'material', 'LicorDeCalandraca.wav')
    
    # Output file path
    output_file = os.path.join(os.path.dirname(__file__), 'output_level1.wav')
    
    print("="*60)
    print("AAC Level 1 Encoder/Decoder Test")
    print("="*60)
    
    # Run the demo
    SNR = demo_aac_1(input_file, output_file)

    # Get audio information
    audio, fs = sf.read(input_file)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    duration = len(audio) / fs

    # Analyze frame types
    aac_seq_1 = aac_coder_1(input_file)
    frame_counts = analyze_frame_types(aac_seq_1)
    
    # Summary
    print("\n" + "✓ Encoding/Decoding Complete!" + "\n")
    print(f"Input file:   {os.path.basename(input_file)}")
    print(f"Output file:  {os.path.basename(output_file)}")
    print(f"Duration:     {duration:.2f}s")
    print(f"Sample rate:  {fs} Hz")
    print(f"Total frames: {len(aac_seq_1)}")
    print(f"Frame types:  OLS={frame_counts['OLS']}, LSS={frame_counts['LSS']}, ESH={frame_counts['ESH']}, LPS={frame_counts['LPS']}")
    
    # Audio Quality
    print("\n" + "Audio Quality:" + "\n")
    print(f"SNR: {SNR:.2f} dB")
    
    # Interpret the result
    if SNR > 100:
        print("Quality:      Excellent (near-perfect reconstruction)")
    elif SNR > 60:
        print("Quality:      Very Good (transparent)")
    elif SNR > 40:
        print("Quality:      Good")
    else:
        print("Quality:      Fair (some artifacts)")
    
    print("="*60)


if __name__ == "__main__":
    main()
