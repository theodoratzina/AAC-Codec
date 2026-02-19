import os
import soundfile as sf
from pathlib import Path
from aac_codec_1 import demo_aac_1, aac_coder_1


def make_clickable(file_path, display_text=None):
    """Wraps text in an ANSI OSC 8 escape sequence to make it a clickable file link."""
    
    if display_text is None:
        display_text = os.path.basename(file_path)
        
    # Safely convert Windows backslashes to standard file:/// URLs
    file_uri = Path(os.path.abspath(file_path)).as_uri()
    
    return f"\033]8;;{file_uri}\a{display_text}\033]8;;\a"


def main():
    """Main function to run the AAC Level 1 encoder/decoder demo."""

    # Path to the audio file
    input_file = os.path.join(os.path.dirname(__file__), '..', 'material', 'LicorDeCalandraca.wav')
    
    # Output file path
    output_file = os.path.join(os.path.dirname(__file__), 'output_level1.wav')
    
    print("="*60)
    print("AAC Level 1 Encoder/Decoder")
    print("="*60)
    
    # Run the demo
    SNR = demo_aac_1(input_file, output_file)

    # Get audio information
    audio, fs = sf.read(input_file)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    duration = len(audio) / fs
    
    # Summary
    print("\n" + "✓ Encoding/Decoding Complete!" + "\n")
    print(f"Input file:   {make_clickable(input_file)}")
    print(f"Output file:  {make_clickable(output_file)}")
    print(f"Duration:     {duration:.2f}s")
    print(f"Sample rate:  {fs} Hz")
    
    # Audio Quality
    print("\n" + "Audio Quality:")
    print("-" * 60)
    print(f"SNR: {SNR:.2f} dB" + "\n")
    
    # Interpret the result
    if SNR > 100:
        print("Quality: Excellent")
    elif SNR > 60:
        print("Quality: Very Good")
    elif SNR > 40:
        print("Quality: Good")
    else:
        print("Quality: Fair")
    
    print("="*60)


if __name__ == "__main__":
    main()
