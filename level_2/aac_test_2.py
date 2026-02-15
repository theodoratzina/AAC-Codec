from aac_codec_2 import demo_aac_2
import os


def main():
    """Run Level 2 AAC test with TNS"""
    
    # Path to the audio file
    input_file = os.path.join(os.path.dirname(__file__), '..', 'material', 'LicorDeCalandraca.wav')
    
    # Output file path
    output_file = os.path.join(os.path.dirname(__file__), 'output_level2.wav')
    
    print("="*60)
    print("AAC Level 2 Encoder/Decoder Test (with TNS)")
    print("="*60)
    
    # Run the demo
    SNR = demo_aac_2(input_file, output_file)
    
    # Summary
    print("\n" + "="*60)
    print("✓ Encoding/Decoding Complete!")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Final SNR: {SNR:.2f} dB")
    
    # Interpret the result
    if SNR > 100:
        print("Quality: Excellent (near-perfect reconstruction)")
    elif SNR > 60:
        print("Quality: Very Good (transparent)")
    elif SNR > 40:
        print("Quality: Good")
    else:
        print("Quality: Fair (some artifacts)")
    
    print("="*60)


if __name__ == "__main__":
    main()
