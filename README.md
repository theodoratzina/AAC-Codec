# 🎵 AAC Audio Codec

An end-to-end implementation of a simplified **Advanced Audio Coding (AAC)** encoder/decoder, developed for the course *Multimedia Systems* (2025–2026) at the Aristotle University of Thessaloniki, Department of Electrical and Computer Engineering.

The codec is built incrementally across **three levels**, each adding a new processing stage on top of the previous one.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Levels of Implementation](#levels-of-implementation)
  - [Level 1 — Sequence Segmentation Control & Filterbank](#level-1--sequence-segmentation-control--filterbank)
  - [Level 2 — Temporal Noise Shaping](#level-2--temporal-noise-shaping)
  - [Level 3 — Full Codec](#level-3--full-codec)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project implements a **simplified AAC codec** adhering to 3GPP TS 26.403 standards and to a simplified variant of the MPEG AAC psychoacoustic model. It operates on **stereo audio at 48 kHz** and uses **waveform compression.**

The codec achieves a compression ratio of ~**7.78:1** (1536 kbps → ~197 kbps) while masking quantization noise below the human hearing threshold.

> **Note:** The Mid/Side stereo stage and the Bit Reservoir are not implemented, as per the assignment specification.

---

## Architecture

### Encoder Pipeline

```
Input WAV → SSC → FilterBank → TNS → Psychoacoustic Model → Quantizer → Huffman → Output .mat file
```

### Decoder Pipeline

```
Input .mat file → iHuffman → iQuantizer → iTNS → iFilterbank → Output WAV
```

---

## Project Structure

```
.
├── level_1/                    # Level 1: SSC + Filterbank
│   ├── aac_utils.py
│   ├── aac_ssc.py
│   ├── aac_filterbank.py
│   ├── aac_codec_1.py
│   └── aac_test_1.py
│
├── level_2/                    # Level 2: Level 1 + TNS
│   ├── aac_utils.py
│   ├── aac_ssc.py
│   ├── aac_filterbank.py
│   ├── aac_tns.py
│   ├── aac_codec_2.py
│   └── aac_test_2.py
│
├── level_3/                    # Level 3: Full Codec
│   ├── aac_utils.py
│   ├── aac_ssc.py
│   ├── aac_filterbank.py   
│   ├── aac_tns.py
│   ├── aac_psycho.py
│   ├── aac_quantizer.py
│   ├── huff_utils.py
│   ├── aac_codec_3.py
│   └── aac_test_3.py
│
├── material/
│   ├── TableB219.mat           # Psychoacoustic band tables (48 kHz)
│   ├── huffCodebooks.mat       # AAC Huffman codebook LUTs
│   └── LicorDeCalandraca.wav   # Test audio file (stereo, 48 kHz)
│
└── report.pdf
```

---

## Levels of Implementation

### Level 1 — Sequence Segmentation Control & Filterbank

**New stages:** `SSC` - `FilterBank` / `iFilterBank`

The first level implements the fundamental building blocks of the codec:

- **Sequence Segmentation Control (SSC):** Classifies each 2048-sample frame into one of four types.
  | Code | Type | Description |
  |------|------|-------------|
  | `OLS` | `ONLY_LONG_SEQUENCE` | Spectrally stationary frames |
  | `ESH` | `EIGHT_SHORT_SEQUENCE` | Transient / rapidly changing frames |
  | `LSS` | `LONG_START_SEQUENCE` | Transition from OLS → ESH |
  | `LPS` | `LONG_STOP_SEQUENCE`  | Transition from ESH → OLS |

  Detection uses a high-pass filtered look-ahead at the next frame and computes *attack values* across 8 sub-regions of 128 samples.

- **Filterbank:** Applies MDCT (Modified Discrete Cosine Transform) with 50% frame overlap.
  - Long frames → 1 window of N=2048 → **1024 MDCT coefficients**
  - Short frames → 8 sub-windows of N=256 → **128×8 MDCT coefficients**
  - Supports **KBD** (Kaiser-Bessel-Derived, α=6/4) and **SIN** (sinusoid) windows
  - Asymmetric windows for LSS and LPS transition frames
  - Decoder reconstructs signal via IMDCT + **Overlap-Add**

---

### Level 2 — Temporal Noise Shaping

**New stages:** `TNS` / `iTNS`

Addresses pre-echo artifacts in transient frames by applying **Linear Prediction** in the frequency domain:

- **Temporal Temporal Noise Shaping**:
  
  - Normalizes MDCT coefficients per psychoacoustic band using energy-based smoothed weights
  - Solves the normal equations $\mathbf{R}\,\mathbf{a} = \mathbf{r}$ for an order p=4 LP filter
  - Quantizes LP coefficients with a **4-bit uniform quantizer**
  - Applies FIR filter $H_{TNS}(z) = 1 - a_1 z^{-1} - \ldots - a_p z^{-p}$ to MDCT coefficients
  - Includes **stability check**: verifies all poles of the inverse filter lie within the unit circle $|z| < 1$

---

### Level 3 — Full Codec

**New stages:** `Psychoacoustic Model` - `Quantizer` - `Huffman Coding`

The complete lossy compression pipeline:

- **Psychoacoustic Model**:
  - Computes the **Signal-to-Mask Ratio (SMR)** per frequency band using a Hann-windowed FFT
  - Models auditory masking via the MPEG **spreading function**, tonality index, and absolute threshold of hearing
  - Uses band tables `B219a` (long, 69 bands) and `B219b` (short, 42 bands) from `TableB219.mat`

- **Quantizer**:
  - Non-uniform 3/4-power law quantization $S = \text{sign}(X) \cdot \lfloor |X|^{3/4} \cdot 2^{-\alpha/4} + 0.4054 \rfloor$, guided by the SMR thresholds from the psychoacoustic model
  - Scale factors iteratively optimized per band and DPCM-encoded (max delta = 60)

- **Huffman Coding**: *(pre-built utility, not implemented from scratch)*
  - 11 standard AAC codebooks provided via `huffCodebooks.mat`
  - Automatic codebook selection based on maximum absolute coefficient value; escape coding for large values

---

## Results

| Level | SNR (dB) | Bitrate | Compression Ratio | Notes |
|-------|----------|---------|-------------------|-------|
| 1 | 253.99 | N/A | N/A | Lossless (no quantization) |
| 2 | 253.99 | N/A | N/A | Lossless (no quantization) |
| 3 | 10.07 | ~197.4 kbps | **7.78:1** | Lossy (full codec) |

Original uncompressed bitrate: **1536 kbps** (48 kHz × 2 ch × 16 bit)

---

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- soundfile
- matplotlib

Install dependencies:

```bash
pip install numpy scipy soundfile matplotlib
```

### Required Data Files

Make sure to place `material/` directory at the project root:

```
material/
├── TableB219.mat
├── huffCodebooks.mat
└── LicorDeCalandraca.wav
```

---

## Usage

### Run the Level 3 Full Codec Demo

```bash
cd level_3/
python aac_test_3.py
```

This will:
1. Encode the test WAV file using the full pipeline
2. Save the encoded data to `aac_coded_level3.mat`
3. Decode and save the output to `output_level3.wav`
4. Print SNR, bitrate, compression ratio, and frame statistics
5. Plot the psychoacoustic masking threshold for a sample frame

### Run individual level demos

```bash
cd level_1/ && python aac_test_1.py
cd level_2/ && python aac_test_2.py
```
