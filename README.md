# Audio Feature Extraction

This repository contains a Python script for extracting audio features using FFT and recognizing audio samples as "Yes" or "No" based on their features.

## How it works
### Loading and Preprocessing Audio Files:
Audio files are loaded, and the sampling frequency is determined. The librosa library is used to load the audio files.

### FFT Calculation:
The FFT (Fast Fourier Transform) of the audio files is calculated. FFT is the representation of a signal in the frequency domain, allowing for the analysis of the frequency components of the audio files.

### Visualization of Frequency Components:
The FFT results are plotted using the plot_spectrum function to visualize the amplitudes of the frequency components and their corresponding frequencies.

### Feature Extraction:
Features are extracted based on the FFT results of the audio files. In this example, the average of the first N frequency components is taken.

### Recognition Algorithm:
Using a simple threshold method, the features of the audio files are evaluated against a specific threshold.
## How to Use

1. Place your audio files in the specified directory.
2. Run the `SH_P1.py` script to analyze the audio files and recognize new audio samples.

## Requirements

- `librosa`
- `numpy`
- `matplotlib`

## Contribution

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new pull request.
