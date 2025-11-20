import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# === EDIT THIS PATH to where you put dev-clean/test-clean ===
LIBRISPEECH_PATH = "/path/to/LibriSpeech"  # e.g. "./LibriSpeech"

# Example file mentioned in PDF (adjust to your exact path)
example_file = os.path.join(LIBRISPEECH_PATH, "dev-clean", "84", "121123", "84-121123-0000.flac")

# Load
y, sr = librosa.load(example_file, sr=None)  # sr=None => preserve original sr
duration = len(y) / sr
amin, amax = y.min(), y.max()

print(f"File: {example_file}")
print(f"Sampling rate: {sr} Hz")
print(f"Duration: {duration:.3f} s")
print(f"Amplitude min: {amin:.6f}, max: {amax:.6f}")

# Waveform plot
plt.figure(figsize=(10,3))
times = np.arange(len(y)) / sr
librosa.display.waveshow(y, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.tight_layout()
plt.show()

# STFT parameters: window/hop in ms
win_ms = 50   # 50 ms window (change if needed)
hop_ms = 50   # 50 ms hop
n_fft = 1
# choose n_fft as next power of two >= window_samples
win_samples = int(sr * win_ms / 1000)
n_fft = 1 << (win_samples - 1).bit_length()
hop_length = int(sr * hop_ms / 1000)

S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_samples)
S_mag = np.abs(S)
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

# Plot spectrogram (dB)
plt.figure(figsize=(10,4))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title(f"Spectrogram (magnitude in dB), window {win_ms}ms, hop {hop_ms}ms")
plt.tight_layout()
plt.show()

# MFCCs
n_mfcc = 13
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_samples)
print("MFCC shape (n_mfcc x frames):", mfcc.shape)

plt.figure(figsize=(10,4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
plt.colorbar()
plt.xlabel("Time (s)")
plt.ylabel("MFCC coefficient index")
plt.title("MFCCs")
plt.tight_layout()
plt.show()
