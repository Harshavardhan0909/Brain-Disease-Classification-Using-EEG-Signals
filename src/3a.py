import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Define paths
subject_id = "sub-006"  # Change for different subjects
dataset_root = r"C:\Users\harsh\OneDrive\Desktop\BCI\ds004504"
unprocessed_file = os.path.join(dataset_root, subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")
output_folder = r"C:\Users\harsh\OneDrive\Desktop\BCI\outputs"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load EEG data
raw = mne.io.read_raw_eeglab(unprocessed_file, preload=True)

# Pick a single channel for visualization
channel_idx = 18  # Change index to visualize different channels
channel_name = raw.ch_names[channel_idx]
eeg_data = raw.get_data(picks=channel_idx)  # Extract raw data for this channel
time = np.arange(eeg_data.shape[1]) / raw.info['sfreq']  # Time axis

# Plot raw EEG signal
plt.figure(figsize=(10, 4))
plt.plot(time, eeg_data.T, color="blue", linewidth=0.8)
plt.title(f"Raw EEG Data - {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.savefig(os.path.join(output_folder, f"{subject_id}_raw_eeg.png"))
plt.show()

# Apply preprocessing (Bandpass Filter: 1-50 Hz)
raw.filter(1, 50, fir_design='firwin')

# Get cleaned data
clean_eeg_data = raw.get_data(picks=channel_idx)

# Plot cleaned EEG signal
plt.figure(figsize=(10, 4))
plt.plot(time, clean_eeg_data.T, color="green", linewidth=0.8)
plt.title(f"Cleaned EEG Data - {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.savefig(os.path.join(output_folder, f"{subject_id}_cleaned_eeg.png"))
plt.show()

# Compute Spectrogram (Frequency vs Time)
f, t, Sxx = spectrogram(eeg_data[0], fs=raw.info['sfreq'], nperseg=500)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud', cmap='jet')
plt.colorbar(label="Power (dB)")
plt.title(f"Raw EEG Spectrogram - {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(os.path.join(output_folder, f"{subject_id}_raw_spectrogram.png"))
plt.show()

# Compute Spectrogram for Cleaned EEG
f, t, Sxx_clean = spectrogram(clean_eeg_data[0], fs=raw.info['sfreq'], nperseg=500)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, np.log(Sxx_clean), shading='gouraud', cmap='jet')
plt.colorbar(label="Power (dB)")
plt.title(f"Cleaned EEG Spectrogram - {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(os.path.join(output_folder, f"{subject_id}_cleaned_spectrogram.png"))
plt.show()

print(f"Graphs saved in: {output_folder}")
