import mne
import matplotlib.pyplot as plt
import os

# Define paths (update these paths as per your dataset structure)
subject_id = "sub-088"  # Change this to the desired subject
processed_folder = rf"C:\Users\harsh\OneDrive\Desktop\BCI\ds004504\derivatives\{subject_id}\eeg"
unprocessed_folder = rf"C:\Users\harsh\OneDrive\Desktop\BCI\ds004504\{subject_id}\eeg"
output_folder = rf"C:\Users\harsh\OneDrive\Desktop\BCI\outputs\{subject_id}"  # Save plots here

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# File paths
unprocessed_file = os.path.join(unprocessed_folder, f"{subject_id}_task-eyesclosed_eeg.set")
processed_file = os.path.join(processed_folder, f"{subject_id}_task-eyesclosed_eeg.set")

# Function to load and plot EEG data
def process_eeg(file_path, data_type):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load EEG data
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # Print basic EEG info
    print(f"Processing {data_type} data for {subject_id}")
    print("Sampling frequency:", raw.info['sfreq'])
    print("Number of channels:", len(raw.ch_names))
    print("EEG channel names:", raw.ch_names)

    # Plot raw EEG data
    fig1 = raw.plot(n_channels=10, duration=5, scalings='auto', show=False)
    fig1.savefig(os.path.join(output_folder, f"{data_type}_raw_eeg.png"))

    # Apply bandpass filter (1-50 Hz)
    raw.filter(1, 50)

    # Plot Power Spectral Density (PSD)
    fig2 = raw.compute_psd().plot(show=False)
    fig2.savefig(os.path.join(output_folder, f"{data_type}_psd.png"))

    print(f"Plots saved in: {output_folder}")

# Process both unprocessed and processed EEG data
process_eeg(unprocessed_file, "unprocessed")
process_eeg(processed_file, "processed")
