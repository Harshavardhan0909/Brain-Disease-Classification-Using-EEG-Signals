import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Define dataset paths
dataset_root = r"C:\Users\harsh\OneDrive\Desktop\BCI\ds004504"
subjects = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]  # Select at least 5 subjects
output_folder = r"C:\Users\harsh\OneDrive\Desktop\BCI\outputs"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Store data for PCA/t-SNE analysis
eeg_data_list = []
subject_labels = []

# Loop through selected subjects
for subject_id in subjects:
    print(f"\nProcessing Subject: {subject_id}")
    
    # Load EEG file
    file_path = os.path.join(dataset_root, subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.filter(1, 50)  # Bandpass filter 1-50 Hz

    # Get all 19 EEG channels
    all_channels = raw.ch_names[:19]  # First 19 channels are EEG
    
    # Plot EEG signals for all channels
    fig, axes = plt.subplots(nrows=len(all_channels), ncols=1, figsize=(12, 14), sharex=True)
    time = np.arange(raw.n_times) / raw.info['sfreq']  # Time axis
    
    for i, ch in enumerate(all_channels):
        eeg_data = raw.get_data(picks=ch)[0]
        axes[i].plot(time, eeg_data, label=f"{ch}", linewidth=0.8)
        axes[i].set_ylabel("Amplitude (µV)")
        axes[i].legend(loc="upper right")
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EEG Signals for {subject_id} (All 19 Channels)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{subject_id}_all_channels_eeg.png"))
    plt.show()

    # Plot Power Spectral Density (PSD) for all channels
    raw.compute_psd().plot(picks=all_channels, show=False)
    plt.savefig(os.path.join(output_folder, f"{subject_id}_all_channels_psd.png"))
    plt.show()

    # Store mean EEG power for subject-wise analysis
    eeg_features = raw.get_data(picks=all_channels).mean(axis=1)  # Mean power per channel
    eeg_data_list.append(eeg_features)
    subject_labels.append(subject_id)

# Convert EEG features to numpy array
eeg_data_matrix = np.array(eeg_data_list)

# Heatmap of EEG power per subject
plt.figure(figsize=(12, 6))
sns.heatmap(eeg_data_matrix, annot=True, xticklabels=all_channels, yticklabels=subject_labels, cmap="coolwarm")
plt.title("Brain Activity Heatmap Across Subjects")
plt.xlabel("EEG Channels")
plt.ylabel("Subjects")
plt.savefig(os.path.join(output_folder, "subject_brain_heatmap.png"))
plt.show()

# Perform PCA on EEG features
pca = PCA(n_components=2)
eeg_pca = pca.fit_transform(eeg_data_matrix)

# Perform t-SNE for subject clustering
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
eeg_tsne = tsne.fit_transform(eeg_data_matrix)

# Scatter plot for PCA
plt.figure(figsize=(8, 6))
plt.scatter(eeg_pca[:, 0], eeg_pca[:, 1], c=np.arange(len(subjects)), cmap="viridis", edgecolors="k")
for i, txt in enumerate(subject_labels):
    plt.annotate(txt, (eeg_pca[i, 0], eeg_pca[i, 1]), fontsize=10)
plt.title("PCA: EEG Feature Clustering by Subject")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(output_folder, "pca_subjects.png"))
plt.show()

# Scatter plot for t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(eeg_tsne[:, 0], eeg_tsne[:, 1], c=np.arange(len(subjects)), cmap="coolwarm", edgecolors="k")
for i, txt in enumerate(subject_labels):
    plt.annotate(txt, (eeg_tsne[i, 0], eeg_tsne[i, 1]), fontsize=10)
plt.title("t-SNE: EEG Feature Clustering by Subject")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig(os.path.join(output_folder, "tsne_subjects.png"))
plt.show()

print(f"All plots saved in: {output_folder}")
