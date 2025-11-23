import os
import random
import glob
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

PATH_LIBRISPEECH = "./LibriSpeech"

FILES_PER_SPEAKER = 20
TRAIN_SPEAKERS = 66
TEST_SPEAKERS = 14

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def load_speakers_txt(path, subset_filter=None):
    """
    Reads SPEAKERS.TXT and returns {speaker_id: (gender, subset)}
    If subset_filter is provided, only include speakers from those subsets.
    """
    spk_dict = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            speaker_id = parts[0].strip()
            gender = parts[1].strip()[0].upper()
            subset = parts[2].strip()

            # Filter by subset if specified
            if subset_filter is None or subset in subset_filter:
                spk_dict[speaker_id] = (gender, subset)
    return spk_dict


def list_speaker_files(base_path, speaker_id):
    """
    Returns a list of all .flac files for a given speaker in the given dataset.
    """
    pattern = os.path.join(base_path, speaker_id, "*", "*.flac")
    return glob.glob(pattern)


def extract_features_from_file(path):
    """
    Computes MFCC mean and Pitch statistics for a single audio file.
    Returns:
        mfcc_vector (n_mfcc)
        pitch_vector (5 statistics)
    """
    try:
        y, sr = librosa.load(path)
    except Exception:
        return None, None

    # MFCC extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = mfcc.mean(axis=1)

    f0 = librosa.yin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = f0[np.isfinite(f0)]

    pitch_stats = np.array([
        np.min(f0),
        np.max(f0),
        np.mean(f0),
        np.median(f0),
        np.var(f0)
    ])

    return mfcc_mean, pitch_stats


def extract_speaker_features(base_path, speaker_id):
    """
    Extracts MFCC and Pitch features for exactly 20 random files of one speaker.
    Returns:
        mean_mfcc_vector (n_mfcc,)
        mean_pitch_vector (5,)
    """
    files = list_speaker_files(base_path, speaker_id)
    if len(files) == 0:
        return None, None

    random.shuffle(files)
    files = files[:FILES_PER_SPEAKER]

    mfcc_list = []
    pitch_list = []

    for f in files:
        mf, pt = extract_features_from_file(f)
        if mf is None:
            continue
        mfcc_list.append(mf)
        pitch_list.append(pt)

    if len(mfcc_list) == 0:
        return None, None

    mfcc_arr = np.vstack(mfcc_list)
    pitch_arr = np.vstack(pitch_list)

    return mfcc_arr.mean(axis=0), pitch_arr.mean(axis=0)


speakers = load_speakers_txt(
    os.path.join(PATH_LIBRISPEECH, "SPEAKERS.TXT"),
    subset_filter=["dev-clean", "test-clean"]
)

# Convert to list of tuples (speaker_id, gender, full_path_to_subset)
all_speakers = []
for spk, (gender, subset) in speakers.items():
    # Path includes the subset directory (dev-clean or test-clean)
    subset_path = os.path.join(PATH_LIBRISPEECH, subset)
    all_speakers.append((spk, gender, subset_path))

random.shuffle(all_speakers)
train_speakers = all_speakers[:TRAIN_SPEAKERS]
test_speakers = all_speakers[TRAIN_SPEAKERS:TRAIN_SPEAKERS + TEST_SPEAKERS]


def build_feature_matrix(speaker_list):
    X_mfcc = []
    X_pitch = []
    y = []

    for speaker_id, gender, base_path in tqdm(speaker_list, desc="Extracting speaker features"):
        mfcc_vec, pitch_vec = extract_speaker_features(base_path, speaker_id)
        X_mfcc.append(mfcc_vec)
        X_pitch.append(pitch_vec)
        y.append(1 if gender == "M" else 0)

    return np.array(X_mfcc), np.array(X_pitch), np.array(y)


print("Extracting features for training speakers...")
X_mfcc_train, X_pitch_train, y_train = build_feature_matrix(train_speakers)

print("Extracting features for test speakers...")
X_mfcc_test, X_pitch_test, y_test = build_feature_matrix(test_speakers)



print("TRAINING SVM ON MFCC FEATURES")

scaler_mfcc = StandardScaler()
X_mfcc_train_s = scaler_mfcc.fit_transform(X_mfcc_train)
X_mfcc_test_s = scaler_mfcc.transform(X_mfcc_test)

svm_mfcc = SVC(kernel="rbf", C=10, gamma="scale")
svm_mfcc.fit(X_mfcc_train_s, y_train)

y_pred_mfcc = svm_mfcc.predict(X_mfcc_test_s)

print("\nMFCC-ONLY RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_mfcc))
print("Confusion Matrix:")
cm_mfcc = confusion_matrix(y_test, y_pred_mfcc)
print(cm_mfcc)
print("Classification Report:")
print(classification_report(y_test, y_pred_mfcc, target_names=["Female", "Male"]))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mfcc, display_labels=["Female", "Male"])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - MFCC Features')
plt.tight_layout()
plt.savefig('confusion_matrix_mfcc.png', dpi=300, bbox_inches='tight')
print("Saved confusion matrix plot to: confusion_matrix_mfcc.png")
plt.close()



print("TRAINING SVM ON PITCH FEATURES")

scaler_pitch = StandardScaler()
X_pitch_train_s = scaler_pitch.fit_transform(X_pitch_train)
X_pitch_test_s = scaler_pitch.transform(X_pitch_test)

svm_pitch = SVC(kernel="rbf", C=10, gamma="scale")
svm_pitch.fit(X_pitch_train_s, y_train)

y_pred_pitch = svm_pitch.predict(X_pitch_test_s)

print("\nPITCH-ONLY RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_pitch))
print("Confusion Matrix:")
cm_pitch = confusion_matrix(y_test, y_pred_pitch)
print(cm_pitch)
print("Classification Report:")
print(classification_report(y_test, y_pred_pitch, target_names=["Female", "Male"]))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pitch, display_labels=["Female", "Male"])
disp.plot(ax=ax, cmap='Greens', values_format='d')
plt.title('Confusion Matrix - Pitch Features')
plt.tight_layout()
plt.savefig('confusion_matrix_pitch.png', dpi=300, bbox_inches='tight')
plt.close()

# Create side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_mfcc, display_labels=["Female", "Male"])
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title(f'MFCC Features\nAccuracy: {accuracy_score(y_test, y_pred_mfcc):.3f}')

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_pitch, display_labels=["Female", "Male"])
disp2.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title(f'Pitch Features\nAccuracy: {accuracy_score(y_test, y_pred_pitch):.3f}')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

