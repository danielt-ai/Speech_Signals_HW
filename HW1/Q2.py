import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# === EDIT THIS PATH ===
LIBRISPEECH_PATH = "/path/to/LibriSpeech"  # folder containing dev-clean and test-clean
SPEAKERS_TXT = os.path.join(LIBRISPEECH_PATH, "SPEAKERS.TXT")  # check actual location

# Parameters
n_mfcc = 13
win_ms = 25
hop_ms = 10
random_seed = 42
train_speakers_count = 66  # default split; change if you prefer

def read_speakers_txt(path):
    """
    Parse SPEAKERS.TXT. Returns dict speaker_id -> gender ('M' or 'F')
    The file format may have columns: speaker-id, sex, subset, minutes, name
    Adapt parsing if needed.
    """
    d = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            # common SRT format: <speaker-id> <gender> ...
            speaker_id = parts[0]
            gender = parts[1][0].upper()  # M or F
            d[speaker_id] = gender
    return d

speakers_gender = read_speakers_txt(SPEAKERS_TXT)
all_speakers = sorted(speakers_gender.keys())
print(f"Total speakers found in SPEAKERS.TXT: {len(all_speakers)}")

def gather_speaker_files(base_path, speaker_id):
    # search both dev-clean and test-clean
    pattern1 = os.path.join(base_path, "dev-clean", speaker_id, "*", "*.flac")
    pattern2 = os.path.join(base_path, "test-clean", speaker_id, "*", "*.flac")
    files = glob.glob(pattern1) + glob.glob(pattern2)
    return files

def extract_speaker_features(files, n_mfcc=13, sr_target=None):
    """
    For a speaker, compute:
      - mean MFCCs across all frames (n_mfcc)
      - pitch f0 per frame using librosa.yin; compute min,max,mean,median,var
    Returns (mfcc_mean_vec, pitch_stats_vec)
    """
    mfccs_list = []
    pitch_all = []
    for fpath in files:
        try:
            y, sr = librosa.load(fpath, sr=sr_target)
        except Exception as e:
            print("Could not load", fpath, e)
            continue
        # compute frames params from ms
        win_samples = int(sr * win_ms / 1000)
        hop_length = int(sr * hop_ms / 1000)
        n_fft = 1 << (win_samples - 1).bit_length()

        # MFCC
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_samples)
        mfccs_list.append(mf)

        # pitch (yin) - get f0 per frame; if unvoiced, yin may return np.nan; handle it
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=win_samples, hop_length=hop_length)
            # keep only finite values
            f0_clean = f0[np.isfinite(f0)]
            if len(f0_clean) > 0:
                pitch_all.append(f0_clean)
        except Exception as e:
            # if yin fails, skip
            pass

    # aggregate MFCCs: concatenate frames across files then mean per coefficient
    if len(mfccs_list) == 0:
        return None, None
    mf_concat = np.hstack(mfccs_list)  # (n_mfcc, total_frames)
    mf_mean = np.mean(mf_concat, axis=1)  # shape (n_mfcc,)

    # pitch stats
    if len(pitch_all) == 0:
        pitch_stats = np.array([0,0,0,0,0], dtype=float)
    else:
        pitch_concat = np.hstack(pitch_all)
        pitch_stats = np.array([
            np.min(pitch_concat),
            np.max(pitch_concat),
            np.mean(pitch_concat),
            np.median(pitch_concat),
            np.var(pitch_concat)
        ])
    return mf_mean, pitch_stats

# Build dataset
speaker_features = []
speaker_labels = []
failed = []
for sp in tqdm(all_speakers):
    files = gather_speaker_files(LIBRISPEECH_PATH, sp)
    if len(files) == 0:
        failed.append(sp)
        continue
    mf_mean, pitch_stats = extract_speaker_features(files, n_mfcc=n_mfcc)
    if mf_mean is None:
        failed.append(sp)
        continue
    # create three variants:
    feat_mfcc = mf_mean
    feat_pitch = pitch_stats
    feat_combined = np.concatenate([feat_mfcc, feat_pitch])
    speaker_features.append({
        "speaker": sp,
        "mfcc": feat_mfcc,
        "pitch": feat_pitch,
        "combined": feat_combined,
        "gender": speakers_gender[sp]
    })

print(f"Built features for {len(speaker_features)} speakers; failed: {failed}")

# Create arrays and split speakers
df = pd.DataFrame(speaker_features)
# Map gender to label 0/1
df['label'] = df['gender'].map(lambda x: 1 if x.startswith('M') else 0)  # M->1, F->0

# Random speaker split (speaker-level)
rng = np.random.RandomState(random_seed)
speaker_ids = df['speaker'].tolist()
rng.shuffle(speaker_ids)
n_train = train_speakers_count
train_sids = set(speaker_ids[:n_train])
test_sids = set(speaker_ids[n_train:])

train_df = df[df['speaker'].isin(train_sids)]
test_df  = df[df['speaker'].isin(test_sids)]

print(f"Train speakers: {len(train_df)}, Test speakers: {len(test_df)}")

def prepare_Xy(df, feature_key='combined'):
    X = np.vstack(df[feature_key].values)
    y = df['label'].values
    return X, y

# Compare feature sets
for key in ['mfcc', 'pitch', 'combined']:
    print("\n\n=== Feature:", key, "===")
    X_train, y_train = prepare_Xy(train_df, feature_key=key)
    X_test, y_test   = prepare_Xy(test_df, feature_key=key)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Quick SVM with grid search for C, gamma
    param_grid = {'C':[0.1,1,10,100], 'gamma':['scale','auto', 0.01, 0.001]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train_s, y_train)
    best = clf.best_estimator_
    print("Best params:", clf.best_params_)

    y_pred = best.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['F','M']))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
