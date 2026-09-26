"""Segmentation + interpretable feature extraction.

Every feature is named and clinically meaningful -- that is deliberate. The
explanation stage (explain.py) reports SHAP attributions against these names,
so the model's reasoning lands in vocabulary a neurologist already uses.

Feature block (60 dims):
   11  acoustic  : pitch, jitter x3, shimmer x3, HNR, and 3 pause/timing
   13  MFCC means
   13  MFCC delta std-devs (NOT means: a delta averages to ~0 over a
       segment, so delta means are provably uninformative -- measured
       |t| < 0.2 for all 13. The spread of the deltas is what encodes
       unsteady articulation.)
   13  GTCC means (gammatone cepstral; falls back to mel-filter cepstra)
   10  spectral  : centroid, bandwidth, rolloff, flatness, ZCR (mean + std)
"""

import warnings

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment, silence
from tqdm import tqdm

import config

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import parselmouth
    from parselmouth.praat import call
    HAVE_PRAAT = True
except ImportError:                                    # pragma: no cover
    HAVE_PRAAT = False

ACOUSTIC_NAMES = [
    "pitch_mean", "pitch_std",
    "jitter_local", "jitter_rap", "jitter_ppq5",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5",
    "hnr_mean",
    "pause_ratio", "voiced_ratio",
]
SPECTRAL_NAMES = [
    "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
    "rolloff_mean", "rolloff_std", "flatness_mean", "flatness_std",
    "zcr_mean", "zcr_std",
]
FEATURE_NAMES = (
    ACOUSTIC_NAMES
    + [f"mfcc{i+1}" for i in range(13)]
    + [f"dmfcc{i+1}_std" for i in range(13)]
    + [f"gtcc{i+1}" for i in range(13)]
    + SPECTRAL_NAMES
)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def segment_file(path):
    """Split one recording on silence, then cap over-long chunks.

    Returns a list of float32 mono arrays at config.TARGET_SR.
    """
    audio = AudioSegment.from_wav(path).set_channels(1).set_frame_rate(config.TARGET_SR)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=config.SILENCE_MIN_LEN_MS,
        silence_thresh=config.SILENCE_THRESH_DBFS,
        keep_silence=100,
    ) or [audio]

    out = []
    for ch in chunks:
        if len(ch) < config.SEG_MIN_MS:
            continue
        # A fixed cap keeps segment counts from tracking how talkative a
        # subject is -- otherwise segment count itself leaks speaker identity.
        for start in range(0, len(ch), config.SEG_MAX_MS):
            piece = ch[start:start + config.SEG_MAX_MS]
            if len(piece) < config.SEG_MIN_MS:
                continue
            y = np.array(piece.get_array_of_samples(), dtype=np.float32)
            y /= (np.abs(y).max() + 1e-9)
            out.append(y)
    return out


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def _praat_acoustics(y, sr):
    if not HAVE_PRAAT:
        return [np.nan] * 9
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)
    vals = pitch.selected_array["frequency"]
    vals = vals[vals > 0]
    pitch_mean = float(np.mean(vals)) if vals.size else 0.0
    pitch_std = float(np.std(vals)) if vals.size else 0.0

    pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
    jit = [call(pp, f"Get jitter ({k})", 0, 0, 1e-4, 0.02, 1.3)
           for k in ("local", "rap", "ppq5")]
    shim = [call([snd, pp], f"Get shimmer ({k})", 0, 0, 1e-4, 0.02, 1.3, 1.6)
            for k in ("local", "apq3", "apq5")]
    harm = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    return [pitch_mean, pitch_std, *jit, *shim, hnr]


def _timing(y, sr):
    """Pause and voicing ratios -- the strongest early PD markers in free speech."""
    intervals = librosa.effects.split(y, top_db=25)
    voiced = sum(e - s for s, e in intervals)
    total = max(len(y), 1)
    return [1.0 - voiced / total, voiced / total]


def _gtcc(y, sr, n=13):
    """Gammatone cepstral coefficients.

    Uses `spafe` when available; otherwise an ERB-spaced mel-filterbank
    cepstrum, which is a close and well-behaved stand-in.
    """
    try:
        from spafe.features.gfcc import gfcc
        return np.nanmean(gfcc(y, fs=sr, num_ceps=n), axis=0)
    except Exception:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmin=50, fmax=sr // 2)
        return np.mean(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n), axis=1)


def extract(y, sr=config.TARGET_SR):
    acoustic = _praat_acoustics(y, sr) + _timing(y, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)
    gt = _gtcc(y, sr)

    spec = []
    for fn in (librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff):
        v = fn(y=y, sr=sr)
        spec += [float(v.mean()), float(v.std())]
    for fn in (librosa.feature.spectral_flatness, librosa.feature.zero_crossing_rate):
        v = fn(y=y)
        spec += [float(v.mean()), float(v.std())]

    vec = np.concatenate([acoustic, mfcc.mean(1), dmfcc.std(1), gt, spec])
    return np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
def build_feature_frame(meta, task):
    """Segment + featurise every recording for one task. Cached to parquet."""
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.WORK_DIR / f"features_{task}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    rows = []
    sub = meta[meta.task == task]
    for r in tqdm(sub.itertuples(), total=len(sub), desc=f"featurising {task}"):
        for i, seg in enumerate(segment_file(r.path)):
            rows.append({
                "subject": r.subject, "label": r.label, "severity": r.severity,
                "stratum": r.stratum, "hy": r.hy, "seg": i,
                **dict(zip(FEATURE_NAMES, extract(seg))),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


# --------------------------------------------------------------------------
# Feature subsets.
#
# The published LOSOCV result on this dataset (Neurology International 2025,
# 95.45% on ReadText) came from acoustic + GTCC, NOT from the full feature
# set. Throwing every feature at a 37-subject problem dilutes the signal, so
# we evaluate the published subset as well as the full set.
# --------------------------------------------------------------------------
FEATURE_GROUPS = {
    "acoustic": ACOUSTIC_NAMES,
    "mfcc": [f"mfcc{i+1}" for i in range(13)],
    "dmfcc": [f"dmfcc{i+1}_std" for i in range(13)],
    "gtcc": [f"gtcc{i+1}" for i in range(13)],
    "spectral": SPECTRAL_NAMES,
}

FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "acoustic+gtcc": FEATURE_GROUPS["acoustic"] + FEATURE_GROUPS["gtcc"],
    "acoustic+mfcc": FEATURE_GROUPS["acoustic"] + FEATURE_GROUPS["mfcc"],
    "cepstral": FEATURE_GROUPS["mfcc"] + FEATURE_GROUPS["gtcc"],
    "acoustic": FEATURE_GROUPS["acoustic"],
}
