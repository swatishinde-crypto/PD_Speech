"""
ssl_features.py
===============

Self-supervised speech representations, with optional speaker normalisation.

WHY THIS, AND NOT MORE HAND-CRAFTED FEATURES
--------------------------------------------
Our hand-crafted pipeline (jitter, shimmer, HNR, MFCC, GTCC) reached 0.655
accuracy under leave-one-subject-out, while reaching 0.93-0.97 whenever the
test speaker had been seen in training. The features encode a great deal
about *who is speaking* and comparatively little that transfers to a new
speaker.

Self-supervised models such as wav2vec2 are trained on thousands of speakers
to predict masked speech content. Their intermediate layers are known to
encode articulation and prosody while being substantially less
speaker-specific than low-level acoustic descriptors -- which is exactly the
property we need.

LAYER CHOICE
------------
Not all layers are equal. In wav2vec2-base, the lowest layers stay close to
the waveform (speaker- and channel-heavy) and the highest layers specialise
toward the phonetic pre-training objective. Paralinguistic and articulatory
information peaks in the middle. We extract every layer once and evaluate
them, rather than assuming.

SPEAKER NORMALISATION (CMVN)
----------------------------
Cepstral mean-variance normalisation, applied PER SPEAKER: each subject's
embeddings are z-scored using that subject's own mean and variance.

Two things this buys us:

  1. It removes the constant offset that identifies a speaker, and with it
     the recording-level confound we measured (PD files ~2.8 dB louder,
     AUC 0.705 on loudness alone). A per-speaker mean shift absorbs channel
     gain directly.
  2. It is label-free. At test time we use only the held-out speaker's own
     audio -- no labels, no other subjects -- so it does not leak. This is
     standard practice in speaker-independent speech pathology work.
"""

import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
import features as handcrafted

warnings.filterwarnings("ignore")

MODEL_NAME = "facebook/wav2vec2-base"
SR = 16000

_model = None
_extractor = None


def _load():
    global _model, _extractor
    if _model is None:
        from transformers import AutoFeatureExtractor, AutoModel
        _extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        _model.eval()
    return _extractor, _model


@torch.no_grad()
def embed_segment(y, sr=SR):
    """Return a (n_layers, 2*hidden) array: mean and std pooled over time.

    Std pooling matters here. The mean captures the average articulatory
    configuration; the standard deviation captures how much it VARIES over
    the utterance -- which is the direct analogue of the reduced articulatory
    range seen in hypokinetic dysarthria.
    """
    fe, model = _load()
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    inp = fe(y, sampling_rate=SR, return_tensors="pt")
    out = model(**inp).hidden_states          # tuple: n_layers+1 tensors
    vecs = []
    for h in out:
        h = h.squeeze(0)                       # (time, hidden)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).numpy())
    return np.stack(vecs).astype(np.float32)


def build_ssl_frame(meta, task, layers=None):
    """Segment each recording and embed every segment. Cached per task.

    Segmentation is IDENTICAL to the hand-crafted pipeline, so the only thing
    that changes between the two feature sets is the representation.
    """
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.WORK_DIR / f"ssl_{task}.npz"
    meta_cache = config.WORK_DIR / f"ssl_{task}_meta.parquet"

    if cache.exists() and meta_cache.exists():
        return np.load(cache)["X"], pd.read_parquet(meta_cache)

    rows, embs = [], []
    sub = meta[meta.task == task]
    for r in tqdm(sub.itertuples(), total=len(sub), desc=f"wav2vec2 {task}"):
        for i, seg in enumerate(handcrafted.segment_file(r.path)):
            embs.append(embed_segment(seg))
            rows.append({"subject": r.subject, "label": r.label,
                         "severity": r.severity, "stratum": r.stratum,
                         "hy": r.hy, "seg": i})

    X = np.stack(embs)                        # (n_segments, n_layers, dim)
    md = pd.DataFrame(rows)
    np.savez_compressed(cache, X=X)
    md.to_parquet(meta_cache, index=False)
    return X, md


def speaker_cmvn(X, subjects):
    """Per-speaker mean-variance normalisation.

    Uses ONLY each speaker's own segments, no labels -- so it is safe to
    apply to a held-out subject without leaking anything.
    """
    Xn = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        blk = X[m]
        mu = blk.mean(axis=0, keepdims=True)
        sd = blk.std(axis=0, keepdims=True) + 1e-6
        Xn[m] = (blk - mu) / sd
    return Xn


def to_frame(X_layer, md, prefix="w2v"):
    """Turn one layer's embedding matrix into the DataFrame the evaluator wants."""
    cols = [f"{prefix}{i}" for i in range(X_layer.shape[1])]
    return pd.concat([md.reset_index(drop=True),
                      pd.DataFrame(X_layer, columns=cols)], axis=1)
