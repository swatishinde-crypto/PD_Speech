"""
italian_extract.py
==================

Feature extraction for the external corpus: Italian Parkinson's Voice and
Speech (Dimauro et al., 2017; IEEE DataPort, doi:10.21227/aw6b-tg17).

Task used: the read passage "Il ramarro della zia" (files B1 and B2, first
and second reading). It is the only material in this corpus that matches the
ReadText task of MDVR-KCL.

Groups used: people with Parkinson's disease (PD) against the elderly
healthy controls (EHC). The young controls are excluded because their age
difference from the patients would confound the comparison.

Subject identity. Several patient names appear in more than one sub-folder
of the PD directory (one in three). The spreadsheet lists them with the
same surname initial and age in each place, so they are treated as
one speaker. Grouping them keeps every recording of a speaker on the same
side of each leave-one-subject-out split. Folder names are patient names,
so every speaker is replaced by a code (IT_PD01, IT_HC01, ...) before
anything is written to disk.

Files that are byte-identical (one patient's B2 duplicates the B1 file) are
kept once.

Segmentation, peak normalisation, the 60 hand-crafted features and the
wav2vec2 embeddings are computed with exactly the same code as for MDVR-KCL,
so the two corpora differ only in their audio.

Outputs (work_italian/):
    it_meta.csv           one row per recording, with sample rate and level
    it_hand.parquet       hand-crafted features, one row per segment
    it_ssl.npz            wav2vec2 embeddings (segments x 13 layers x 1536)
    it_ssl_meta.parquet   segment metadata aligned with it_ssl.npz
"""

import glob
import hashlib
import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import features as H
import ssl_features as S

warnings.filterwarnings("ignore")

ROOT = "Italian Parkinson's Voice and speech"
OUT = "work_italian"
GROUPS = {"22 Elderly Healthy Control": 0, "28 People with Parkinson's disease": 1}


def _norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def build_meta():
    rows, seen = [], set()
    for grp, lab in GROUPS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, grp, "**", "*.wav"), recursive=True)):
            base = os.path.basename(f)
            if not re.match(r"B[12]", base):
                continue
            digest = hashlib.md5(open(f, "rb").read()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            spk = _norm_name(os.path.basename(os.path.dirname(f)))
            y, sr = sf.read(f, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            rms = np.sqrt(np.mean(y ** 2)) + 1e-12
            rows.append({
                "path": f, "file": base, "reading": base[:2],
                "subject": ("PD_" if lab else "HC_") + spk.replace(" ", "_"),
                "label": lab, "sr": sr, "duration": len(y) / sr,
                "mean_dbfs": 20 * np.log10(rms),
                "peak_dbfs": 20 * np.log10(np.abs(y).max() + 1e-12),
            })
    meta = pd.DataFrame(rows)
    meta["subject"] = meta.subject.map(pseudonyms(meta.subject))
    return meta


def pseudonyms(names):
    """Replace folder names (which are patient names) with codes such as
    IT_PD01, assigned in alphabetical order within each group, so that no
    name appears in any output file."""
    codes = {}
    for prefix, grp in (("PD_", "IT_PD"), ("HC_", "IT_HC")):
        for k, n in enumerate(sorted(n for n in set(names) if n.startswith(prefix)), 1):
            codes[n] = f"{grp}{k:02d}"
    return codes


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = build_meta()
    meta.drop(columns="path").to_csv(f"{OUT}/it_meta.csv", index=False)
    print(meta.groupby("label").subject.nunique().rename("speakers"))
    print(meta.groupby("label").size().rename("recordings"))

    hand, emb, md = [], [], []
    for r in tqdm(meta.itertuples(), total=len(meta), desc="Italian B1/B2"):
        for i, seg in enumerate(H.segment_file(r.path)):
            info = {"subject": r.subject, "label": r.label, "file": r.file,
                    "sr": r.sr, "seg": i}
            hand.append({**info, **dict(zip(H.FEATURE_NAMES, H.extract(seg)))})
            emb.append(S.embed_segment(seg))
            md.append(info)

    pd.DataFrame(hand).to_parquet(f"{OUT}/it_hand.parquet", index=False)
    np.savez_compressed(f"{OUT}/it_ssl.npz", X=np.stack(emb))
    pd.DataFrame(md).to_parquet(f"{OUT}/it_ssl_meta.parquet", index=False)
    print("segments:", len(md))


if __name__ == "__main__":
    main()
