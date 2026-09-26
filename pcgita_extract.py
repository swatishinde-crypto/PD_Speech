"""
pcgita_extract.py
=================

Feature extraction for the second external corpus: PC-GITA (Orozco-Arroyave
et al., LREC 2014), obtained from the corpus authors under their licence.

Tasks used, chosen to match the two MDVR-KCL tasks:
    ReadText    the read sentence set "ayer fui al medico"
    Monologue   the spontaneous monologue, the analogue of SpontaneousDialogue

Speakers: the 50 patients and 50 healthy controls listed in the metadata
spreadsheet, identified by the recording code (AVPEPUDEA.... for patients,
AVPEPUDEAC... for controls). The folder "las que sobraron" holds 17
recordings from speakers who are absent from the spreadsheet and therefore
carry no clinical information; they are not used.

Clinical scores taken from the spreadsheet: total MDS-UPDRS, the UPDRS speech
item (the analogue of UPDRS-II-5 in MDVR-KCL), Hoehn & Yahr stage, sex, age
and years since diagnosis. Controls carry no scores and are treated as 0,
exactly as controls are treated in the MDVR-KCL pipeline.

Segmentation, peak normalisation, the 60 hand-crafted features and the
wav2vec2 embeddings use the same code as the other two corpora, so only the
audio differs.

Outputs (work_pcgita/):
    pcgita_meta.csv          one row per recording, with level and duration
    pcgita_hand_<task>.parquet    hand-crafted features, one row per segment
    pcgita_ssl_<task>.npz         wav2vec2 embeddings (segments x 13 x 1536)
    pcgita_ssl_<task>_meta.parquet   segment metadata aligned with the above
"""

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import features as H
import ssl_features as S

warnings.filterwarnings("ignore")

ROOT = "PC-GITA_per_task_44100Hz"
OUT = "work_pcgita"
META = os.path.join(ROOT, "Copia de PCGITA_metadata.xlsx")
TASKS = {
    "ReadText": "read text/ayerfuialmedico/sin normalizar/[hp][cd]/*.wav",
    "Monologue": "monologue/sin normalizar/[hp][cd]/*.wav",
}


def clinical():
    m = pd.read_excel(META, header=0)
    m.columns = ["code", "updrs", "updrs_speech", "hy", "sex", "age", "years"]
    m["code"] = m.code.str.strip()
    return m.set_index("code")


def build_meta():
    cl = clinical()
    rows = []
    for task, pattern in TASKS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
            code = re.match(r"(AVPEPUDEA?C?\d+)", os.path.basename(f)).group(1)
            if code not in cl.index:
                continue
            c = cl.loc[code]
            label = int(not code.startswith("AVPEPUDEAC"))
            y, sr = sf.read(f, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            rms = np.sqrt(np.mean(y ** 2)) + 1e-12
            rows.append({
                "path": f, "task": task, "subject": code, "label": label,
                "severity": float(c.updrs_speech) if label else 0.0,
                "hy": float(c.hy) if label else 0.0,
                "updrs": float(c.updrs) if label else 0.0,
                "sex": c.sex, "age": float(c.age),
                "sr": sr, "duration": len(y) / sr,
                "mean_dbfs": 20 * np.log10(rms),
                "peak_dbfs": 20 * np.log10(np.abs(y).max() + 1e-12),
            })
    meta = pd.DataFrame(rows)
    # stratum mirrors the MDVR-KCL convention: 0, 1, or 2+ on the speech item
    meta["stratum"] = meta.severity.clip(upper=2).astype(int)
    return meta


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = build_meta()
    meta.drop(columns="path").to_csv(f"{OUT}/pcgita_meta.csv", index=False)
    print(meta.groupby(["task", "label"]).subject.nunique().rename("speakers"))
    pdx = meta[meta.label == 1].drop_duplicates("subject")
    print("H&Y:", pdx.hy.value_counts().sort_index().to_dict())
    print("UPDRS speech item:", pdx.severity.value_counts().sort_index().to_dict())

    for task in TASKS:
        sub = meta[meta.task == task]
        hand, emb, md = [], [], []
        for r in tqdm(sub.itertuples(), total=len(sub), desc=f"PC-GITA {task}"):
            for i, seg in enumerate(H.segment_file(r.path)):
                info = {"subject": r.subject, "label": r.label,
                        "severity": r.severity, "stratum": r.stratum,
                        "hy": r.hy, "sex": r.sex, "age": r.age, "seg": i}
                hand.append({**info, **dict(zip(H.FEATURE_NAMES, H.extract(seg)))})
                emb.append(S.embed_segment(seg))
                md.append(info)
        pd.DataFrame(hand).to_parquet(f"{OUT}/pcgita_hand_{task}.parquet", index=False)
        np.savez_compressed(f"{OUT}/pcgita_ssl_{task}.npz", X=np.stack(emb))
        pd.DataFrame(md).to_parquet(f"{OUT}/pcgita_ssl_{task}_meta.parquet", index=False)
        print(task, "segments:", len(md), flush=True)


if __name__ == "__main__":
    main()
