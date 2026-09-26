"""
confounds.py
============

Audits recording-level confounds: properties of the audio FILE that separate
the classes without containing any speech-pathology information.

Why this module exists. On this dataset we measured:

    ReadText              duration  HC 149.8s  PD 127.3s   AUC 0.699  p=0.028
    ReadText              loudness  HC -39.0dB PD -36.3dB  AUC 0.705  p=0.036
    SpontaneousDialogue   loudness  HC -39.3dB PD -36.5dB  AUC 0.705  p=0.040

A classifier that knows nothing except how loud the recording is reaches
AUC ~0.70 on both tasks. PD recordings are systematically ~2.8 dB louder --
almost certainly an artefact of how the sessions were run (microphone
distance, gain, room), not a property of parkinsonian speech. Hypokinetic
dysarthria makes speech *quieter*, so the effect runs opposite to the
clinical expectation, which is what marks it as an artefact.

Any model on this dataset must therefore be compared against a
confound-only baseline. If the full acoustic model does not clearly beat a
model built from duration and loudness alone, its headline accuracy is not
evidence of a speech biomarker.
"""

import warnings

import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")


def recording_metadata(meta):
    """Per-recording file-level properties, with no speech analysis at all."""
    rows = []
    for r in meta.itertuples():
        a = AudioSegment.from_wav(r.path)
        rows.append({
            "subject": r.subject, "task": r.task, "label": r.label,
            "severity": r.severity, "stratum": r.stratum,
            "duration_s": len(a) / 1000.0,
            "dbfs": a.dBFS,                      # mean level
            "max_dbfs": a.max_dBFS,              # peak level
            "rms": a.rms,
        })
    return pd.DataFrame(rows)


def univariate_audit(md, task):
    """Per-variable: group means, t-test, Mann-Whitney, and single-variable AUC."""
    s = md[md.task == task]
    out = []
    for col in ("duration_s", "dbfs", "max_dbfs", "rms"):
        hc, pdg = s[s.label == 0][col], s[s.label == 1][col]
        t, p = ttest_ind(hc, pdg, equal_var=False)
        _, pu = mannwhitneyu(hc, pdg)
        auc = roc_auc_score(s.label, s[col])
        out.append({
            "task": task, "variable": col,
            "HC_mean": hc.mean(), "PD_mean": pdg.mean(),
            "t": t, "p_ttest": p, "p_mannwhitney": pu,
            # Direction-free: a variable that predicts perfectly in reverse
            # is just as much of a confound.
            "auc": max(auc, 1 - auc),
            "direction": "PD higher" if pdg.mean() > hc.mean() else "PD lower",
        })
    return pd.DataFrame(out)


def confound_only_baseline(md, task):
    """LOSOCV using ONLY file-level metadata. This is the number to beat.

    Leave-one-subject-out logistic regression on duration and loudness. If the
    full acoustic pipeline cannot clearly beat this, its performance is not
    attributable to speech.
    """
    s = md[md.task == task].reset_index(drop=True)
    cols = ["duration_s", "dbfs", "max_dbfs", "rms"]
    X, y, g = s[cols].values, s.label.values, s.subject.values

    probs = np.zeros(len(s))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    pred = (probs >= 0.5).astype(int)
    pdx = s[s.label == 1]
    asym = pred[(s.label == 1) & (s.stratum == 0)]
    symp = pred[(s.label == 1) & (s.stratum >= 1)]
    return {
        "task": task, "model": "confound-only (duration + loudness)",
        "accuracy": float((pred == y).mean()),
        "auc": float(roc_auc_score(y, probs)),
        "sensitivity": float(pred[y == 1].mean()),
        "specificity": float(1 - pred[y == 0].mean()),
        "recall_asymptomatic": float(asym.mean()) if len(asym) else np.nan,
        "sis": float(asym.mean() / symp.mean()) if len(symp) and symp.mean() else np.nan,
        "n": len(s),
    }


def run_audit(meta, out_dir=None):
    out_dir = out_dir or config.WORK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    md = recording_metadata(meta)
    md.to_csv(out_dir / "recording_metadata.csv", index=False)

    uni = pd.concat([univariate_audit(md, t) for t in config.TASKS],
                    ignore_index=True)
    base = pd.DataFrame([confound_only_baseline(md, t) for t in config.TASKS])

    uni.to_csv(out_dir / "confound_univariate.csv", index=False)
    base.to_csv(out_dir / "confound_baseline.csv", index=False)

    print("\n--- univariate confound audit ---")
    print(uni.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- confound-only LOSOCV baseline (the number to beat) ---")
    print(base.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return md, uni, base


if __name__ == "__main__":
    import labels
    run_audit(labels.build_table(verbose=False))
