"""Leave-one-subject-out evaluation with severity-stratified reporting.

The headline number in this project is NOT overall accuracy. It is the
Severity-Independence Score.

    SIS = recall on PD patients rated UPDRS-II-5 == 0
          -------------------------------------------
          recall on PD patients rated UPDRS-II-5 >= 1

A model that only hears audible voice degradation scores near 0: it catches
the symptomatic patients and misses the rest. A model that has learnt
something about the disease itself scores near 1. Published work on this
dataset reports a single pooled accuracy, which cannot distinguish the two.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
import models


def segment_accuracy(seg_pred):
    """Segment-level accuracy under the same LOSOCV folds.

    Reported alongside the subject-level number because published work is not
    always explicit about which one it quotes, and on this dataset they differ
    substantially.
    """
    return float(((seg_pred.prob >= 0.5).astype(int) == seg_pred.label).mean())


def _subject_level(seg_pred):
    """Aggregate segment probabilities to one decision per subject."""
    g = seg_pred.groupby("subject").agg(
        prob=("prob", "mean"), label=("label", "first"),
        severity=("severity", "first"), stratum=("stratum", "first"),
        hy=("hy", "first"), n_seg=("prob", "size"),
    ).reset_index()
    g["pred"] = (g["prob"] >= 0.5).astype(int)
    return g


def losocv(df, *, model="samtn", use_severity=True, use_adversary=True,
           feature_set="all", tune=True):
    """Leave-one-subject-out. No subject contributes to its own prediction.

    `feature_set` selects a named subset from features.FEATURE_SETS. The
    published LOSOCV result on this dataset used acoustic+GTCC rather than
    every feature, so "all" is not automatically the right comparison.

    `tune` enables a nested, subject-grouped grid search for the SVM arm.
    """
    import features as _feat

    meta_cols = {"subject", "label", "severity", "stratum", "hy", "seg"}
    if feature_set == "all":
        feat_cols = [c for c in df.columns if c not in meta_cols]
    else:
        wanted = _feat.FEATURE_SETS[feature_set]
        feat_cols = [c for c in wanted if c in df.columns]
    subjects = sorted(df.subject.unique())
    spk_index = {s: i for i, s in enumerate(subjects)}

    out = []
    for held in subjects:
        tr = df[df.subject != held]
        te = df[df.subject == held]

        scaler = StandardScaler().fit(tr[feat_cols].values)
        Xtr, Xte = scaler.transform(tr[feat_cols].values), scaler.transform(te[feat_cols].values)

        if model == "svm":
            # Groups are passed so the INNER tuning splits are subject-aware
            # too -- otherwise hyperparameter selection leaks across subjects.
            clf = models.make_svm_baseline(tune=tune, groups=tr.subject.values)
            if tune:
                clf.fit(Xtr, tr.label.values, groups=tr.subject.values)
            else:
                clf.fit(Xtr, tr.label.values)
            prob = clf.predict_proba(Xte)[:, 1]
        else:
            # Speaker ids are remapped per fold so the head always sees a
            # contiguous label space over the 36 training subjects.
            local = {s: i for i, s in enumerate(sorted(tr.subject.unique()))}
            net = models.train_samtn(
                Xtr, tr.label.values.astype(np.float32),
                tr.severity.values.astype(np.float32),
                tr.subject.map(local).values,
                n_speakers=len(local),
                use_severity=use_severity, use_adversary=use_adversary,
            )
            prob, _ = models.predict(net, Xte)

        out.append(pd.DataFrame({
            "subject": held, "prob": prob, "label": te.label.values,
            "severity": te.severity.values, "stratum": te.stratum.values,
            "hy": te.hy.values,
        }))

    seg = pd.concat(out, ignore_index=True)
    subj = _subject_level(seg)
    subj.attrs["segment_accuracy"] = segment_accuracy(seg)
    subj.attrs["n_features"] = len(feat_cols)
    subj.attrs["feature_set"] = feature_set
    return subj


def report(subj, name="model"):
    y, p, pr = subj.label.values, subj.pred.values, subj.prob.values
    res = {
        "model": name,
        "accuracy": float((y == p).mean()),
        "auc": float(roc_auc_score(y, pr)) if len(set(y)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y, p)),
        "sensitivity": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "specificity": float(1 - p[y == 0].mean()) if (y == 0).any() else float("nan"),
    }

    # --- the contribution: accuracy broken out by speech-severity rating ---
    pdx = subj[subj.label == 1]
    for s, tag in [(0, "asymptomatic"), (1, "mild"), (2, "moderate_severe")]:
        grp = pdx[pdx.stratum == s]
        res[f"recall_updrs{s}_{tag}"] = float(grp.pred.mean()) if len(grp) else float("nan")
        res[f"n_updrs{s}"] = int(len(grp))

    asym = pdx[pdx.stratum == 0]
    symp = pdx[pdx.stratum >= 1]
    denom = symp.pred.mean() if len(symp) else np.nan
    res["severity_independence_score"] = (
        float(asym.pred.mean() / denom) if len(asym) and denom else float("nan")
    )

    # --- ID31: healthy control the clinicians rated as degraded ------------
    row = subj[subj.subject == config.CONTROL_WITH_DEGRADATION]
    if len(row):
        res["ID31_prob_pd"] = float(row.prob.iloc[0])
        res["ID31_misclassified"] = bool(row.pred.iloc[0] == 1)

    return res


def print_report(res):
    tags = {0: "asymptomatic", 1: "mild", 2: "moderate_severe"}
    print(f"\n=== {res['model']} ===")
    print(f"  accuracy {res['accuracy']:.3f}   AUC {res['auc']:.3f}   MCC {res['mcc']:.3f}")
    print(f"  sensitivity {res['sensitivity']:.3f}   specificity {res['specificity']:.3f}")
    print("  recall by UPDRS-II-5 stratum:")
    for s, tag in tags.items():
        print(f"    {s} ({tag:15s}) n={res[f'n_updrs{s}']:2d}  "
              f"recall {res[f'recall_updrs{s}_{tag}']:.3f}")
    print(f"  Severity-Independence Score: {res['severity_independence_score']:.3f}")
    if "ID31_prob_pd" in res:
        flag = "MISCLASSIFIED as PD" if res["ID31_misclassified"] else "correctly HC"
        print(f"  ID31 (control with rated degradation): "
              f"p(PD)={res['ID31_prob_pd']:.3f}  -> {flag}")
