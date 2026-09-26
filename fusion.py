"""
fusion.py
=========

Combines the two representations that have each been evaluated separately:

    hand-crafted (60 dims)  jitter, shimmer, HNR, pause/timing, MFCC,
                            delta-MFCC std, GTCC, spectral descriptors.
                            Clinically interpretable. LOSOCV 0.655 / 0.562.

    wav2vec2 layer 6 (1536) self-supervised, mid-network, where articulatory
                            information peaks. LOSOCV 0.892 / 0.861.

Nobody has combined them on this dataset. The published subject-independent
reference (95.45% ReadText) used tuned acoustic+GTCC alone.

A NOTE ON SELECTION BIAS -- READ BEFORE QUOTING ANY NUMBER HERE
---------------------------------------------------------------
With 37 subjects, running several configurations and reporting the best one
is itself a source of optimism: the maximum over k noisy estimates is biased
upward even when every individual estimate is unbiased.

Two things are done about it:

  1. The late-fusion WEIGHT is chosen by an inner leave-one-subject-out loop
     over the training subjects only (`nested_late_fusion`). The outer
     estimate for that configuration is therefore honest.

  2. The configuration list is kept deliberately short and fixed in advance,
     and every configuration is reported -- not just the winner. The spread
     across configurations is the reader's guide to how much the top number
     should be discounted.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import features as HC
import labels
import ssl_features as S
import stats

warnings.filterwarnings("ignore")

BEST_LAYER = 6
PUBLISHED = {"ReadText": 0.9545, "SpontaneousDialogue": 0.837}


def _clf():
    """Deliberately low capacity: 1536+ dims, 37 people."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, max_iter=3000,
                                   class_weight="balanced")),
    ])


def _losocv_probs(X, y, subjects):
    """Segment-level out-of-fold probabilities under LOSOCV."""
    probs = np.zeros(len(y))
    for held in np.unique(subjects):
        tr, te = subjects != held, subjects == held
        probs[te] = _clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return probs


def _aggregate(md, probs):
    o = md.copy()
    o["prob"] = probs
    g = o.groupby("subject").agg(
        prob=("prob", "mean"), label=("label", "first"),
        stratum=("stratum", "first"), hy=("hy", "first")).reset_index()
    g["pred"] = (g.prob >= 0.5).astype(int)
    return g


def summarise(g, name, task):
    y, p = g.label.values, g.pred.values
    pdx = g[g.label == 1]
    symp = pdx[pdx.stratum >= 1].pred.mean()
    ci = stats.bootstrap_ci(y, p, g.prob.values, n_boot=2000)
    return {
        "task": task, "config": name,
        "acc": float((y == p).mean()),
        "acc_ci": f"[{ci['accuracy_ci'][0]:.2f},{ci['accuracy_ci'][1]:.2f}]",
        "auc": float(roc_auc_score(y, g.prob)),
        "auc_ci": f"[{ci['auc_ci'][0]:.2f},{ci['auc_ci'][1]:.2f}]",
        "mcc": float(matthews_corrcoef(y, p)),
        "sens": float(p[y == 1].mean()),
        "spec": float(1 - p[y == 0].mean()),
        "hy2": float(pdx[pdx.hy == 2].pred.mean()),
        "hy34": float(pdx[pdx.hy >= 3].pred.mean()),
        "SIS": float(pdx[pdx.stratum == 0].pred.mean() / symp) if symp else np.nan,
        "vs_published": float((y == p).mean()) - PUBLISHED[task],
    }


def nested_late_fusion(Xa, Xb, y, subjects, weights=np.arange(0, 1.01, 0.1)):
    """Late fusion whose blend weight is chosen WITHOUT touching the test fold.

    For each held-out subject we run an inner leave-one-subject-out loop over
    the remaining 36, pick the weight that maximises inner AUC, and only then
    apply it to the held-out subject. The reported score is therefore not
    inflated by having tuned the weight on the test data.
    """
    probs = np.zeros(len(y))
    chosen = []
    for held in np.unique(subjects):
        tr, te = subjects != held, subjects == held
        ytr, str_ = y[tr], subjects[tr]

        # inner loop on training subjects only
        pa_in = _losocv_probs(Xa[tr], ytr, str_)
        pb_in = _losocv_probs(Xb[tr], ytr, str_)
        best_w, best_auc = 0.5, -1
        for w in weights:
            a = roc_auc_score(ytr, w * pa_in + (1 - w) * pb_in)
            if a > best_auc:
                best_auc, best_w = a, w
        chosen.append(best_w)

        pa = _clf().fit(Xa[tr], ytr).predict_proba(Xa[te])[:, 1]
        pb = _clf().fit(Xb[tr], ytr).predict_proba(Xb[te])[:, 1]
        probs[te] = best_w * pa + (1 - best_w) * pb
    return probs, float(np.mean(chosen))


def main():
    meta = labels.build_table(verbose=False)
    rows = []

    for task in config.TASKS:
        print(f"\n{'=' * 72}\n{task}   (published subject-independent reference: "
              f"{PUBLISHED[task]:.3f})\n{'=' * 72}", flush=True)

        hc = HC.build_feature_frame(meta, task)
        X3, md = S.build_ssl_frame(meta, task)

        # The two pipelines share segmentation, so rows must line up exactly.
        assert (hc.subject.values == md.subject.values).all() and \
               (hc.seg.values == md.seg.values).all(), "row misalignment"

        feat_cols = [c for c in hc.columns
                     if c not in {"subject", "label", "severity", "stratum", "hy", "seg"}]
        Xh = hc[feat_cols].values.astype(np.float32)
        Xw = X3[:, BEST_LAYER, :]
        Xm = X3[:, 5:9, :].mean(axis=1)          # mid-block average, layers 5-8
        y, subj = md.label.values, md.subject.values

        configs = {
            "hand-crafted only (60d)": Xh,
            f"wav2vec2 L{BEST_LAYER} only (1536d)": Xw,
            "wav2vec2 L5-8 mean (1536d)": Xm,
            "EARLY fusion: L6 + hand-crafted": np.hstack([Xw, Xh]),
            "EARLY fusion: L5-8 mean + hand-crafted": np.hstack([Xm, Xh]),
        }
        for name, X in configs.items():
            g = _aggregate(md, _losocv_probs(X, y, subj))
            r = summarise(g, name, task)
            rows.append(r)
            print(f"  {name:42s} acc {r['acc']:.3f} {r['acc_ci']}  "
                  f"AUC {r['auc']:.3f}  MCC {r['mcc']:.3f}  H&Y2 {r['hy2']:.3f}  "
                  f"vs pub {r['vs_published']:+.3f}", flush=True)
            g.to_csv(config.WORK_DIR /
                     f"fusion_preds_{task}_{name[:18].replace(' ', '_').replace(':','')}.csv",
                     index=False)

        p, wbar = nested_late_fusion(Xw, Xh, y, subj)
        g = _aggregate(md, p)
        r = summarise(g, f"LATE fusion (nested weight, mean w={wbar:.2f})", task)
        rows.append(r)
        print(f"  {r['config']:42s} acc {r['acc']:.3f} {r['acc_ci']}  "
              f"AUC {r['auc']:.3f}  MCC {r['mcc']:.3f}  H&Y2 {r['hy2']:.3f}  "
              f"vs pub {r['vs_published']:+.3f}", flush=True)
        g.to_csv(config.WORK_DIR / f"fusion_preds_{task}_late.csv", index=False)

    t = pd.DataFrame(rows)
    t.to_csv(config.WORK_DIR / "fusion_results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n\n================ ALL FUSION RESULTS ================")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
