"""
run_final.py
============

Statistical validation of the headline claim, plus the corrected channel
normalisation.

Three things happen here:

1. The best SSL configuration (wav2vec2 layer 6) is re-run under LOSOCV,
   its per-subject predictions are saved, and it is compared against the
   hand-crafted baseline with bootstrap CIs, exact McNemar on the
   subject-level decisions, and a paired bootstrap on AUC.

2. A permutation test. With 37 subjects, an accuracy of 0.89 needs to be
   shown to be beyond what label shuffling produces. We refit the entire
   LOSOCV pipeline on permuted labels many times and report where the real
   score falls in that null distribution. This is the check a reviewer will
   ask for first.

3. Channel normalisation done correctly. Per-speaker CMVN collapsed to
   AUC 0.500 because z-scoring each subject by their own mean sets every
   subject's mean to zero, and on this data the class signal lives in the
   between-speaker mean. We instead try:

       per-segment   normalise each segment by its own statistics --
                     removes gain and slow channel drift, keeps
                     between-speaker structure
       global        z-score using TRAINING-set statistics only, which is
                     ordinary feature scaling and leaks nothing
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import labels
import ssl_features as S
import stats

warnings.filterwarnings("ignore")

BEST_LAYER = 6
N_PERM = 200


def normalise(X, subjects, mode):
    """Channel/gain normalisation variants (see module docstring)."""
    if mode == "none":
        return X
    if mode == "per_segment":
        # Each segment standardised by its own mean/std across dimensions.
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        return (X - mu) / sd
    if mode == "per_speaker":
        return S.speaker_cmvn(X, subjects)
    raise ValueError(mode)


def losocv(X, md, seed=config.SEED, labels_override=None):
    """Leave-one-subject-out with a low-capacity linear model."""
    y_all = md.label.values if labels_override is None else labels_override
    subjects = md.subject.values
    probs = np.zeros(len(md))
    for held in np.unique(subjects):
        tr, te = subjects != held, subjects == held
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.05, max_iter=3000,
                                       class_weight="balanced")),
        ])
        pipe.fit(X[tr], y_all[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]

    out = md.copy()
    out["prob"] = probs
    out["_y"] = y_all
    g = out.groupby("subject").agg(
        prob=("prob", "mean"), label=("_y", "first"),
        stratum=("stratum", "first"), hy=("hy", "first")).reset_index()
    g["pred"] = (g.prob >= 0.5).astype(int)
    return g


def summarise(g, name):
    y, p = g.label.values, g.pred.values
    pdx = g[g.label == 1]
    symp = pdx[pdx.stratum >= 1].pred.mean()
    return {
        "model": name,
        "acc": float((y == p).mean()),
        "auc": float(roc_auc_score(y, g.prob)),
        "mcc": float(matthews_corrcoef(y, p)),
        "sens": float(p[y == 1].mean()),
        "spec": float(1 - p[y == 0].mean()),
        "hy2": float(pdx[pdx.hy == 2].pred.mean()),
        "hy34": float(pdx[pdx.hy >= 3].pred.mean()),
        "SIS": float(pdx[pdx.stratum == 0].pred.mean() / symp) if symp else np.nan,
    }


def main():
    meta = labels.build_table(verbose=False)
    all_rows, perm_rows, sig_rows = [], [], []

    for task in config.TASKS:
        print(f"\n{'=' * 70}\n{task}\n{'=' * 70}")
        X3, md = S.build_ssl_frame(meta, task)
        Xb = X3[:, BEST_LAYER, :]

        # --- normalisation variants -----------------------------------
        best_g = None
        for mode in ("none", "per_segment", "per_speaker"):
            g = losocv(normalise(Xb, md.subject.values, mode), md)
            r = summarise(g, f"wav2vec2 L{BEST_LAYER} | norm={mode}")
            ci = stats.bootstrap_ci(g.label, g.pred, g.prob, n_boot=2000)
            r.update(task=task, norm=mode,
                     acc_ci=f"[{ci['accuracy_ci'][0]:.2f},{ci['accuracy_ci'][1]:.2f}]",
                     auc_ci=f"[{ci['auc_ci'][0]:.2f},{ci['auc_ci'][1]:.2f}]")
            all_rows.append(r)
            print(f"  norm={mode:12s} acc {r['acc']:.3f} {r['acc_ci']}  "
                  f"AUC {r['auc']:.3f} {r['auc_ci']}  MCC {r['mcc']:.3f}  "
                  f"H&Y2 {r['hy2']:.3f}", flush=True)
            if mode == "none":
                best_g = g
                g.to_csv(config.WORK_DIR / f"ssl_preds_{task}.csv", index=False)

        # --- significance vs the hand-crafted baseline -----------------
        base_file = config.WORK_DIR / f"subject_preds_{task}_SAMTN_(both).csv"
        if base_file.exists():
            b = pd.read_csv(base_file).sort_values("subject").reset_index(drop=True)
            s = best_g.sort_values("subject").reset_index(drop=True)
            common = b.subject.isin(s.subject)
            b, s = b[common], s[s.subject.isin(b.subject)]
            mc = stats.mcnemar(b.label.values, b.pred.values, s.pred.values)
            ab = stats.bootstrap_auc_diff(b.label.values, b.prob.values, s.prob.values)
            sig_rows.append({"task": task, "vs": "hand-crafted SAMTN",
                             **mc, "auc_diff": ab["auc_diff"],
                             "auc_ci_low": ab["ci"][0], "auc_ci_high": ab["ci"][1],
                             "auc_p": ab["p_two_sided"]})
            print(f"  vs hand-crafted: McNemar b01={mc['b01']} b10={mc['b10']} "
                  f"p={mc['p_value']:.4f}   dAUC {ab['auc_diff']:+.3f} "
                  f"{tuple(round(v, 3) for v in ab['ci'])} p={ab['p_two_sided']:.4f}")

        # --- permutation test ------------------------------------------
        real = summarise(best_g, "real")["acc"]
        rng = np.random.default_rng(config.SEED)
        subj_lab = md.drop_duplicates("subject").set_index("subject").label
        null = []
        for i in range(N_PERM):
            # Permute labels at SUBJECT level, not segment level -- permuting
            # segments would leave each subject's segments inconsistently
            # labelled and make the null far too easy to beat.
            perm = subj_lab.sample(frac=1.0, random_state=int(rng.integers(1e9)))
            mapping = dict(zip(subj_lab.index, perm.values))
            null.append(summarise(losocv(Xb, md,
                        labels_override=md.subject.map(mapping).values),
                        "perm")["acc"])
            if (i + 1) % 50 == 0:
                print(f"    permutation {i+1}/{N_PERM}", flush=True)
        null = np.array(null)
        p_perm = float((null >= real).mean())
        perm_rows.append({"task": task, "real_acc": real,
                          "null_mean": float(null.mean()),
                          "null_p95": float(np.quantile(null, 0.95)),
                          "p_permutation": p_perm, "n_perm": N_PERM})
        print(f"  PERMUTATION: real {real:.3f}  null mean {null.mean():.3f}  "
              f"null 95th {np.quantile(null, 0.95):.3f}  p={p_perm:.4f}")
        np.save(config.WORK_DIR / f"perm_null_{task}.npy", null)

    pd.DataFrame(all_rows).to_csv(config.WORK_DIR / "ssl_final.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(config.WORK_DIR / "permutation.csv", index=False)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(config.WORK_DIR / "ssl_significance.csv",
                                      index=False)
    print("\nwrote ssl_final.csv, permutation.csv, ssl_significance.csv")


if __name__ == "__main__":
    main()
