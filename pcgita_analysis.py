"""
pcgita_analysis.py
==================

Validation on PC-GITA. Run pcgita_extract.py first.

Every modelling choice is carried over unchanged from MDVR-KCL: the same
segmentation, the same 60 hand-crafted descriptors, the same frozen
wav2vec2-base with mean+SD pooling, logistic regression (C=0.05, balanced)
and participant-level aggregation. Layer 6 was fixed on MDVR-KCL before
either external corpus was processed.

Analyses
  1. Leave-one-subject-out on both tasks: hand-crafted against wav2vec2
     layer 6, with bootstrap CIs, exact McNemar, paired bootstrap on AUC and
     a participant-level permutation test.
  2. Layer profile, layers 0-12, both tasks.
  3. Recall by Hoehn & Yahr stage and by the UPDRS speech item, including
     the six stage-1 patients that MDVR-KCL does not contain, and the
     Severity-Independence Score.
  4. Recording-level checks: mean and peak level alone.
  5. Cross-corpus transfer in both directions, read speech and spontaneous
     speech, against MDVR-KCL and (for read speech) the Italian corpus, with
     the training corpus's or each corpus's own standardisation.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import features as H
import stats

warnings.filterwarnings("ignore")

W = "work_pcgita"
LAYER = 6
N_PERM = 200
TASKS = ("ReadText", "Monologue")
rng = np.random.default_rng(config.SEED)


def clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.05, max_iter=3000,
                                               class_weight="balanced"))])


def aggregate(md, p, y=None):
    keep = [c for c in ("subject", "stratum", "hy") if c in md]
    d = md[keep].copy()
    d["prob"] = p
    d["label"] = md.label.values if y is None else y
    agg = {"prob": ("prob", "mean"), "label": ("label", "first")}
    for c in ("stratum", "hy"):
        if c in keep:
            agg[c] = (c, "first")
    g = d.groupby("subject").agg(**agg).reset_index()
    g["pred"] = (g.prob >= 0.5).astype(int)
    return g


def losocv(X, md, y=None):
    y = md.label.values if y is None else y
    s = md.subject.values
    p = np.zeros(len(md))
    for held in np.unique(s):
        tr, te = s != held, s == held
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return aggregate(md, p, y)


def summ(g, name, **extra):
    y, p = g.label.values, g.pred.values
    ci = stats.bootstrap_ci(y, p, g.prob.values)
    out = {"model": name, **extra, "n": len(g),
           "acc": (y == p).mean(), "acc_lo": ci["accuracy_ci"][0], "acc_hi": ci["accuracy_ci"][1],
           "auc": roc_auc_score(y, g.prob), "auc_lo": ci["auc_ci"][0], "auc_hi": ci["auc_ci"][1],
           "mcc": matthews_corrcoef(y, p),
           "sens": p[y == 1].mean(), "spec": 1 - p[y == 0].mean()}
    if "hy" in g:
        pdx = g[g.label == 1]
        symp = pdx[pdx.stratum >= 1].pred.mean()
        out.update(hy1=pdx[pdx.hy <= 1].pred.mean(),
                   hy2=pdx[(pdx.hy > 1) & (pdx.hy < 3)].pred.mean(),
                   hy34=pdx[pdx.hy >= 3].pred.mean(),
                   u0=pdx[pdx.stratum == 0].pred.mean(),
                   u1=pdx[pdx.stratum == 1].pred.mean(),
                   u2=pdx[pdx.stratum >= 2].pred.mean(),
                   SIS=pdx[pdx.stratum == 0].pred.mean() / symp if symp else np.nan)
    return out


def show(r):
    print(f"  {r['model']:34s} acc {r['acc']:.3f} [{r['acc_lo']:.2f},{r['acc_hi']:.2f}]  "
          f"AUC {r['auc']:.3f} [{r['auc_lo']:.2f},{r['auc_hi']:.2f}]  MCC {r['mcc']:.3f}  "
          f"sens {r['sens']:.2f} spec {r['spec']:.2f}", flush=True)


def load(task):
    hand = pd.read_parquet(f"{W}/pcgita_hand_{task}.parquet")
    X = np.load(f"{W}/pcgita_ssl_{task}.npz")["X"]
    md = pd.read_parquet(f"{W}/pcgita_ssl_{task}_meta.parquet")
    return hand, X, md


def main():
    meta = pd.read_csv(f"{W}/pcgita_meta.csv")
    rows, sig_rows, lay_rows = [], [], []
    # recorded so that the verification script needs no derived features,
    # which are not redistributable for this corpus
    pd.DataFrame([{"task": t, "segments": len(pd.read_parquet(
        f"{W}/pcgita_ssl_{t}_meta.parquet"))} for t in TASKS]).to_csv(
        f"{W}/segment_counts.csv", index=False)

    for task in TASKS:
        print(f"\n{'=' * 70}\nPC-GITA {task}\n{'=' * 70}")
        hand, X3, md = load(task)
        Xb = X3[:, LAYER, :]

        g_h = losocv(hand[H.FEATURE_NAMES].values, hand)
        g_w = losocv(Xb, md)
        for g, n in ((g_h, "hand-crafted (60-d)"), (g_w, f"wav2vec2 layer {LAYER}")):
            r = summ(g, n, analysis="within", task=task)
            rows.append(r)
            show(r)
        g_h.to_csv(f"{W}/preds_hand_{task}.csv", index=False)
        g_w.to_csv(f"{W}/preds_w2v_{task}.csv", index=False)

        a = g_h.sort_values("subject").reset_index(drop=True)
        b = g_w.sort_values("subject").reset_index(drop=True)
        mc = stats.mcnemar(a.label.values, a.pred.values, b.pred.values)
        ad = stats.bootstrap_auc_diff(a.label.values, a.prob.values, b.prob.values)
        print(f"  McNemar b01={mc['b01']} b10={mc['b10']} p={mc['p_value']:.4f};  "
              f"dAUC {ad['auc_diff']:+.3f} [{ad['ci'][0]:+.3f},{ad['ci'][1]:+.3f}] "
              f"p={ad['p_two_sided']:.4f}")

        subj = md.drop_duplicates("subject")[["subject", "label"]]
        real = roc_auc_score(g_w.label, g_w.prob)
        null = []
        for _ in range(N_PERM):
            perm = dict(zip(subj.subject, rng.permutation(subj.label.values)))
            gp = losocv(Xb, md, y=md.subject.map(perm).values)
            null.append(roc_auc_score(gp.label, gp.prob))
        null = np.array(null)
        p_perm = (1 + (null >= real).sum()) / (1 + N_PERM)
        np.save(f"{W}/perm_null_{task}.npy", null)
        print(f"  permutation: AUC {real:.3f}, null mean {null.mean():.3f}, "
              f"95th pct {np.quantile(null, .95):.3f}, p = {p_perm:.4f}", flush=True)
        sig_rows.append({"task": task, **mc, "auc_diff": ad["auc_diff"],
                         "auc_diff_lo": ad["ci"][0], "auc_diff_hi": ad["ci"][1],
                         "auc_p": ad["p_two_sided"], "perm_auc": real,
                         "perm_null_mean": null.mean(),
                         "perm_null_95": np.quantile(null, 0.95), "perm_p": p_perm})

        print("  layer profile")
        for L in range(X3.shape[1]):
            g = losocv(X3[:, L, :], md)
            lay_rows.append({"task": task, "layer": L,
                             "acc": (g.label == g.pred).mean(),
                             "auc": roc_auc_score(g.label, g.prob),
                             "mcc": matthews_corrcoef(g.label, g.pred)})
            print(f"    L{L:2d} acc {lay_rows[-1]['acc']:.3f} "
                  f"AUC {lay_rows[-1]['auc']:.3f}", flush=True)

    pd.DataFrame(sig_rows).to_csv(f"{W}/significance.csv", index=False)
    pd.DataFrame(lay_rows).to_csv(f"{W}/layer_sweep.csv", index=False)

    # ------------------------------------------------------------ checks
    print("\nRecording-level checks")
    conf = []
    for task in TASKS:
        per = meta[meta.task == task].groupby("subject").agg(
            label=("label", "first"), mean_dbfs=("mean_dbfs", "mean"),
            peak_dbfs=("peak_dbfs", "mean"), duration=("duration", "mean")).reset_index()
        for c in ("mean_dbfs", "peak_dbfs", "duration"):
            auc = roc_auc_score(per.label, per[c])
            conf.append({"task": task, "variable": c, "auc": max(auc, 1 - auc)})
            print(f"  {task:10s} {c:10s} AUC {max(auc, 1 - auc):.3f}")
    pd.DataFrame(conf).to_csv(f"{W}/recording_checks.csv", index=False)

    # ------------------------------------------------------- cross-corpus
    print("\nCross-corpus transfer (layer %d)" % LAYER)
    src = {}
    for t in ("ReadText", "SpontaneousDialogue"):
        src[("MDVR-KCL", t)] = (
            np.load(f"{config.WORK_DIR}/ssl_{t}.npz")["X"][:, LAYER, :],
            pd.read_parquet(f"{config.WORK_DIR}/ssl_{t}_meta.parquet"))
    src[("Italian", "ReadText")] = (
        np.load("work_italian/it_ssl.npz")["X"][:, LAYER, :],
        pd.read_parquet("work_italian/it_ssl_meta.parquet"))
    for t in TASKS:
        _, X3, md = load(t)
        src[("PC-GITA", t)] = (X3[:, LAYER, :], md)

    pairs = [
        (("MDVR-KCL", "ReadText"), ("PC-GITA", "ReadText")),
        (("PC-GITA", "ReadText"), ("MDVR-KCL", "ReadText")),
        (("MDVR-KCL", "SpontaneousDialogue"), ("PC-GITA", "Monologue")),
        (("PC-GITA", "Monologue"), ("MDVR-KCL", "SpontaneousDialogue")),
        (("Italian", "ReadText"), ("PC-GITA", "ReadText")),
        (("PC-GITA", "ReadText"), ("Italian", "ReadText")),
    ]
    for (sc, st), (tc, tt) in pairs:
        Xs, ms = src[(sc, st)]
        Xt, mt = src[(tc, tt)]
        for scaling in ("source statistics", "per-corpus statistics"):
            if scaling.startswith("per"):
                Xs_, Xt_ = StandardScaler().fit_transform(Xs), StandardScaler().fit_transform(Xt)
            else:
                Xs_, Xt_ = Xs, Xt
            m = clf().fit(Xs_, ms.label.values)
            g = aggregate(mt, m.predict_proba(Xt_)[:, 1])
            r = summ(g, f"{sc} ({st}) -> {tc} ({tt})", analysis="cross",
                     scaling=scaling, task=tt)
            rows.append(r)
            print(f"  [{scaling}]", end="")
            show(r)
            g.to_csv(f"{W}/cross_{sc}_{st}_to_{tc}_{tt}_{scaling.split()[0]}.csv",
                     index=False)

    pd.DataFrame(rows).to_csv(f"{W}/results.csv", index=False)
    print("\nsaved", f"{W}/results.csv")


if __name__ == "__main__":
    main()
