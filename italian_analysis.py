"""
italian_analysis.py
===================

External validation on the Italian Parkinson's Voice and Speech corpus.
Run italian_extract.py first.

Every modelling choice is carried over unchanged from the MDVR-KCL study:
the same segmentation, the same 60 hand-crafted features, the same frozen
wav2vec2-base with mean+SD pooling, the same logistic regression (C=0.05,
balanced) and the same participant-level aggregation. Layer 6 was fixed on
MDVR-KCL before this corpus was examined; the full layer sweep is reported
only to show the profile, not to pick a layer.

Analyses
  1. Leave-one-subject-out: hand-crafted vs wav2vec2 layer 6, with bootstrap
     CIs, exact McNemar, paired bootstrap on AUC and a subject-level
     permutation test.
  2. Layer profile (layers 0-12) under the same protocol.
  3. Recording-level checks: how well sample rate, mean level and peak level
     alone separate the groups, and a sensitivity run restricted to speakers
     recorded at 16 kHz.
  4. Cross-corpus transfer, ReadText only: train on MDVR-KCL and test on the
     Italian corpus, and the reverse. Features are standardised either with
     the training corpus's statistics or with each corpus's own statistics
     (label-free, computed from the unlabelled test corpus).
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

W = "work_italian"
LAYER = 6
N_PERM = 200
rng = np.random.default_rng(config.SEED)


def clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.05, max_iter=3000,
                                               class_weight="balanced"))])


def losocv(X, md, y=None):
    y = md.label.values if y is None else y
    s = md.subject.values
    p = np.zeros(len(md))
    for held in np.unique(s):
        tr, te = s != held, s == held
        m = clf().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return aggregate(md, p, y)


def aggregate(md, p, y=None):
    d = md[["subject", "sr"]].copy() if "sr" in md else md[["subject"]].copy()
    d["prob"] = p
    d["label"] = md.label.values if y is None else y
    g = d.groupby("subject").agg(prob=("prob", "mean"), label=("label", "first"),
                                 **({"sr": ("sr", "first")} if "sr" in d else {})
                                 ).reset_index()
    g["pred"] = (g.prob >= 0.5).astype(int)
    return g


def summ(g, name, **extra):
    y, p = g.label.values, g.pred.values
    ci = stats.bootstrap_ci(y, p, g.prob.values)
    return {"model": name, **extra, "n": len(g),
            "acc": (y == p).mean(), "acc_lo": ci["accuracy_ci"][0], "acc_hi": ci["accuracy_ci"][1],
            "auc": roc_auc_score(y, g.prob), "auc_lo": ci["auc_ci"][0], "auc_hi": ci["auc_ci"][1],
            "mcc": matthews_corrcoef(y, p),
            "sens": p[y == 1].mean(), "spec": 1 - p[y == 0].mean()}


def show(r):
    print(f"  {r['model']:38s} acc {r['acc']:.3f} [{r['acc_lo']:.2f},{r['acc_hi']:.2f}]  "
          f"AUC {r['auc']:.3f} [{r['auc_lo']:.2f},{r['auc_hi']:.2f}]  MCC {r['mcc']:.3f}  "
          f"sens {r['sens']:.2f} spec {r['spec']:.2f}", flush=True)


def main():
    meta = pd.read_csv(f"{W}/it_meta.csv")
    hand = pd.read_parquet(f"{W}/it_hand.parquet")
    X3 = np.load(f"{W}/it_ssl.npz")["X"]
    md = pd.read_parquet(f"{W}/it_ssl_meta.parquet")
    rows = []

    print("corpus:", meta.groupby("label").subject.nunique().to_dict(),
          "speakers;", len(meta), "recordings;", len(md), "segments")

    # ---------------------------------------------------------------- 1
    print("\n1. Within-corpus LOSOCV")
    g_h = losocv(hand[H.FEATURE_NAMES].values, hand)
    g_w = losocv(X3[:, LAYER, :], md)
    for g, n in ((g_h, "hand-crafted (60-d)"), (g_w, f"wav2vec2 layer {LAYER}")):
        r = summ(g, n, analysis="within"); rows.append(r); show(r)
    g_h.to_csv(f"{W}/preds_hand.csv", index=False)
    g_w.to_csv(f"{W}/preds_w2v_L{LAYER}.csv", index=False)

    a = g_h.sort_values("subject").reset_index(drop=True)
    b = g_w.sort_values("subject").reset_index(drop=True)
    mc = stats.mcnemar(a.label.values, a.pred.values, b.pred.values)
    ad = stats.bootstrap_auc_diff(a.label.values, a.prob.values, b.prob.values)
    sig = {**mc, "auc_diff": ad["auc_diff"], "auc_diff_lo": ad["ci"][0],
           "auc_diff_hi": ad["ci"][1], "auc_p": ad["p_two_sided"]}
    print(f"  McNemar b01={mc['b01']} b10={mc['b10']} p={mc['p_value']:.4f};  "
          f"dAUC {ad['auc_diff']:+.3f} [{ad['ci'][0]:+.3f},{ad['ci'][1]:+.3f}] p={ad['p_two_sided']:.4f}")

    # subject-level permutation: shuffle labels across speakers
    subj = md.drop_duplicates("subject")[["subject", "label"]]
    real_auc = roc_auc_score(g_w.label, g_w.prob)
    null = []
    for _ in range(N_PERM):
        perm = dict(zip(subj.subject, rng.permutation(subj.label.values)))
        yp = md.subject.map(perm).values
        gp = losocv(X3[:, LAYER, :], md, y=yp)
        null.append(roc_auc_score(gp.label, gp.prob))
    null = np.array(null)
    p_perm = (1 + (null >= real_auc).sum()) / (1 + N_PERM)
    np.save(f"{W}/perm_null.npy", null)
    sig.update(perm_auc=real_auc, perm_null_mean=null.mean(),
               perm_null_95=np.quantile(null, 0.95), perm_p=p_perm)
    print(f"  permutation: AUC {real_auc:.3f}, null mean {null.mean():.3f}, "
          f"95th pct {np.quantile(null, .95):.3f}, p = {p_perm:.4f}")
    pd.DataFrame([sig]).to_csv(f"{W}/significance.csv", index=False)

    # ---------------------------------------------------------------- 2
    print("\n2. Layer profile")
    lay = []
    for L in range(X3.shape[1]):
        g = losocv(X3[:, L, :], md)
        lay.append({"layer": L, "acc": (g.label == g.pred).mean(),
                    "auc": roc_auc_score(g.label, g.prob),
                    "mcc": matthews_corrcoef(g.label, g.pred)})
        print(f"  L{L:2d} acc {lay[-1]['acc']:.3f} AUC {lay[-1]['auc']:.3f}", flush=True)
    pd.DataFrame(lay).to_csv(f"{W}/layer_sweep.csv", index=False)

    # ---------------------------------------------------------------- 3
    print("\n3. Recording-level checks")
    per = meta.groupby("subject").agg(label=("label", "first"), sr=("sr", "first"),
                                      mean_dbfs=("mean_dbfs", "mean"),
                                      peak_dbfs=("peak_dbfs", "mean")).reset_index()
    conf = []
    for c in ("sr", "mean_dbfs", "peak_dbfs"):
        auc = roc_auc_score(per.label, per[c])
        conf.append({"variable": c, "auc": max(auc, 1 - auc)})
        print(f"  {c:10s} AUC {max(auc, 1 - auc):.3f}")
    print("  speakers by group and sample rate:\n",
          per.groupby(["label", "sr"]).size().to_string())
    pd.DataFrame(conf).to_csv(f"{W}/recording_checks.csv", index=False)

    keep = per[per.sr == 16000].subject
    m16 = md.subject.isin(keep).values
    g16 = losocv(X3[m16, LAYER, :], md[m16].reset_index(drop=True))
    r = summ(g16, f"wav2vec2 layer {LAYER}, 16 kHz speakers only", analysis="within_16k")
    rows.append(r); show(r)
    h16 = hand.subject.isin(keep).values
    g16h = losocv(hand.loc[h16, H.FEATURE_NAMES].values, hand[h16].reset_index(drop=True))
    r = summ(g16h, "hand-crafted, 16 kHz speakers only", analysis="within_16k")
    rows.append(r); show(r)
    # recall split by recording rate, full model
    for sr_, grp in g_w[g_w.label == 1].groupby("sr"):
        print(f"  PD recall at {sr_} Hz: {grp.pred.mean():.3f} (n={len(grp)})")

    # ---------------------------------------------------------------- 4
    print("\n4. Cross-corpus transfer (ReadText, layer %d)" % LAYER)
    Xm = np.load(f"{config.WORK_DIR}/ssl_ReadText.npz")["X"][:, LAYER, :]
    mm = pd.read_parquet(f"{config.WORK_DIR}/ssl_ReadText_meta.parquet")
    Xi, mi = X3[:, LAYER, :], md
    cross = []
    for src, (Xs, ms), tgt, (Xt, mt) in (
            ("MDVR-KCL", (Xm, mm), "Italian", (Xi, mi)),
            ("Italian", (Xi, mi), "MDVR-KCL", (Xm, mm))):
        for scaling in ("source statistics", "per-corpus statistics"):
            if scaling == "per-corpus statistics":
                Xs_ = StandardScaler().fit_transform(Xs)
                Xt_ = StandardScaler().fit_transform(Xt)
            else:
                Xs_, Xt_ = Xs, Xt
            m = clf().fit(Xs_, ms.label.values)
            g = aggregate(mt, m.predict_proba(Xt_)[:, 1])
            r = summ(g, f"{src} -> {tgt}", analysis="cross", scaling=scaling)
            rows.append(r); cross.append(r)
            print(f"  [{scaling}]", end=""); show(r)
            g.to_csv(f"{W}/cross_{src}_to_{tgt}_{scaling.split()[0]}.csv", index=False)

    pd.DataFrame(rows).to_csv(f"{W}/results.csv", index=False)
    print("\nsaved", f"{W}/results.csv")


if __name__ == "__main__":
    main()
