"""
run_ssl.py
==========

Evaluates the lever-1 + lever-2 pipeline: wav2vec2 embeddings, optionally
with per-speaker CMVN, under the SAME leave-one-subject-out protocol as
everything else.

Two factors are varied:

    layer  0..12   which wav2vec2 layer to read out. Layer 0 is the CNN
                   feature encoder output (closest to the waveform,
                   speaker-heavy); layer 12 is the top transformer layer
                   (specialised toward the pre-training objective).
                   Articulatory / paralinguistic content usually peaks in
                   the middle, but we measure rather than assume.

    cmvn   on/off  per-speaker mean-variance normalisation.

The comparison that matters is not "which layer wins" but whether ANY
configuration clearly beats the hand-crafted baseline of 0.655 (ReadText)
and 0.562 (SpontaneousDialogue) -- and whether recall on the early-stage
H&Y 2 patients improves on 0.375, which is the clinically interesting number.

    python run_ssl.py --task ReadText
    python run_ssl.py --task ReadText --classifier mlp
"""

import argparse
import warnings

import numpy as np
import pandas as pd

import config
import labels
import ssl_features as S

warnings.filterwarnings("ignore")

# Hand-crafted reference points from the main experiment, for the comparison
# column. These came from the identical LOSOCV protocol.
BASELINE = {
    "ReadText": {"acc": 0.655, "hy2": 0.375},
    "SpontaneousDialogue": {"acc": 0.562, "hy2": 0.250},
}


def losocv_ssl(Xl, md, classifier="logreg"):
    """Leave-one-subject-out on one layer's embeddings.

    A linear model is the right default here: wav2vec2 embeddings are 1536-d
    and we have ~660 segments from 37 people. Anything with more capacity
    memorises speakers, which is the exact failure we are trying to avoid.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    subjects = md.subject.values
    probs = np.zeros(len(md))

    for held in np.unique(subjects):
        tr, te = subjects != held, subjects == held
        if classifier == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0,
                                max_iter=600, random_state=config.SEED)
        else:
            clf = LogisticRegression(C=0.05, max_iter=3000,
                                     class_weight="balanced")
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(Xl[tr], md.label.values[tr])
        probs[te] = pipe.predict_proba(Xl[te])[:, 1]

    # Aggregate segment probabilities to one decision per subject.
    out = md.copy()
    out["prob"] = probs
    g = out.groupby("subject").agg(
        prob=("prob", "mean"), label=("label", "first"),
        stratum=("stratum", "first"), hy=("hy", "first")).reset_index()
    g["pred"] = (g.prob >= 0.5).astype(int)

    y, p = g.label.values, g.pred.values
    pdx = g[g.label == 1]
    symp = pdx[pdx.stratum >= 1].pred.mean()
    return g, {
        "acc": float((y == p).mean()),
        "auc": float(roc_auc_score(y, g.prob)),
        "mcc": float(matthews_corrcoef(y, p)),
        "sens": float(p[y == 1].mean()),
        "spec": float(1 - p[y == 0].mean()),
        "hy2": float(pdx[pdx.hy == 2].pred.mean()),
        "hy34": float(pdx[pdx.hy >= 3].pred.mean()),
        "u0": float(pdx[pdx.stratum == 0].pred.mean()),
        "SIS": float(pdx[pdx.stratum == 0].pred.mean() / symp) if symp else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ReadText", choices=config.TASKS)
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "mlp"])
    args = ap.parse_args()

    meta = labels.build_table(verbose=False)
    X, md = S.build_ssl_frame(meta, args.task)
    print(f"{args.task}: embeddings {X.shape}  ({md.subject.nunique()} subjects)\n")

    base = BASELINE[args.task]
    rows = []
    for layer in range(X.shape[1]):
        for cmvn in (False, True):
            Xl = X[:, layer, :]
            if cmvn:
                Xl = S.speaker_cmvn(Xl, md.subject.values)
            _, m = losocv_ssl(Xl, md, args.classifier)
            m.update(layer=layer, cmvn=cmvn,
                     d_acc=m["acc"] - base["acc"],
                     d_hy2=m["hy2"] - base["hy2"])
            rows.append(m)
            print(f"layer {layer:2d}  cmvn={str(cmvn):5s}  acc {m['acc']:.3f} "
                  f"({m['d_acc']:+.3f})  AUC {m['auc']:.3f}  MCC {m['mcc']:.3f}  "
                  f"H&Y2 {m['hy2']:.3f} ({m['d_hy2']:+.3f})", flush=True)

    t = pd.DataFrame(rows)[
        ["layer", "cmvn", "acc", "d_acc", "auc", "mcc", "sens", "spec",
         "hy2", "d_hy2", "hy34", "u0", "SIS"]]
    t.to_csv(config.WORK_DIR / f"ssl_sweep_{args.task}_{args.classifier}.csv",
             index=False)

    print("\n=== best by accuracy ===")
    print(t.sort_values("acc", ascending=False).head(6)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== best by early-stage (H&Y 2) recall ===")
    print(t.sort_values("hy2", ascending=False).head(6)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nhand-crafted baseline: acc {base['acc']:.3f}  H&Y2 {base['hy2']:.3f}")


if __name__ == "__main__":
    main()
