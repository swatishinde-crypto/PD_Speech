"""
sweep.py
========

Establish the honest best baseline before drawing any conclusion from the
proposed model.

The published LOSOCV reference on this dataset (Neurology International 2025)
reports 95.45% on ReadText using acoustic + GTCC features with a tuned SVM.
Our first pass reached only ~0.70 with an untuned SVM on all 60 features.
Two things could explain that gap:

  1. no hyperparameter tuning
  2. the wrong feature subset -- "all features" is not what they used

This script varies exactly those two factors, holding the evaluation protocol
(leave-one-subject-out, subject-grouped inner tuning folds) fixed. If a tuned
SVM on acoustic+GTCC reaches the published range, our pipeline is sound and
the earlier number was a tuning artefact. If it does not, the difference is
in feature extraction and we go looking there instead.

    python sweep.py --task ReadText
"""

import argparse
import time

import pandas as pd

import config
import evaluate
import features
import labels

FEATURE_SETS = ["all", "acoustic+gtcc", "acoustic+mfcc", "cepstral", "acoustic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ReadText", choices=config.TASKS)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    meta = labels.build_table(verbose=False)
    df = features.build_feature_frame(meta, args.task)
    print(f"{args.task}: {len(df)} segments, {df.subject.nunique()} subjects\n")

    rows = []
    for fs in FEATURE_SETS:
        for tune in ([False] if args.no_tune else [False, True]):
            t0 = time.time()
            subj = evaluate.losocv(df, model="svm", feature_set=fs, tune=tune)
            res = evaluate.report(subj, name=f"svm|{fs}|tune={tune}")
            rows.append({
                "feature_set": fs,
                "n_features": subj.attrs["n_features"],
                "tuned": tune,
                "subject_acc": res["accuracy"],
                "segment_acc": subj.attrs["segment_accuracy"],
                "auc": res["auc"],
                "mcc": res["mcc"],
                "sens": res["sensitivity"],
                "spec": res["specificity"],
                "recall_updrs0": res["recall_updrs0_asymptomatic"],
                "sis": res["severity_independence_score"],
                "ID31_prob": res.get("ID31_prob_pd"),
                "secs": round(time.time() - t0),
            })
            r = rows[-1]
            print(f"{fs:15s} n={r['n_features']:2d} tuned={str(tune):5s} "
                  f"subj_acc {r['subject_acc']:.3f}  seg_acc {r['segment_acc']:.3f}  "
                  f"AUC {r['auc']:.3f}  SIS {r['sis']:.3f}  ({r['secs']}s)")

    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.WORK_DIR / f"sweep_{args.task}.csv", index=False)
    print("\n=== sorted by AUC ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
