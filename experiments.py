"""
experiments.py
==============

The two extra experiments that turn a results table into an argument.

1. `leakage_comparison`  -- runs the SAME features and the SAME model twice,
   once with a random segment-level split (wrong, what much of the literature
   does) and once with leave-one-subject-out (correct). The difference between
   the two accuracies is a direct measurement of speaker leakage.

2. `cross_task`          -- trains on the read-aloud recordings and tests on
   the spontaneous conversation recordings, and vice versa. Checks whether
   the model learned something about the speaker's motor control, or merely
   about a fixed passage of text everyone read.

Both are cheap to run and both anticipate the first questions a reviewer asks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
import models


def _feature_columns(df):
    """Everything that is not metadata is a feature.

    Kept in one place so that adding a metadata column later cannot silently
    leak a label into the feature matrix -- a classic and invisible bug.
    """
    meta = {"subject", "label", "severity", "stratum", "hy", "seg"}
    return [c for c in df.columns if c not in meta]


def _fit_predict(Xtr, tr, Xte, model, use_severity, use_adversary):
    """Train one model on a fold and return test-set probabilities."""
    if model == "svm":
        clf = models.make_svm_baseline().fit(Xtr, tr.label.values)
        return clf.predict_proba(Xte)[:, 1]

    # The speaker-adversarial head needs speaker ids in a contiguous range
    # 0..n-1, and which subjects are present changes from fold to fold, so
    # the mapping has to be rebuilt every time.
    local = {s: i for i, s in enumerate(sorted(tr.subject.unique()))}
    net = models.train_samtn(
        Xtr,
        tr.label.values.astype(np.float32),
        tr.severity.values.astype(np.float32),
        tr.subject.map(local).values,
        n_speakers=len(local),
        use_severity=use_severity,
        use_adversary=use_adversary,
    )
    prob, _ = models.predict(net, Xte)
    return prob


# ==========================================================================
# EXPERIMENT 1 -- how much of the published accuracy is speaker leakage?
# ==========================================================================
def leakage_comparison(df, arms, n_splits=5):
    """Segment-level (leaky) vs subject-level (honest) cross-validation.

    The ONLY difference between the two conditions is the splitter:

        StratifiedKFold       ignores `subject`. A subject's segments can be
                              split across train and test. Leaky.

        StratifiedGroupKFold  groups by `subject`, so every segment from a
                              given person stays on one side of the split.
                              Honest.

    Everything else -- features, scaling, model, hyperparameters, seed -- is
    held identical, which is what makes the comparison a clean measurement
    rather than a confound.

    Note we use 5-fold grouped CV here rather than full leave-one-subject-out,
    purely so the two conditions have a comparable number of training subjects
    per fold. The headline results in evaluate.py still use LOSOCV.

    Returns a long-format DataFrame: split, arm, accuracy.
    """
    feat = _feature_columns(df)
    X_all, y_all, g_all = df[feat].values, df.label.values, df.subject.values
    rows = []

    for arm_name, kw in arms:
        model = kw.get("model", "samtn")
        use_sev = kw.get("use_severity", True)
        use_adv = kw.get("use_adversary", True)

        for split_name, splitter, groups in [
            ("segment-level (leaky)",
             StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED),
             None),
            ("subject-level (honest)",
             StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED),
             g_all),
        ]:
            correct, total = 0, 0
            for tr_i, te_i in splitter.split(X_all, y_all, groups):
                tr, te = df.iloc[tr_i], df.iloc[te_i]

                # Fit the scaler on TRAINING data only. Fitting it on
                # everything is itself a form of leakage.
                sc = StandardScaler().fit(tr[feat].values)
                prob = _fit_predict(sc.transform(tr[feat].values), tr,
                                    sc.transform(te[feat].values),
                                    model, use_sev, use_adv)

                # Scored at SEGMENT level in both conditions, deliberately:
                # aggregating to subjects would partly mask the leakage we
                # are trying to measure.
                correct += int(((prob >= 0.5).astype(int) == te.label.values).sum())
                total += len(te)

            rows.append({"split": split_name, "arm": arm_name,
                         "accuracy": correct / total})
            print(f"  {arm_name:32s} {split_name:24s} acc {correct / total:.3f}")

    return pd.DataFrame(rows)


# ==========================================================================
# EXPERIMENT 2 -- does it transfer between speech tasks?
# ==========================================================================
def cross_task(df_read, df_spont, arms):
    """Train on one speech task, evaluate on the other.

    Why this matters. In the read-aloud task every participant reads the same
    passage ("The North Wind and the Sun"). A model could score well simply by
    comparing renditions of identical text -- which would not generalise to a
    patient talking naturally, and so would be clinically worthless.

    If performance holds up when we train on read speech and test on
    spontaneous speech, whatever the model found is a property of the
    speaker rather than of the script.

    Important detail: there is no subject overlap to worry about here in the
    usual sense -- the same 37 people appear in both tasks -- so this is NOT a
    subject-independent test. It is a *task*-independent test, and we say so
    explicitly in the paper rather than overselling it. (ID18 appears only in
    the read task, which is handled automatically since we simply use whatever
    rows each frame contains.)

    Returns a DataFrame: arm, train_task, test_task, accuracy, subject accuracy.
    """
    from sklearn.metrics import roc_auc_score

    feat = _feature_columns(df_read)
    rows = []

    for train_df, test_df, tr_name, te_name in [
        (df_read, df_spont, "ReadText", "SpontaneousDialogue"),
        (df_spont, df_read, "SpontaneousDialogue", "ReadText"),
    ]:
        sc = StandardScaler().fit(train_df[feat].values)
        Xtr = sc.transform(train_df[feat].values)
        Xte = sc.transform(test_df[feat].values)

        for arm_name, kw in arms:
            prob = _fit_predict(
                Xtr, train_df, Xte,
                kw.get("model", "samtn"),
                kw.get("use_severity", True),
                kw.get("use_adversary", True),
            )

            seg_acc = float(((prob >= 0.5).astype(int) == test_df.label.values).mean())

            # Aggregate segments to one decision per subject -- this is the
            # number that belongs in the paper.
            agg = (pd.DataFrame({"subject": test_df.subject.values,
                                 "label": test_df.label.values,
                                 "prob": prob})
                   .groupby("subject")
                   .agg(prob=("prob", "mean"), label=("label", "first")))
            subj_acc = float(((agg.prob >= 0.5).astype(int) == agg.label).mean())
            subj_auc = (float(roc_auc_score(agg.label, agg.prob))
                        if agg.label.nunique() > 1 else float("nan"))

            rows.append({
                "arm": arm_name, "train_task": tr_name, "test_task": te_name,
                "segment_accuracy": seg_acc, "subject_accuracy": subj_acc,
                "subject_auc": subj_auc,
            })
            print(f"  {arm_name:32s} {tr_name[:9]:>9s} -> {te_name[:9]:<9s} "
                  f"subj acc {subj_acc:.3f}  AUC {subj_auc:.3f}")

    return pd.DataFrame(rows)
