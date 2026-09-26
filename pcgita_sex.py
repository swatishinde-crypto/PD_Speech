"""Accuracy by sex on PC-GITA.

MDVR-KCL is 19/21 male among controls, so principle R6 could only be
implemented by severity there. PC-GITA is balanced (50 women, 50 men, mean
age 61 in both groups), so the same metrics are reported within each sex.
"""

import warnings

import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import stats

warnings.filterwarnings("ignore")

W = "work_pcgita"
meta = pd.read_csv(f"{W}/pcgita_meta.csv").drop_duplicates("subject")[["subject", "sex"]]
rows = []
for task in ("ReadText", "Monologue"):
    for name, f in (("hand-crafted (60-d)", "hand"), ("wav2vec2 layer 6", "w2v")):
        g = pd.read_csv(f"{W}/preds_{f}_{task}.csv").merge(meta, on="subject")
        for sex, s in g.groupby("sex"):
            ci = stats.bootstrap_ci(s.label, s.pred, s.prob)
            rows.append({
                "task": task, "model": name, "sex": sex, "n": len(s),
                "acc": (s.label == s.pred).mean(),
                "acc_lo": ci["accuracy_ci"][0], "acc_hi": ci["accuracy_ci"][1],
                "auc": roc_auc_score(s.label, s.prob),
                "mcc": matthews_corrcoef(s.label, s.pred),
                "sens": s.pred[s.label == 1].mean(),
                "spec": 1 - s.pred[s.label == 0].mean()})

out = pd.DataFrame(rows)
out.to_csv(f"{W}/by_sex.csv", index=False)
print(out.round(3).to_string(index=False))
