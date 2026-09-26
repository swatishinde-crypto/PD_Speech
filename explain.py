"""Interpretability, reported separately for the asymptomatic subgroup.

The headline figure of the paper comes from here: what does the model hear
in the 7 patients whose speech a clinician rated as normal? If those
attributions concentrate on timing features (pause_ratio, voiced_ratio) while
the symptomatic group's concentrate on phonatory ones (jitter, shimmer, HNR),
that is a concrete, falsifiable claim about an early marker -- and it is
stated in vocabulary a neurologist can check.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import models


def shap_by_stratum(df, n_background=200, out_dir=None):
    import shap

    out_dir = out_dir or config.WORK_DIR
    feat_cols = [c for c in df.columns
                 if c not in {"subject", "label", "severity", "stratum", "hy", "seg"}]

    scaler = StandardScaler().fit(df[feat_cols].values)
    X = scaler.transform(df[feat_cols].values)
    local = {s: i for i, s in enumerate(sorted(df.subject.unique()))}

    # Explanations are for the model as a whole, so this one is fit on all
    # subjects. It is never used to produce an accuracy number.
    net = models.train_samtn(
        X, df.label.values.astype(np.float32), df.severity.values.astype(np.float32),
        df.subject.map(local).values, n_speakers=len(local),
    )

    rng = np.random.default_rng(config.SEED)
    bg = X[rng.choice(len(X), min(n_background, len(X)), replace=False)]

    def f(a):
        p, _ = models.predict(net, a.astype(np.float32))
        return p

    explainer = shap.KernelExplainer(f, shap.kmeans(bg, 25))

    rows = []
    for stratum, tag in [(0, "asymptomatic_PD"), (1, "mild_PD"), (2, "moderate_severe_PD")]:
        sel = df[(df.label == 1) & (df.stratum == stratum)]
        if sel.empty:
            continue
        idx = sel.sample(min(120, len(sel)), random_state=config.SEED).index
        vals = explainer.shap_values(X[df.index.get_indexer(idx)], nsamples=200, silent=True)
        mean_abs = np.abs(np.asarray(vals)).mean(0)
        rows.append(pd.Series(mean_abs, index=feat_cols, name=tag))

    table = pd.DataFrame(rows).T
    table = table.sort_values(table.columns[0], ascending=False)
    table.to_csv(out_dir / "shap_by_severity.csv")

    top = table.head(15)
    ax = top.plot.barh(figsize=(9, 7))
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Which acoustic markers drive the prediction, by rated severity")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_by_severity.png", dpi=180)
    plt.close()
    return table
