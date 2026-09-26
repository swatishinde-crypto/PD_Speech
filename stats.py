"""Statistical support. With 37 subjects, point estimates alone are not
publishable -- every headline number needs an interval, and every claimed
improvement needs a test.
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import config


def bootstrap_ci(y, pred, prob=None, n_boot=2000, alpha=0.05, seed=config.SEED):
    """Subject-level bootstrap CIs for accuracy, AUC, MCC, sens, spec."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    prob = np.asarray(prob) if prob is not None else None
    n = len(y)
    acc, auc, mcc, sen, spe = [], [], [], [], []

    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2:
            continue
        acc.append((y[i] == pred[i]).mean())
        mcc.append(matthews_corrcoef(y[i], pred[i]))
        sen.append(pred[i][y[i] == 1].mean())
        spe.append(1 - pred[i][y[i] == 0].mean())
        if prob is not None:
            auc.append(roc_auc_score(y[i], prob[i]))

    def ci(v):
        if not len(v):
            return (np.nan, np.nan)
        return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))

    return {"accuracy_ci": ci(acc), "auc_ci": ci(auc), "mcc_ci": ci(mcc),
            "sensitivity_ci": ci(sen), "specificity_ci": ci(spe)}


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test -- did arm B change the decisions arm A got wrong?

    Exact binomial rather than chi-square: the discordant counts here are
    small enough that the asymptotic approximation is not safe.
    """
    from scipy.stats import binomtest

    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    b01 = int(((a == y) & (b != y)).sum())   # A right, B wrong
    b10 = int(((a != y) & (b == y)).sum())   # A wrong, B right
    if b01 + b10 == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    p = binomtest(b10, b01 + b10, 0.5).pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p)}


def bootstrap_auc_diff(y, prob_a, prob_b, n_boot=2000, seed=config.SEED):
    """Paired bootstrap on the AUC difference (B - A)."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(prob_a), np.asarray(prob_b)
    n, diffs = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    diffs = np.array(diffs)
    return {
        "auc_diff": float(diffs.mean()),
        "ci": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


def severity_correlation(subj_true, subj_pred_severity):
    """Does the severity head actually track UPDRS-II-5? Spearman + CI."""
    from scipy.stats import spearmanr

    rho, p = spearmanr(subj_true, subj_pred_severity)
    return {"spearman_rho": float(rho), "p_value": float(p)}
