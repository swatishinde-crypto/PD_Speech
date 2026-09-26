"""
figures.py
==========

Produces EVERY figure and table the paper needs, in publication-ready form.

Output layout
-------------
    work/figures/*.png    300 dpi, for drafts and slides
    work/figures/*.pdf    vector, for the actual submission
    work/tables/*.csv     for you to inspect in Excel
    work/tables/*.tex     LaTeX, paste straight into the manuscript

What gets produced, and where it goes in the paper
--------------------------------------------------
    Table 1   cohort description                -> Section "Dataset"
    Figure 1  severity decoupling               -> Section "Dataset" (motivation)
    Table 2   main ablation with CIs            -> Section "Results"
    Figure 2  per-subject probabilities         -> Section "Results" (KEY FIGURE)
    Figure 3  ROC curves                        -> Section "Results"
    Figure 4  confusion matrices                -> Section "Results"
    Figure 5 + Table 3  leakage demonstration   -> Section "Results"
    Table 4   significance tests                -> Section "Results"
    Table 5   cross-task generalisation         -> Section "Results"
    (SHAP figure comes from explain.py)         -> Section "Interpretability"

Style note: we use a single consistent colour for HC (blue) and PD (red)
across every figure. Reviewers notice inconsistent colour coding.
"""

import matplotlib

# "Agg" = a non-interactive backend. This MUST be set before importing pyplot.
# It lets the script save figures on a server or in Colab where there is no
# screen to draw on. Without it the script crashes in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

import config

# "paper" context and a modest font scale produce figures that stay legible
# when a journal shrinks them to one column width.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Fixed colours, used everywhere. Blue = healthy, red = disease.
PALETTE = {"HC": "#4C78A8", "PD": "#E45756"}


# ==========================================================================
# small helpers
# ==========================================================================
def _dirs():
    """Create (and return) the output folders. Safe to call repeatedly."""
    fig = config.WORK_DIR / "figures"
    tab = config.WORK_DIR / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab


def _save(fig_obj, name):
    """Save one figure as both PNG (drafts) and PDF (submission)."""
    figs, _ = _dirs()
    for ext in ("png", "pdf"):
        fig_obj.savefig(figs / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig_obj)          # close it, or memory fills up over many figures


def _latex(df, name, caption, label):
    """Write one table as CSV (for you) and LaTeX (for the manuscript).

    `label` becomes the \\label{...} you cite with \\ref{...} in the text.
    Note the doubled backslashes in captions: LaTeX needs \\& for an ampersand.
    """
    _, tabs = _dirs()
    df.to_csv(tabs / f"{name}.csv", index=False)
    with open(tabs / f"{name}.tex", "w", encoding="utf-8") as fh:
        fh.write(df.to_latex(index=False, float_format="%.3f",
                             caption=caption, label=label, escape=True))


# ==========================================================================
# TABLE 1 -- who is in the dataset
# ==========================================================================
def table1_cohort(meta):
    """Standard cohort table. Every clinical paper opens with one of these.

    Two deliberate choices worth defending to a reviewer:

    1. We report H&Y as median [min-max], which makes it immediately visible
       that the minimum PD stage is 2 -- i.e. this dataset contains NO
       early-stage Parkinson's, despite the dataset title saying it does.

    2. We break out the UPDRS-II-5 counts individually rather than pooling
       them, because the count in the "0" column (seven patients) is the
       entire premise of the paper.

    Participant sex is NOT distributed with the recordings -- it appears only
    as an aggregate in the dataset authors' own description (19 of 21 controls
    male). We say so in the caption rather than silently omitting it.
    """
    # drop_duplicates: each subject appears twice in `meta` (once per speech
    # task), and a cohort table counts people, not recordings.
    subs = meta.drop_duplicates("subject").set_index("subject")

    rows = []
    for grp, tag in [(0, "Healthy control"), (1, "Parkinsons disease")]:
        g = subs[subs.label == grp]
        rows.append({
            "Group": tag,
            "n": len(g),
            "H&Y (median [min-max])": (
                "0" if grp == 0
                else f"{g.hy.median():.0f} [{g.hy.min()}-{g.hy.max()}]"
            ),
            "UPDRS-II-5 = 0": int((g.updrs2_5 == 0).sum()),
            "UPDRS-II-5 = 1": int((g.updrs2_5 == 1).sum()),
            "UPDRS-II-5 = 2": int((g.updrs2_5 == 2).sum()),
            "UPDRS-II-5 = 3": int((g.updrs2_5 == 3).sum()),
        })

    t = pd.DataFrame(rows)
    _latex(
        t, "table1_cohort",
        "Cohort description. No PD participant is staged below H\\&Y 2, and "
        "seven PD participants carry a UPDRS-II-5 speech rating of 0. "
        "Participant sex is not distributed with the recordings and is "
        "reported only in aggregate by the dataset authors (19/21 controls "
        "male).",
        "tab:cohort",
    )
    return t


# ==========================================================================
# FIGURE 1 -- the motivating observation (no model involved)
# ==========================================================================
def fig_severity_decoupling(meta):
    """Scatter: Hoehn & Yahr stage vs rated speech impairment, PD only.

    This figure needs no model at all -- it is pure dataset description, which
    makes it very hard to argue with. It shows that how advanced someone's
    Parkinson's is tells you remarkably little about how impaired their speech
    is. ID20 sits at stage 3 with a speech rating of 0; ID30 sits at stage 2
    with a rating of 1.

    Why this matters for the paper: if a classifier reports 99% accuracy while
    disease stage and voice impairment are this weakly coupled, the classifier
    cannot plausibly be reading disease severity out of the voice. It is
    reading something else -- and our hypothesis is speaker identity.

    Put this figure early, in the Dataset section. It sets up everything.
    """
    subs = meta.drop_duplicates("subject")
    pdx = subs[subs.label == 1]

    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Both axes are small integers, so many subjects land on the exact same
    # point and would hide each other. A little random jitter separates them.
    # Seeded, so the figure is identical every time you regenerate it.
    jit = np.random.default_rng(config.SEED).normal(0, 0.06, len(pdx))

    ax.scatter(pdx.hy + jit, pdx.updrs2_5 + jit,
               s=70, c=PALETTE["PD"], edgecolor="white", zorder=3)

    # Label every point with its subject ID so readers can trace the specific
    # cases we discuss in the text (ID20, ID27, ID30).
    for r, j in zip(pdx.itertuples(), jit):
        ax.annotate(r.subject, (r.hy + j, r.updrs2_5 + j),
                    fontsize=7, xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Hoehn & Yahr stage")
    ax.set_ylabel("UPDRS-II-5 (speech)")
    ax.set_title("Disease stage and rated speech impairment are only\n"
                 "loosely coupled (PD participants, n=16)")
    ax.set_xticks([2, 3, 4])
    ax.set_yticks([0, 1, 2, 3])
    _save(fig, "fig1_severity_decoupling")


# ==========================================================================
# FIGURE 2 -- THE KEY FIGURE OF THE PAPER
# ==========================================================================
def fig_subject_probabilities(subj, task, arm):
    """One dot per subject: predicted p(PD), grouped by rated speech severity.

    This is the figure the paper lives or dies on. Read it left to right:

      * Leftmost column: the 21 healthy controls. Should sit low.
      * Then PD patients rated 0, 1, 2, 3 on speech impairment.

    If the dots climb steadily from left to right, the model is tracking
    audible voice degradation -- and the UPDRS-0 patients will sit down among
    the controls, i.e. the model is useless for exactly the patients where
    early detection would matter.

    If instead the UPDRS-0 patients sit high, clearly separated from the
    controls, the model has found something clinicians cannot hear. That is
    a genuine early-marker result.

    Either outcome is publishable. That is why this experiment is safe to run.

    ID31 -- a healthy control the clinicians rated as having degraded speech --
    is annotated by name, because where that single dot falls is itself an
    argument about what the model is measuring.
    """
    d = subj.copy()
    d["group"] = np.where(d.label == 1, "PD", "HC")

    # Controls get their own x position (-1) so they form a separate column
    # rather than being mixed into the "severity 0" group with PD patients.
    d["x"] = np.where(d.label == 0, -1, d.severity)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sns.stripplot(data=d, x="x", y="prob", hue="group", palette=PALETTE,
                  size=9, jitter=0.22, ax=ax, edgecolor="white", linewidth=0.6)

    # The decision threshold. Anything above this dot line is called "PD".
    ax.axhline(0.5, ls="--", c="grey", lw=1)

    labels_all = ["HC", "PD\nUPDRS 0", "PD\nUPDRS 1", "PD\nUPDRS 2", "PD\nUPDRS 3"]
    ax.set_xticks(range(d.x.nunique()))
    ax.set_xticklabels(labels_all[:d.x.nunique()])
    ax.set_xlabel("")
    ax.set_ylabel("predicted p(PD)")
    ax.set_title(f"{task} - {arm}\n"
                 "per-subject predictions by rated speech severity")

    special = d[d.subject == config.CONTROL_WITH_DEGRADATION]
    if len(special):
        ax.annotate(f"{config.CONTROL_WITH_DEGRADATION}\n(HC rated 0_1_1)",
                    (0, special.prob.iloc[0]), fontsize=8, color="black",
                    xytext=(18, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    safe_arm = arm[:12].strip().replace(" ", "_").replace("(", "").replace(")", "")
    _save(fig, f"fig2_subject_probs_{task}_{safe_arm}")


# ==========================================================================
# FIGURE 3 -- ROC curves
# ==========================================================================
def fig_roc(preds_by_arm, task):
    """Overlaid ROC curves, one per model arm.

    ROC is computed at SUBJECT level, not segment level. That distinction is
    the whole methodological point of this paper, so it belongs in the caption
    as well as in the code.

    With 37 subjects an ROC curve is visibly stepped -- that is honest, not a
    bug. Do not smooth it.

    `preds_by_arm` : {arm name -> subject-level DataFrame with .label, .prob}
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for arm, s in preds_by_arm.items():
        fpr, tpr, _ = roc_curve(s.label, s.prob)
        ax.plot(fpr, tpr, lw=1.9, label=f"{arm} (AUC {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)   # chance line
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"Subject-level ROC, LOSOCV - {task}")
    ax.legend(fontsize=7.5, loc="lower right")
    _save(fig, f"fig3_roc_{task}")


# ==========================================================================
# FIGURE 4 -- confusion matrices
# ==========================================================================
def fig_confusions(preds_by_arm, task):
    """One confusion matrix per arm, side by side.

    With only 37 subjects these are small integer grids, which is actually an
    advantage: a reader can count the errors by eye and check our claims. A
    reviewer who wants to know "which subjects did it miss?" can cross-
    reference against the per-subject CSVs the runner writes.
    """
    n = len(preds_by_arm)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3))
    axes = np.atleast_1d(axes)        # handles the n == 1 case

    for ax, (arm, s) in zip(axes, preds_by_arm.items()):
        cm = confusion_matrix(s.label, s.pred)
        ConfusionMatrixDisplay(cm, display_labels=["HC", "PD"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(arm, fontsize=8.5)
        ax.grid(False)                # gridlines over a heatmap look wrong

    fig.suptitle(f"Confusion matrices, LOSOCV - {task}", y=1.03)
    _save(fig, f"fig4_confusion_{task}")


# ==========================================================================
# FIGURE 5 + TABLE 3 -- the leakage demonstration
# ==========================================================================
def fig_leakage(leak_df, task):
    """Same features, same model, two different ways of splitting the data.

        segment-level split : random split over audio segments. A subject's
                              segments land in BOTH train and test. This is
                              what much of the published work on this dataset
                              does, and it is how you get 99%.

        subject-level split : leave-one-subject-out. A subject appears in
                              train or test, never both. This is correct.

    The bar-height gap between the two is a direct measurement of how much
    of the reported performance was speaker memorisation. Nobody has published
    this number for MDVR-KCL. It is a contribution on its own.

    `leak_df` columns: split, arm, accuracy
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sns.barplot(data=leak_df, x="arm", y="accuracy", hue="split", ax=ax,
                palette=["#F58518", "#4C78A8"])

    # Start the y-axis at 0.4, just under chance, so the gap is readable.
    # (Starting at 0 would squash everything into the top of the panel.)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, ls=":", c="grey")      # chance level for balanced binary
    ax.set_ylabel("accuracy")
    ax.set_xlabel("")
    ax.set_title(f"Random segment splits inflate accuracy - {task}\n"
                 "the gap is speaker identity, not disease signal")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.legend(title="")
    _save(fig, f"fig5_leakage_{task}")

    _latex(leak_df, f"table3_leakage_{task}",
           "Segment-level versus subject-level evaluation using identical "
           "features and models. The inflation is the magnitude of speaker "
           "leakage.", f"tab:leakage_{task}")


# ==========================================================================
# TABLE 2 -- the main results table
# ==========================================================================
def table2_ablation(rows):
    """The paper's central table: every arm, every metric, with CIs.

    Column order is deliberate. Overall accuracy comes first because readers
    expect it, but the columns that carry the argument are the three
    per-stratum recalls and the Severity-Independence Score at the end.
    """
    t = pd.DataFrame(rows)
    keep = ["task", "arm", "accuracy", "accuracy_ci", "auc", "auc_ci", "mcc",
            "sensitivity", "specificity", "recall_updrs0_asymptomatic",
            "recall_updrs1_mild", "recall_updrs2_moderate_severe",
            "severity_independence_score", "ID31_prob_pd"]
    t = t[[c for c in keep if c in t.columns]]

    # CIs arrive as (low, high) tuples; render them as "[0.81, 0.97]".
    for c in ("accuracy_ci", "auc_ci"):
        if c in t.columns:
            t[c] = t[c].apply(
                lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"
                if isinstance(v, (tuple, list)) else v)

    _latex(t, "table2_ablation",
           "Leave-one-subject-out results with 95\\% bootstrap confidence "
           "intervals. Recall is reported separately by UPDRS-II-5 stratum; "
           "SIS is the ratio of asymptomatic to symptomatic recall.",
           "tab:ablation")
    return t


def table4_significance(rows):
    """Is the improvement real, or noise? With n=37 this table is mandatory.

    A reviewer's first question about any ablation is whether the differences
    survive a significance test. Having this table ready pre-empts a revision
    round -- and if the differences are NOT significant, we say so plainly
    and report the effect as exploratory.
    """
    t = pd.DataFrame(rows)
    _latex(t, "table4_significance",
           "Pairwise comparisons against the diagnosis-only baseline: exact "
           "McNemar tests on subject-level decisions and paired bootstrap "
           "tests on AUC.", "tab:significance")
    return t


def table5_crosstask(rows):
    """Train on reading aloud, test on free conversation -- and vice versa.

    This is a cheap but powerful robustness check. A model that only works
    when everyone reads the same fixed passage ("The North Wind and the Sun")
    may simply be comparing renditions of identical text. If performance
    survives the switch to spontaneous speech, the finding is far more likely
    to reflect the speaker's motor control rather than the script.
    """
    t = pd.DataFrame(rows)
    _latex(t, "table5_crosstask",
           "Cross-task generalisation: trained on one speech task and "
           "evaluated on the other, with no subject overlap.",
           "tab:crosstask")
    return t
