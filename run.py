"""
run.py
======

One command produces every number, table and figure in the paper.

    python run.py                      # everything (slow: ~2-4 h on CPU)
    python run.py --task ReadText      # one speech task only
    python run.py --quick              # skip SHAP and the leakage study
    python run.py --skip-shap          # SHAP is the slowest single step

Pipeline, in order
------------------
    1. Parse filenames -> labels, and run the dataset integrity checks.
       If this step reports anything unexpected, STOP. Every number
       downstream is meaningless if the labels are wrong.

    2. Segment the audio and extract 60 named acoustic features per segment.
       Cached to parquet, so this cost is paid once.

    3. For each model arm, run leave-one-subject-out cross-validation and
       compute severity-stratified metrics with bootstrap CIs.

    4. Significance tests against the diagnosis-only baseline.

    5. The leakage study: segment-level vs subject-level splits.

    6. Cross-task generalisation.

    7. SHAP attributions split by severity stratum.

    8. Write every figure and LaTeX table to work/.

Runtime warning
---------------
LOSOCV means 37 folds. With 5 arms and 2 tasks that is 370 model fits. The
networks are small (about 25k parameters) so each fit is seconds, not minutes,
but budget a few hours for a full run. Use --quick while developing.
"""

import argparse
import warnings

import pandas as pd

import config
import evaluate
import experiments
import features
import figures
import labels
import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# Each entry is one row of the ablation table. The arms are ordered so that
# each line adds exactly one component to the line above it -- that is what
# makes an ablation interpretable.
ARMS = [
    # Reproduces the published protocol (Neurology International 2025), so the
    # proposed model is compared against a fair reference rather than a
    # deliberately weak one.
    ("SVM (published protocol)",
     dict(model="svm")),

    # Same features, neural encoder, single head. Isolates the effect of
    # swapping the classifier from the effect of our auxiliary heads.
    ("MLP, diagnosis only",
     dict(model="samtn", use_severity=False, use_adversary=False)),

    # + the ordinal severity-regression head.
    ("+ severity head",
     dict(model="samtn", use_severity=True, use_adversary=False)),

    # + the speaker-adversarial head (gradient reversal).
    ("+ speaker-adversarial head",
     dict(model="samtn", use_severity=False, use_adversary=True)),

    # Both auxiliary heads: the full proposed model.
    ("SAMTN (both)",
     dict(model="samtn", use_severity=True, use_adversary=True)),
]

# Every significance test compares against this arm.
BASELINE_ARM = "MLP, diagnosis only"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=config.TASKS, default=None,
                    help="run a single speech task instead of both")
    ap.add_argument("--skip-shap", action="store_true",
                    help="skip interpretability (the slowest step)")
    ap.add_argument("--skip-leakage", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="shorthand for --skip-shap --skip-leakage")
    args = ap.parse_args()
    if args.quick:
        args.skip_shap = args.skip_leakage = True

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 -- labels and integrity checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1  labels and dataset integrity")
    print("=" * 70)
    meta = labels.build_table()
    meta.to_csv(config.WORK_DIR / "mdvr_kcl_labels.csv", index=False)

    print("\nTable 1 (cohort):")
    print(figures.table1_cohort(meta).to_string(index=False))
    figures.fig_severity_decoupling(meta)
    print("wrote Figure 1 (severity decoupling)")

    tasks = [args.task] if args.task else list(config.TASKS)
    ablation_rows, crosstask_rows, sig_rows = [], [], []
    frames = {}

    for task in tasks:
        print("\n" + "=" * 70)
        print(f"TASK: {task}")
        print("=" * 70)

        # --------------------------------------------------------------
        # STEP 2 -- features
        # --------------------------------------------------------------
        df = features.build_feature_frame(meta, task)
        frames[task] = df
        print(f"segments {len(df)}   subjects {df.subject.nunique()}")
        print(df.groupby("label").size().rename({0: "HC", 1: "PD"}).to_string())

        # --------------------------------------------------------------
        # STEP 3 -- LOSOCV for every arm
        # --------------------------------------------------------------
        preds_by_arm = {}
        for name, kw in ARMS:
            print(f"\n--- {name} ---")
            subj = evaluate.losocv(df, **kw)
            preds_by_arm[name] = subj

            res = evaluate.report(subj, name=f"{task} | {name}")
            # Bootstrap CIs: with 37 subjects a bare point estimate is not
            # defensible, and reviewers will ask for these anyway.
            res.update(stats.bootstrap_ci(subj.label, subj.pred, subj.prob))
            res["task"], res["arm"] = task, name

            evaluate.print_report(res)
            print(f"  95% CI accuracy {res['accuracy_ci']}   AUC {res['auc_ci']}")
            ablation_rows.append(res)

            safe = name[:12].strip().replace(" ", "_")
            subj.to_csv(config.WORK_DIR / f"subject_preds_{task}_{safe}.csv",
                        index=False)
            figures.fig_subject_probabilities(subj, task, name)

        figures.fig_roc(preds_by_arm, task)
        figures.fig_confusions(preds_by_arm, task)
        print("\nwrote Figures 2-4")

        # --------------------------------------------------------------
        # STEP 4 -- significance vs the diagnosis-only baseline
        # --------------------------------------------------------------
        print("\n--- significance tests ---")
        base = preds_by_arm[BASELINE_ARM]
        for name, s in preds_by_arm.items():
            if name == BASELINE_ARM:
                continue
            mc = stats.mcnemar(base.label.values, base.pred.values, s.pred.values)
            ab = stats.bootstrap_auc_diff(base.label.values, base.prob.values,
                                          s.prob.values)
            sig_rows.append({
                "task": task, "arm": name, "vs": BASELINE_ARM,
                "mcnemar_b01": mc["b01"], "mcnemar_b10": mc["b10"],
                "mcnemar_p": mc["p_value"],
                "auc_diff": ab["auc_diff"],
                "auc_diff_ci_low": ab["ci"][0], "auc_diff_ci_high": ab["ci"][1],
                "auc_diff_p": ab["p_two_sided"],
            })
            print(f"  {name:30s} McNemar p={mc['p_value']:.3f}   "
                  f"dAUC {ab['auc_diff']:+.3f} {ab['ci']}")

        # --------------------------------------------------------------
        # STEP 5 -- leakage study
        # --------------------------------------------------------------
        if not args.skip_leakage:
            print("\n--- leakage study: segment-level vs subject-level ---")
            leak = experiments.leakage_comparison(df, ARMS)
            leak["task"] = task
            figures.fig_leakage(leak, task)
            print("wrote Figure 5 + Table 3")

        # --------------------------------------------------------------
        # STEP 7 -- interpretability
        # --------------------------------------------------------------
        if not args.skip_shap:
            print("\n--- SHAP attributions by severity stratum ---")
            import explain
            tab = explain.shap_by_stratum(df, out_dir=config.WORK_DIR)
            print(tab.head(12).to_string())

    # ------------------------------------------------------------------
    # STEP 6 -- cross-task generalisation (needs both tasks)
    # ------------------------------------------------------------------
    if len(frames) == 2:
        print("\n" + "=" * 70)
        print("cross-task generalisation")
        print("=" * 70)
        ct = experiments.cross_task(frames["ReadText"],
                                    frames["SpontaneousDialogue"], ARMS)
        crosstask_rows = ct.to_dict("records")
        figures.table5_crosstask(ct)
        print("wrote Table 5")

    # ------------------------------------------------------------------
    # STEP 8 -- final tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 2  main ablation")
    print("=" * 70)
    t2 = figures.table2_ablation(ablation_rows)
    print(t2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if sig_rows:
        print("\nTABLE 4  significance")
        print(figures.table4_significance(sig_rows)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nAll outputs under {config.WORK_DIR.resolve()}")
    print("  figures/   PNG + PDF")
    print("  tables/    CSV + LaTeX")


if __name__ == "__main__":
    main()
