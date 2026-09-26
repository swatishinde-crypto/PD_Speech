"""Parse MDVR-KCL filenames into a subject/label table, with integrity checks.

Filename scheme (dataset description):
    SI_HS_HYR_UPDRSII-5_UPDRSIII-18.wav     e.g. ID02_pd_2_0_0.wav

Quirks handled explicitly:
  * ID22hc_0_0_0.wav -- missing underscore after the subject ID
  * ID18 present in ReadText, absent from SpontaneousDialogue
"""

import re

import pandas as pd

import config

# underscore between ID and status is optional -- handles ID22hc_...
PATTERN = re.compile(
    r"^(?P<sid>ID\d+)_?(?P<status>hc|pd)_(?P<hy>\d+)_(?P<u2>\d+)_(?P<u3>\d+)$",
    re.IGNORECASE,
)


def _parse(path):
    m = PATTERN.match(path.stem)
    if not m:
        raise ValueError(f"unparseable filename: {path.name}")
    g = m.groupdict()
    return {
        "subject": g["sid"].upper(),
        "task": path.parent.parent.name,
        "folder": path.parent.name.lower(),
        "label": int(g["status"].lower() == "pd"),
        "hy": int(g["hy"]),
        "updrs2_5": int(g["u2"]),
        "updrs3_18": int(g["u3"]),
        "path": str(path),
    }


def build_table(root=None, verbose=True):
    root = root or config.DATA_ROOT
    rows, bad = [], []
    for wav in sorted(root.rglob("*.wav")):
        try:
            rows.append(_parse(wav))
        except ValueError as e:
            bad.append(str(e))
    if not rows:
        raise FileNotFoundError(f"no .wav files under {root.resolve()}")

    df = pd.DataFrame(rows)
    df["severity"] = df["updrs2_5"]
    # Stratum used for severity-stratified reporting; 2 and 3 are pooled
    # because only 3 patients sit above 1.
    df["stratum"] = df["updrs2_5"].clip(upper=2)

    if verbose:
        _report(df, bad)
    return df


def _report(df, bad):
    print(f"files parsed : {len(df)}  (expect 73)")
    print(f"unparseable  : {len(bad)}")
    for b in bad:
        print("   !", b)

    mismatch = df[df["label"] != (df["folder"] == "pd").astype(int)]
    print(f"folder/name label mismatches: {len(mismatch)}")

    subs = df.drop_duplicates("subject").set_index("subject")
    print(f"unique subjects: {len(subs)}  (expect 37)")
    print(f"  HC {int((subs.label == 0).sum())}   PD {int((subs.label == 1).sum())}")

    pdx = subs[subs.label == 1]
    print("\nPD UPDRS-II-5 distribution (expect 7/6/2/1):")
    print(pdx.updrs2_5.value_counts().sort_index().to_string())
    print("\nPD H&Y distribution (note: no stage 1):")
    print(pdx.hy.value_counts().sort_index().to_string())

    odd = subs[(subs.label == 0) & (subs[["hy", "updrs2_5", "updrs3_18"]].sum(axis=1) > 0)]
    print(f"\nnon-zero healthy controls: {list(odd.index)}")

    per_task = df.groupby("task")["subject"].apply(set)
    if len(per_task) == 2:
        a, b = per_task
        print(f"subjects missing from one task: {sorted(a ^ b)}")

    # Rater disagreement: UPDRS-II-5 vs UPDRS-III-18.
    dis = subs[subs.updrs2_5 != subs.updrs3_18]
    print(f"rater-disagreement subjects: {list(dis.index)}")


if __name__ == "__main__":
    build_table().to_csv("mdvr_kcl_labels.csv", index=False)
    print("\nwrote mdvr_kcl_labels.csv")
