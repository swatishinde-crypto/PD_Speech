"""Central configuration. Edit DATA_ROOT and run `python run.py`."""

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
# Both can be overridden from the shell without editing this file, e.g.
#     MDVR_DATA_ROOT=synthetic_KCL MDVR_WORK_DIR=work_synth python run.py
DATA_ROOT = Path(os.environ.get("MDVR_DATA_ROOT", "26_29_09_2017_KCL"))
WORK_DIR = Path(os.environ.get("MDVR_WORK_DIR", "work"))
SEED = 1337

# --- segmentation ----------------------------------------------------------
# Matches the protocol in Neurology International 2025 (PMC13209858).
SILENCE_MIN_LEN_MS = 500      # 0.5 s of silence marks a boundary
SILENCE_THRESH_DBFS = -16
SEG_MIN_MS = 1000             # drop fragments shorter than this
SEG_MAX_MS = 8000             # split anything longer, keeps segment count balanced
TARGET_SR = 16000             # downsample; 44.1 kHz buys nothing for these features

# --- multi-task loss weights ----------------------------------------------
LAMBDA_SEVERITY = 0.3         # weight on the ordinal speech-severity head
LAMBDA_ADVERSARY = 0.2        # weight on the speaker-adversarial head
GRL_ALPHA = 1.0               # gradient-reversal strength (ramped during training)

# --- training --------------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 128
DROPOUT = 0.3

# --- evaluation ------------------------------------------------------------
TASKS = ("ReadText", "SpontaneousDialogue")
# Subjects the dataset description calls out as informative edge cases.
CONTROL_WITH_DEGRADATION = "ID31"   # hc labelled 0_1_1
MISSING_IN_SPONTANEOUS = "ID18"     # only UPDRS-II-5 == 3 patient; ReadText only
