# Speaker-Independent Parkinson's Disease Detection from Speech

Analysis code for the accompanying manuscript. Two speech representations are
compared under leave-one-subject-out cross-validation: 60 acoustic descriptors
computed with Praat and librosa, and frozen wav2vec2-base embeddings. The same
pipeline is then applied to two further corpora.

## Corpora

None of the audio is included here. Obtain each corpus from its source.

| Corpus | Source |
|---|---|
| MDVR-KCL | https://doi.org/10.5281/zenodo.2867216 (the copy used here came from the Kaggle mirror `nutansingh/mdvr-kcl-dataset`) |
| Italian Parkinson's Voice and Speech | https://doi.org/10.21227/aw6b-tg17 |
| PC-GITA | On request from the corpus authors (Orozco-Arroyave et al., LREC 2014). Not redistributable. |

Unzip MDVR-KCL and point `MDVR_DATA_ROOT` at it. Place the Italian folder
(`Italian Parkinson's Voice and speech/`) and the PC-GITA folder
(`PC-GITA_per_task_44100Hz/`) in the repository root.

## Running

```bash
pip install -r requirements.txt
export MDVR_DATA_ROOT=/path/to/26-29_09_2017_KCL

python labels.py         # parses the filenames; integrity check
python confounds.py      # recording-level baseline
python run.py            # partitioning regimes, stratified reporting
python run_ssl.py --task ReadText    # wav2vec2 layer sweep
python run_final.py      # significance and permutation tests
python fusion.py         # fusion configurations

python italian_extract.py    # external corpus 1  (~25 min, CPU)
python italian_analysis.py

python pcgita_extract.py     # external corpus 2  (~30 min, CPU)
python pcgita_analysis.py
python pcgita_sex.py
```

`labels.py` must report 73 files parsed, 0 unparseable, 37 subjects, a PD
UPDRS-II-5 distribution of 7/6/2/1, ID31 as the only non-zero control, and
ID18 absent from one task. `italian_extract.py` must report 21 control and 24
patient speakers over 93 recordings; `pcgita_extract.py` must report 50 and 50.

## Results

`results/` holds the output of the runs above: per-participant predictions,
layer sweeps, significance tests, permutation nulls and recording-level
checks, with `results/italian/` and `results/pcgita/` for the external
corpora. `python restore_results.py` copies them into the working directories
the scripts read from, so the numbers can be inspected without re-running
anything.

Speaker identifiers in the external results are codes (IT_PD01, and PC-GITA's
own recording codes); no participant names appear anywhere.

## Files

    config.py             paths, segmentation and training settings
    labels.py             filename parsing and dataset checks
    features.py           segmentation and the 60 acoustic descriptors
    ssl_features.py       wav2vec2 embedding extraction and pooling
    models.py             SVM baseline; multi-task network with gradient reversal
    evaluate.py           leave-one-subject-out evaluation, stratified metrics
    experiments.py        partitioning-regime and cross-task studies
    confounds.py          recording-level audit
    stats.py              bootstrap CIs, McNemar, permutation helpers
    fusion.py             early and nested late fusion
    sweep.py              layer and feature-set sweeps
    figures.py            result tables and figures
    explain.py            SHAP attribution by severity stratum
    run.py, run_ssl.py, run_final.py      experiment drivers
    italian_extract.py, italian_analysis.py     external corpus 1
    pcgita_extract.py, pcgita_analysis.py, pcgita_sex.py   external corpus 2
    restore_results.py    copies results/ into the working directories

## Notes

Random seeds are fixed (`config.SEED = 1337`) and `requirements.txt` pins the
package versions used (Python 3.12.10). Results were produced on CPU; GPU
execution may alter the last decimal place.
