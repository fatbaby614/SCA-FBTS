# SCA-FBTS

**Spatial Covariance-based Filter-Bank Tangent Space classification for automatic sleep staging from low-channel EEG.**

SCA-FBTS is a deliberately lightweight and fully specified framework for 5-class sleep staging
(Wake, N1, N2, N3, REM). It is assembled from standard, well-understood components — a fixed
filter bank, OAS-shrunk spatial covariance matrices, affine-invariant tangent-space embeddings,
ANOVA feature selection, an RBF-kernel SVM and an N1-aware temporal smoothing stage — and is
designed for the low-channel, portable-EEG regime rather than for state-of-the-art accuracy on
large montages.

This repository contains the complete experimental pipeline, the evaluation code for three public
databases, and the scripts that produce every table and figure of the accompanying manuscript.

---

## Method overview

Each 30-s epoch of multichannel EEG is processed in five stages:

1. **Fixed filter bank** — the signal is decomposed into `K = 8` contiguous, non-adaptive
   sub-bands between 0.5 and 30 Hz (delta 0.5–4, low theta 4–6, high theta 6–8, low alpha 8–10,
   high alpha 10–12, sigma 12–14, low beta 14–20, high beta 20–30 Hz). Each band is extracted
   with a 4th-order Butterworth band-pass filter applied forward and backward (zero-phase).
2. **Per-band covariance** — the epoch-wise temporal mean is removed and a spatial covariance
   matrix is estimated per band and per epoch, regularised by Oracle Approximating Shrinkage
   (OAS) to guarantee positive definiteness.
3. **Tangent-space mapping** — each covariance matrix is mapped to the tangent space of the SPD
   manifold at the band-specific Fréchet mean of the **training folds** under the affine-invariant
   Riemannian metric. The vectorised band-wise representations are concatenated into a single
   feature vector (24 features at `C = 2` channels, 168 at `C = 6`).
4. **Feature selection and classification** — ANOVA F-score selection retains
   `min(100, dim)` features; an RBF-kernel SVM with class-balanced weights outputs
   Platt-calibrated posterior probabilities.
5. **N1-aware smoothing** — a non-causal centred moving average (`w = 3` epochs) smooths the
   hypnogram, except that epochs tentatively scored as the minority stage N1 keep their
   unsmoothed posterior.

All reference points, standardisation statistics and model parameters are fitted on training-fold
data only; hyperparameters are fixed in advance and never tuned on a test fold. See
[code/sca_fbts.py](code/sca_fbts.py) for the implementation.

---

## Repository structure

```
code/                            experiments, analyses and figure/table generators
  sca_fbts.py                      proposed method (filter bank + tangent space + SVM + smoothing)
  algorithms_collection.py         all compared algorithms, including the deep baselines
  data_loader_sleep.py             Sleep-EDF / ISRUC / DREAMS loading and epoching
  evaluate_sleep.py                single-dataset evaluation with subject-wise cross-validation
  run_all_experiments.py           orchestrator for the full 19-step pipeline
  run_full_experiment_tmux.sh      Linux launcher (tmux-friendly, writes date-stamped logs)
  run_fbts_ablation.py             component ablation of SCA-FBTS
  run_noise_robustness_experiment.py  additive-noise robustness study (controlled SNR)
  run_sleep_channel_ablation.py    channel ablation on Sleep-EDF
  run_channel_importance_analysis.py, run_band_contribution_analysis.py
  run_parameter_sensitivity.py     number of bands / tangent dimensionality / window length
  run_causal_smoothing_ablation.py causal vs. non-causal smoothing
  run_cross_dataset.py             independent-database evaluation
  generate_*.py                    tables, hypnograms, t-SNE, efficiency and training-time figures
  visualization.py                 shared plotting utilities
  config/                          dataset, algorithm and experiment configuration
results/                         all outputs (never hand-edited)
  sleep_edf/  isruc/  dreams/      per-database evaluation summaries (CSV)
  common/                          shared tables and figures (CSV, .tex, PNG)
  logs/                            per-run logs
data/                            placeholder only — datasets are not redistributed here
requirements.txt                 pinned runtime dependencies
```

---

## Getting started

Python 3.10–3.13 is supported.

```bash
git clone https://github.com/fatbaby614/SCA-FBTS.git
cd SCA-FBTS
pip install -r requirements.txt
```

The deep baselines require PyTorch. On a CUDA machine, install a matching build first:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Quick sanity check before running anything heavy:

```bash
python code/check_imports.py
python code/run_all_experiments.py --quick      # small-sample smoke test of every step
```

---

## Datasets

The three databases are **not** redistributed in this repository. Download them from their
original sources and pass the root directories on the command line.

| Database | Used here | Source |
|---|---|---|
| Sleep-EDF Expanded v1.0.0 (Sleep Cassette) | 135 recordings / 72 subjects, Fpz-Cz + Pz-Oz @ 100 Hz | PhysioNet, DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676) |
| ISRUC-Sleep | 40 subjects, one night each, 6 EEG channels @ 200 Hz | DOI [10.1016/j.cmpb.2015.10.013](https://doi.org/10.1016/j.cmpb.2015.10.013) |
| DREAMS Patients | 27 recordings, 2 channels, 5-s labels aggregated to 30-s epochs | Zenodo, DOI [10.5281/zenodo.2650142](https://doi.org/10.5281/zenodo.2650142) |

Path arguments used by the pipeline:

```
--sleep-edf-path  /path/to/sleep-edf-database-expanded-1.0.0
--isruc-path      /path/to/ISRUC-SLEEP
--dreams-path     /path/to/DREAMS/DatabasePatients
```

---

## Usage

### Full pipeline

The canonical run on a Linux workstation (19 steps: evaluation on all three databases, all
ablations, statistical tests and every figure/table generator):

```bash
cd code
chmod +x run_full_experiment_tmux.sh
./run_full_experiment_tmux.sh
```

It runs in the foreground; launch it inside tmux (`tmux new -s sleep_run`) to survive an SSH
disconnect. The equivalent explicit command is:

```bash
python code/run_all_experiments.py \
    --output-dir results \
    --sleep-edf-path /path/to/sleep-edf-database-expanded-1.0.0 \
    --isruc-path     /path/to/ISRUC-SLEEP \
    --dreams-path    /path/to/DREAMS/DatabasePatients \
    --continue-on-error
```

`--continue-on-error` keeps the remaining steps running if one step fails; `--quick` runs every
step on a small sample. Individual steps can be re-run in isolation:

```bash
python code/run_all_experiments.py --step 5      # step 5 of 19, e.g. the noise study
```

### Single-dataset evaluation

```bash
python code/evaluate_sleep.py \
    --dataset sleep_edf \
    --data-path /path/to/sleep-edf-database-expanded-1.0.0 \
    --algorithms SCA-FBTS TinySleepNet DeepSleepNet SSC-SleepNet \
                 HandcraftedFeatures+RF RiemannTangentSpace MDM \
    --cv-mode subject \
    --output-dir results/sleep_edf
```

Useful options: `--dataset {dummy,sleep_edf,isruc,dreams,combined}`, `--subjects 0~9`,
`--recording 1 2`, `--select-channels`, `--select-n-channels N`, `--tsne`, `--epochs 300`,
`--check-gpu`.

### Regenerating a single figure or table

Every artifact in `results/` is produced by a standalone script, for example:

```bash
python code/generate_efficiency_vs_accuracy.py \
    --csv results/sleep_edf/sleep_evaluation_summary_sleepedf_20260907_022941.csv \
    --output results/common
```

---

## Evaluation protocol

- **Subject-wise 5-fold cross-validation** (`GroupKFold`); all epochs of a recording always stay
  in the same fold, so no subject appears in both training and test data.
- **The subject is the independent statistical unit** for every reported test.
- Per-subject **Wilcoxon signed-rank** tests with Bonferroni correction
  (α = 0.05/6 ≈ 0.0083 for the six paired comparisons) and a **Friedman** test across the
  seven algorithms.
- Reference points, feature-selection statistics, class weights and calibration are all fitted
  inside the training folds; a fixed seed of 42 is used for every stochastic component.

## Compared algorithms

| Algorithm | Family | Notes |
|---|---|---|
| **SCA-FBTS** | proposed | filter bank + per-band tangent space + class-balanced SVM + N1-aware smoothing |
| Riemannian TS | classical | single broadband OAS covariance + tangent space + SVM (no filter bank, no smoothing) |
| MDM | classical | OAS covariances classified by minimum distance to Fréchet class means |
| Handcrafted + RF | classical | time-domain statistics and relative band powers + random forest (300 trees) |
| TinySleepNet | deep | re-implementation of the CNN stage, without the LSTM layer |
| DeepSleepNet | deep | re-implementation of the dual-branch CNN encoder, without the bi-LSTM layer |
| SSC-SleepNet | deep | compact CNN–Transformer baseline implemented here; no published counterpart |

**Implementation notes.** All three deep baselines are re-implementations in which the recurrent
sequence layer is omitted, so every epoch is classified independently; none of them uses
sequence-level smoothing. They therefore measure representation quality under the same
epoch-wise decision rule, not the accuracy of the original published architectures.

---

## Reported results

The authoritative numbers are the CSV tables under `results/`; the manuscript's tables and
figures are generated from them. For reference, the Sleep-EDF run in
`results/sleep_edf/sleep_evaluation_summary_sleepedf_20260907_022941.csv` reports SCA-FBTS at
76.63 % overall accuracy with κ = 0.679 and macro-F1 = 0.697, a 13.6 MB serialised model and
31.0 ms per 30-s epoch inference. Nothing in this repository is hand-entered: every value is
written by the scripts in `code/`.

---

## Citation

The manuscript describing this work is currently under review. Until it is published, please
cite the repository:

```bibtex
@misc{scafbts2026,
  title        = {{SCA-FBTS}: Spatial Covariance-based Filter-Bank Tangent Space
                  Classification for Sleep Staging from Low-Channel {EEG}},
  author       = {Tan, Huang and Wang, Xinrong and Zhang, Li and Li, Xiangzhu and Yin, Guangqiang},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Manuscript under review},
  url          = {https://github.com/fatbaby614/SCA-FBTS}
}
```

This entry will be updated with the journal reference once the paper is accepted.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This work uses the Sleep-EDF Expanded database (PhysioNet), the ISRUC-Sleep database and the
DREAMS database. We thank the authors and maintainers of these resources for making their data
publicly available. The implementation builds on the open-source scientific Python ecosystem,
notably NumPy, SciPy, scikit-learn, MNE and PyRiemann.

## Contact

For questions about the code or the experiments, please open an issue at
<https://github.com/fatbaby614/SCA-FBTS/issues>.