# DLMMDD Synthetic Image Attribution Challenge: 1st Place Solution

This repository contains the notebooks used for the winning entry in the **DLMMDD Workshop Synthetic Source Attribution Challenge**. The best private leaderboard accuracy was **0.997333**. A separate submission reached the best public score of **0.999333**.

## Official Results

The scores below were recorded after the competition closed. When a notebook produced several files, the table lists each file that received an official score.

| Notebook | Method | CSV submitted | Public LB | Private LB |
|---|---|---|---:|---:|
| `DLMMDD_EVA02_Large_+_ConvNeXtV2_Balanced.ipynb` | Three-fold EVA02-Large and ConvNeXt V2-Large ensemble with exact balanced assignment | `submission_diverse_balanced_exact.csv` | **0.999333** | 0.996666 |
| `DLMMDD_prior_logit_ensemble.ipynb` | Five-fold calibrated EVA02-Large ensemble with Sinkhorn balancing | `submission_balanced_sinkhorn.csv` | 0.998000 | **0.997333** |
| `DLMMDD_prior_logit_ensemble.ipynb` | Five-fold calibrated EVA02-Large ensemble with exact balanced assignment | `submission_balanced_exact.csv` | 0.998000 | **0.997333** |
| `[5folds] DLMMDD_ensemble_with_tta_and_calibration.ipynb` | Five-fold calibrated EVA02-Large ensemble before balancing | `submission_calibrated_ensemble.csv` | 0.996000 | **0.997333** |
| `DLMMDD_EVA02_Large.ipynb` | Three-fold EVA02-Large processed-validation model | `submission_pad_processedval_eva02_large448_96gb_orig.csv` | 0.996666 | 0.994666 |
| `DLMMDD_EVA02_Large448_96GB_LBPush_5Fold.ipynb` | Five-fold EVA02-Large, original test view | `submission_pad_processedval_eva02_large448_96gb_5fold_orig.csv` | 0.995333 | 0.995333 |
| `DLMMDD_ConvNext_v2_large.ipynb` | Three-fold ConvNeXt V2-Large ensemble | `submission_convnextv2_large.csv` | 0.989333 | 0.993333 |

## Dataset Access

Join the Kaggle competition before running the notebooks. The dataset is private to accepted competition participants, so each user must authenticate with a personal Kaggle credential.

1. Join the competition on Kaggle.
2. Create or download `kaggle.json` from the Kaggle account settings.
3. Upload the credential in Colab, or place it in the standard Kaggle configuration directory for a local run.
4. Run the authentication and competition-download cells.

The Colab notebooks expect data under:

```text
/root/.cache/kagglehub/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge
```

Edit the configuration paths when using another location. Do not commit a Kaggle credential to this repository.

## Checkpoints

Training can be skipped when compatible checkpoints are available:

- [Three-fold EVA02-Large checkpoint archive](https://drive.google.com/file/d/1Dm04AUvwuxiP7gE45paJEzIL0W4T9_aJ/view?usp=sharing)
- [Three-fold ConvNeXt V2-Large checkpoint archive](https://drive.google.com/file/d/1MZkfmhfamrz9ELg8NUUrsmI4c6xoKnVQ/view?usp=sharing)

The five-fold private-best path also needs EVA02-Large folds 3 and 4. Generate them with `DLMMDD_EVA02_Large448_96GB_LBPush_5Fold.ipynb`, or place a complete five-fold archive in the location configured by the calibrated-inference notebook.

For Colab, upload each archive to `/content/` or `/content/drive/MyDrive/` and edit the path variables. For a local run, extract the archives into the project directory and point the notebook configuration to those directories.

## Private-Best Path

The private-best submission uses EVA02-Large only. ConvNeXt and pseudo-label fine-tuning are not part of this path.

1. Run `DLMMDD_EVA02_Large.ipynb` to train folds 0, 1, and 2. It writes checkpoints to `dlmmdd_pad_processedval_eva02_large448_96gb_outputs`.
2. Put those checkpoints in the output directory used by `DLMMDD_EVA02_Large448_96GB_LBPush_5Fold.ipynb`. With `force_retrain=False`, the notebook reuses folds 0-2 and trains folds 3-4.
3. Archive the resulting five checkpoints as `dlmmdd_pad_processedval_eva02_large448_96gb_outputs5folds.zip`, or edit the archive path in `[5folds] DLMMDD_ensemble_with_tta_and_calibration.ipynb`.
4. Run the calibrated-inference notebook. It fits one temperature per fold on the matching processed-validation split, predicts four test views, and averages the five calibrated log-probability matrices with equal weight.
5. Keep the generated `ensemble_logprobs.npy` and `ensemble_ids.csv` in the same row order. Run `DLMMDD_prior_logit_ensemble.ipynb` to create the raw, Sinkhorn-balanced, and exact-balanced submissions.

The four inference views combine scales `1.0` and `0.92`, each with and without a horizontal flip. The fitted fold temperatures were `0.558371`, `0.598287`, `0.585643`, `0.594193`, and `0.586100`.

The test set contains 3,000 images from ten sources. The competition composition and the observed training counts establish a target of 300 test images per class. Sinkhorn scaling adjusts the class marginals softly and changed three raw predictions. Exact assignment changed seven predictions while enforcing exactly 300 predictions per class. Both submissions retained the private score of 0.997333.

## Best-Public Path

The best-public submission follows a separate branch:

1. Run `DLMMDD_EVA02_Large.ipynb` and `DLMMDD_ConvNext_v2_large.ipynb`, or load their three-fold checkpoint archives.
2. Run `DLMMDD_EVA02_Large_+_ConvNeXtV2_Balanced.ipynb`.
3. Submit `submission_diverse_balanced_exact.csv`.

This notebook calibrates each fold, evaluates four test views, and performs an out-of-fold weight sweep. The recorded sweep selected a ConvNeXt V2-Large weight of `0.40` and an EVA02-Large weight of `0.60`. Exact balanced assignment then changed five predictions. This branch achieved the highest public score but not the highest private score.

## Artifact Integrity

The two generated test-prediction arrays are not committed. Recreate them with the calibrated-inference notebook and compare their SHA-256 hashes before applying marginal balancing:

| Artifact | SHA-256 |
|---|---|
| `[5folds] DLMMDD_ensemble_with_tta_and_calibration.ipynb` | `e4eb095a27e7104b487cc11f99816f59833967a48fceeb532187350fac86003f` |
| `DLMMDD_EVA02_Large448_96GB_LBPush_5Fold.ipynb` | `441ca3c3cad22d1f19f754021711e28c7370f7c62458fba6a3138cda84e8d199` |
| `ensemble_logprobs.npy` | `48defd0eb038ca8424f1460b5f63a3374e0c85dafdd6bcf1836a3e52d9af386c` |
| `ensemble_ids.csv` | `af9d7adce4699b68f7541040e1c3f2a5b1b6c5590d260afd7c598de6f5dd95ed` |

## Hardware and Runtime

All recorded runs used Google Colab. Runtime comparisons are approximate because the notebooks ran on different GPU models and included different amounts of inference and file handling.

| Stage | Hardware | Recorded or estimated time |
|---|---|---:|
| EVA02-Large folds 0-2 | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB | 169.5 min from epoch logs |
| EVA02-Large folds 3-4 | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB | about 113.7 min from epoch timings |
| Five-fold EVA02-Large training total | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB | about 283.2 min |
| Five-fold calibration and four-view inference | NVIDIA L4 | 59.6 min for the inference cell |
| ConvNeXt V2-Large folds 0-2 | NVIDIA A100-SXM4-80GB | 93.9 min from epoch logs |
| Diverse ensemble and marginal balancing | NVIDIA L4 Colab session | consolidated runtime not captured |

The 59.6-minute calibrated-inference figure excludes the separate dataset download and checkpoint extraction cells. Marginal balancing runs on the saved `3000 x 10` score matrix and takes little time compared with model inference.
