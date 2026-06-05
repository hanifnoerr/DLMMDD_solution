# 1st Place Solution - DLMMDD Synthetic Image Attribution Challenge

This repository contains our notebooks for the **DLMMDD Workshop Synthetic Source Attribution Challenge**.
Final leaderboard score: **0.997333 private accuracy**.
Best public submission: **0.999333 public accuracy**.

## Notebook Results

Scores below are the official Kaggle public/private leaderboard scores captured after the competition. If a notebook writes more than one CSV, the table lists the CSV that was officially scored.

| Notebook | Method | CSV to submit | Public LB | Private LB | Notes |
|---|---|---:|---:|---:|---|
| `DLMMDD_EVA02_Large_+_ConvNeXtV2_Balanced.ipynb` | EVA02-Large + ConvNeXtV2 diverse ensemble with exact balanced assignment | `submission_diverse_balanced_exact.csv` | 0.999333 | 0.996666 | Best public score |
| `DLMMDD_prior_logit_ensemble.ipynb` | Prior-logit ensemble with Sinkhorn balancing | `submission_balanced_sinkhorn.csv` | 0.998000 | 0.997333 | Best private-score submission |
| `DLMMDD_prior_logit_ensemble.ipynb` | Prior-logit ensemble with exact balancing | `submission_balanced_exact.csv` | 0.998000 | 0.997333 | Same private score as Sinkhorn |
| `DLMMDD_EVA02_Large.ipynb` | EVA02-Large processed-validation base model | `submission_pad_processedval_eva02_large448_96gb_orig.csv` | 0.996666 | 0.994666 | Main base checkpoint generator |
| `DLMMDD_ConvNext_v2_large.ipynb` | ConvNeXt V2-Large fold ensemble | `submission_convnextv2_large.csv` | 0.989333 | 0.993333 | CNN baseline checkpoint generator |

## Dataset and Credential Setup

To run the notebooks from scratch, you must join the Kaggle competition and use your own Kaggle credentials.

1. Join the competition page on Kaggle.
2. Create or download your Kaggle API token from your Kaggle account settings.
3. Upload `kaggle.json` in Colab, or place it in the standard Kaggle config directory on your local machine.
4. Run the notebook cells that authenticate with Kaggle and download the competition data.

The notebooks expect the competition dataset to be available in the paths used by Kaggle/Colab, for example:

```text
/root/.cache/kagglehub/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge
```

If you use a different directory, edit the dataset paths in the notebook configuration cells.

## Checkpoint Setup

If you do not want to train the base models yourself, **download** these checkpoint archives and place them where the downstream notebooks expect them.

[dlmmdd_pad_processedval_eva02_large448_96gb_outputs.zip](https://drive.google.com/file/d/1Dm04AUvwuxiP7gE45paJEzIL0W4T9_aJ/view?usp=sharing)

[dlmmdd_robust_outputs.zip](https://drive.google.com/file/d/1MZkfmhfamrz9ELg8NUUrsmI4c6xoKnVQ/view?usp=sharing)


For Google Colab:

1. Download the checkpoint `.zip` files from the links below.
2. Upload them to `/content/`, `/content/drive/MyDrive/`, or another Colab-accessible directory.
3. Edit the notebook path variables so they point to your uploaded files.

For local execution:

1. Download and extract the checkpoint archives into your local project directory.
2. Edit the checkpoint paths in the notebook configuration cells.

## Run Order

1. Run `paper/public999333private996666/DLMMDD_EVA02_Large.ipynb`.
   This trains the EVA02-Large base model and produces the base submission plus the `dlmmdd_pad_processedval_eva02_large448_96gb_outputs` checkpoint directory.

2. Run the ConvNeXtV2 checkpoint notebook.
   The diverse ensemble expects a ConvNeXtV2 checkpoint archive named `dlmmdd_robust_outputs.zip`. In this repository, the archived ConvNeXtV2-Large notebook is `DLMMDD_ConvNext_v2_large.ipynb`.

3. Run `DLMMDD_EVA02_Large_+_ConvNeXtV2_Balanced.ipynb`.
   Required inputs:
   - `dlmmdd_pad_processedval_eva02_large448_96gb_outputs.zip` or `DLMMDD_EVA02_Large.ipynb` notebook output
   - `dlmmdd_robust_outputs.zip` or `DLMMDD_ConvNext_v2_large.ipynb` notebook output

   Main output:
   - `submission_diverse_balanced_exact.csv`

4. Run `DLMMDD_prior_logit_ensemble.ipynb`.
   Required inputs:
   - `ensemble_logprobs.npy` and `ensemble_ids.csv` from `dlmmdd_pad_processedval_eva02_large448_96gb_outputs.zip` or `DLMMDD_EVA02_Large_+_ConvNeXtV2_Balanced.ipynb` notebook output

   Main output:
   - `submission_balanced_sinkhorn.csv`
   - `submission_balanced_exact.csv`


## Method Summary
The final solution combines:
- processed-validation checkpointing for the hidden train/test post-processing shift;
- EVA02-Large as the strongest base model;
- ConvNeXtV2 as a complementary CNN family;
- out-of-fold temperature calibration;
- weighted log-softmax ensembling;
- pseudo-label fine-tuning as an additional experiment;
- class-marginal balancing using the known 300-test-images-per-class structure.

## Hardware Notes
The main notebooks were run on Google Colab with different GPUs:
- EVA02-Large base notebook: NVIDIA RTX PRO 6000 Blackwell Server Edition, about 95GB VRAM.
- ConvNeXt V2-Large notebook: NVIDIA A100-SXM4-80GB.
- Final ensemble notebooks: L4 GPU.

