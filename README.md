# BioShield

BioShield is a machine-learning framework for antimicrobial peptide (AMP) classification and computational prioritization of peptide candidates relevant to biofilm control in *Pseudomonas aeruginosa*.

## Overview

BioShield combines:

- 28 physicochemical peptide features
- ESM-2 protein-language-model embeddings
- XGBoost classification
- Group-aware cross-validation for model evaluation
- Platt probability calibration
- Similarity-family-aware train/validation/test splitting
- Independent external evaluation

The final released fusion model uses **28 physicochemical features + 480-dimensional ESM-2 35M embeddings**, for a total of **508 features**.

## Final Dataset

Total sequences: **9,512**

| Split | Samples |
|---|---:|
| Train | 6,658 |
| Validation | 1,427 |
| Test | 1,427 |

Similarity families were kept entirely within individual splits to reduce sequence-similarity leakage.

## Final Model

The final BioShield classifier is an XGBoost binary classification model.

### Features

- 28 physicochemical features
- 480-dimensional ESM-2 35M embedding
- 508 features in total

### Parameters

- `n_estimators=400`
- `max_depth=5`
- `learning_rate=0.05`
- `subsample=0.80`
- `colsample_bytree=0.80`
- `tree_method=hist`
- `random_state=42`

Probability calibration was performed using Platt scaling on 5-fold StratifiedGroupKFold out-of-fold training predictions.

Locked classification threshold:

**0.495**

## Performance

### Group-Aware Nested Cross-Validation

| Metric | Mean ± SD |
|---|---:|
| ROC-AUC | 0.9428 ± 0.0056 |
| PR-AUC | 0.9452 ± 0.0077 |

### Independent Test Set

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9878 |
| PR-AUC | 0.9907 |
| Accuracy | 0.9488 |
| Precision | 0.9474 |
| Sensitivity | 0.9627 |
| Specificity | 0.9310 |
| F1 | 0.9550 |
| MCC | 0.8959 |
| Brier score | 0.0396 |

Independent test confusion matrix:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 580 | 43 |
| Actual Positive | 30 | 774 |

## Repository Structure

### Final model and results

- `models/BioShield_fusion_esm35.pkl` — frozen BioShield fusion model (9,512-sequence dataset)
- `evaluation/BioShield_final_test_metrics.csv` — independent test metrics
- `evaluation/BioShield_final_test_predictions.csv` — independent test predictions
- `data/BioShield_similarity_families.csv` — final similarity-family assignments
- `data/bioshield_feature_index.csv` — feature/index mapping

### Feature representation

The released fusion model uses:
- 28 physicochemical features
- 480-dimensional ESM-2 35M embeddings
- 508 total input features

The repository includes the generated training-time feature matrices used for model development and reproducibility.

### Independent evaluation

- `external_validation/baamps_pa_biofilm_formation_INDEPENDENT_FINAL.csv`
- `baamps_pa_biofilm_formation_INDEPENDENT_FINAL.fasta`
- `external_validation/baamps_pa_biofilm_formation_INDEPENDENT_PREDICTIONS.csv`

### Reproducibility

`feats_BioShield.py` contains the final feature-generation workflow.

Development and validation scripts are retained under:

`scripts/development/`

These include the nested-CV, dataset-audit, family-construction, and final-test evaluation scripts used during development.

The repository intentionally excludes temporary databases, intermediate similarity-search files, obsolete dataset versions, cached files, and experimental artifacts.

## Final Test Protocol

The independent test set was frozen before final evaluation and was evaluated without post-hoc threshold optimization.

The reported independent-test results therefore represent the locked final evaluation.

## Independent External Validation

BioShield was additionally evaluated on an independent set of 41 BaAMPs peptides associated with *Pseudomonas aeruginosa* biofilm formation.

The independent-validation set contained 31 positive and 10 negative sequences. The validation sequences had no exact overlap with the BioShield training set and passed the repository's independence filter requiring no ≥40% identity and ≥80% coverage hit to the BioShield training set.

| Metric | Result |
|---|---:|
| Number of sequences | 41 |
| Positive | 31 |
| Negative | 10 |
| Accuracy | 0.7805 |
| Precision | 0.7750 |
| Sensitivity | 1.0000 |
| Specificity | 0.1000 |
| F1 | 0.8732 |
| MCC | 0.2784 |
| Balanced Accuracy | 0.5500 |
| AUROC | 0.8419 |
| AUPRC | 0.9337 |

Confusion matrix:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 1 | 9 |
| Actual Positive | 0 | 31 |

The predictions and associated metadata are provided in:

`external_validation/baamps_pa_biofilm_formation_INDEPENDENT_PREDICTIONS.csv`

This external validation should be interpreted as an additional independent evaluation rather than as a universal benchmark against other AMP classifiers.

## Intended Use

BioShield is intended for research and computational prioritization of antimicrobial peptide candidates.

Predictions are computational and should be experimentally validated before biological or therapeutic conclusions are made.

## Citation

If you use BioShield in academic work, please cite the associated project or thesis publication when available.

## Usage

```bash
pip install -r requirements.txt

python predict.py \
  --fasta examples/example.fasta \
  --model models/BioShield_fusion_esm35.pkl \
  --output predictions.csv
```

Output columns: `id, sequence, length, raw_probability, calibrated_probability, prediction, prediction_label` (locked threshold 0.495).

Verify model integrity:

```bash
sha256sum -c models/SHA256SUMS.txt
```

The model is a joblib pickle. Load it only from this repository's official release.

## Limitations

- On the independent BaAMPs set (41 peptides), specificity was 0.10: 9 of 10 negatives were predicted positive. The classifier is over-permissive on this external negative set, so external performance should not be assumed to match the internal test results.
- Independent-test scores (ROC-AUC 0.9878) are higher than the nested group-aware CV mean (ROC-AUC 0.9428 ± 0.0056). The nested-CV estimate should be treated as the more conservative estimate of generalization.
- The external set is small (31 positive, 10 negative), so its metrics have wide uncertainty.
- Predictions are computational and require experimental validation.

## Feature files

`BioShield_features/` contains feature matrices for the released model (`phys_BioShield.npy`, `esm35_BioShield.npy`). The `esm8` and `esm150` matrices were used only for ablation comparisons and are not part of the released model.
