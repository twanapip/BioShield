#!/usr/bin/env python3

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

from common import phys, embed


THRESHOLD = 0.495
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = []
    name = None
    seq = []

    with open(path) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq).upper()))

                name = line[1:].strip()
                if not name:
                    name = f"sequence_{len(records)+1}"

                seq = []
            else:
                if name is None:
                    raise ValueError(
                        "Invalid FASTA: sequence found before FASTA header."
                    )
                seq.append(line.replace(" ", ""))

    if name is not None:
        records.append((name, "".join(seq).upper()))

    if not records:
        raise ValueError("No sequences found in FASTA file.")

    return records


def validate_sequences(records):
    for name, seq in records:
        if not seq:
            raise ValueError(f"{name}: empty sequence.")

        invalid = sorted(set(seq) - VALID_AA)

        if invalid:
            raise ValueError(
                f"{name}: unsupported amino-acid character(s): "
                + ", ".join(invalid)
            )


def apply_platt(calibrator, probabilities):
    eps = 1e-6

    p = np.clip(
        np.asarray(probabilities, dtype=float),
        eps,
        1 - eps
    )

    logits = np.log(
        p / (1 - p)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BioShield antimicrobial-peptide activity predictor "
            "(physicochemical + ESM-2 35M + XGBoost)."
        )
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input peptide FASTA file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file."
    )

    parser.add_argument(
        "--model",
        default="models/BioShield_fusion_esm35.pkl",
        help="Path to frozen BioShield model."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="ESM-2 batch size (default: 64)."
    )

    args = parser.parse_args()

    if not os.path.exists(args.fasta):
        sys.exit(f"ERROR: FASTA file not found: {args.fasta}")

    if not os.path.exists(args.model):
        sys.exit(f"ERROR: model file not found: {args.model}")

    try:
        records = read_fasta(args.fasta)
        validate_sequences(records)
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")

    names = [x[0] for x in records]
    seqs = [x[1] for x in records]

    print("=" * 60)
    print("BioShield AMP Predictor")
    print("=" * 60)
    print(f"Sequences: {len(seqs)}")

    print("\nGenerating 28 physicochemical features...")
    X_phys = phys(seqs)

    if X_phys.shape[1] != 28:
        sys.exit(
            f"ERROR: expected 28 physicochemical features, "
            f"got {X_phys.shape[1]}"
        )

    print("Generating ESM-2 35M embeddings...")
    X_esm = embed(
        seqs,
        "esm35",
        bs=args.batch_size
    )

    if X_esm.shape[1] != 480:
        sys.exit(
            f"ERROR: expected 480 ESM features, "
            f"got {X_esm.shape[1]}"
        )

    X = np.hstack([
        X_phys,
        X_esm
    ]).astype(np.float32)

    if X.shape[1] != 508:
        sys.exit(
            f"ERROR: expected 508 total features, "
            f"got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        sys.exit("ERROR: feature matrix contains NaN or Inf.")

    print(f"Feature matrix: {X.shape}")

    bundle = joblib.load(args.model)

    if not isinstance(bundle, dict):
        sys.exit("ERROR: model file is not a BioShield model bundle.")

    if "model" not in bundle or "platt" not in bundle:
        sys.exit(
            "ERROR: model bundle must contain 'model' and 'platt'."
        )

    model = bundle["model"]
    calibrator = bundle["platt"]

    if getattr(model, "n_features_in_", 508) != 508:
        sys.exit(
            "ERROR: frozen model does not expect 508 features."
        )

    print("Running frozen XGBoost classifier...")

    raw_probability = model.predict_proba(X)[:, 1]

    print("Applying Platt calibration...")

    calibrated_probability = apply_platt(
        calibrator,
        raw_probability
    )

    prediction = (
        calibrated_probability >= THRESHOLD
    ).astype(int)

    result = pd.DataFrame({
        "id": names,
        "sequence": seqs,
        "length": [len(s) for s in seqs],
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "prediction": prediction,
        "prediction_label": [
            "AMP" if x == 1 else "Non-AMP"
            for x in prediction
        ],
    })

    result.to_csv(
        args.output,
        index=False
    )

    print("\nPrediction complete.")
    print(f"Threshold: {THRESHOLD}")
    print(f"AMP predictions: {int(prediction.sum())}")
    print(f"Non-AMP predictions: {int((prediction == 0).sum())}")
    print(f"Results: {args.output}")

    print("\nPreview:")
    print(
        result[
            [
                "id",
                "calibrated_probability",
                "prediction_label"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
