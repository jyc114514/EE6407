"""MATLAB v5 loading and Fisher linear discriminant solution for Assignment 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


MAT_TYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
    7: "f4", 9: "f8", 12: "i8", 13: "u8",
}


def read_data_element(buffer: bytes, offset: int, endian: str) -> tuple[int, int, bytes, int]:
    """Read a MATLAB v5 data element, including its small-element form."""
    if offset < 0 or offset + 8 > len(buffer):
        raise ValueError(f"Truncated MATLAB data-element tag at offset {offset}.")
    element_type, small_length = struct.unpack_from(endian + "HH", buffer, offset)
    if 0 < element_type <= 18 and 0 < small_length <= 4:
        payload = buffer[offset + 4 : offset + 4 + small_length]
        return element_type, small_length, payload, offset + 8

    element_type, length = struct.unpack_from(endian + "II", buffer, offset)
    payload_start = offset + 8
    payload_end = payload_start + length
    if payload_end > len(buffer):
        raise ValueError(f"MATLAB element {element_type} exceeds its file boundary.")
    next_offset = payload_end if element_type == 15 else payload_end + ((8 - length % 8) % 8)
    if next_offset > len(buffer):
        next_offset = payload_end
    return element_type, length, buffer[payload_start:payload_end], next_offset


def parse_matrix(payload: bytes, endian: str) -> tuple[str, np.ndarray]:
    position = 0
    flags_type, _, flags, position = read_data_element(payload, position, endian)
    if flags_type != 6 or len(flags) < 4:
        raise ValueError("Unsupported MATLAB array-flags element.")
    array_class = struct.unpack_from(endian + "I", flags, 0)[0] & 0xFF

    dims_type, _, dimensions, position = read_data_element(payload, position, endian)
    if dims_type not in (5, 6) or len(dimensions) == 0 or len(dimensions) % 4:
        raise ValueError("Unsupported MATLAB dimensions element.")
    shape = tuple(int(value) for value in np.frombuffer(dimensions, dtype=endian + "i4"))
    if not shape or any(dimension < 0 for dimension in shape):
        raise ValueError(f"Invalid MATLAB array shape: {shape!r}")

    name_type, name_length, name_payload, position = read_data_element(payload, position, endian)
    if name_type not in (1, 2):
        raise ValueError("Unsupported MATLAB variable-name element.")
    name = name_payload[:name_length].decode("utf-8")

    real_type, _, real_payload, _ = read_data_element(payload, position, endian)
    if real_type not in MAT_TYPES:
        raise ValueError(f"Unsupported MATLAB numeric storage type {real_type} for {name}.")
    dtype = np.dtype(endian + MAT_TYPES[real_type])
    array = np.frombuffer(real_payload, dtype=dtype).copy()
    expected_size = math.prod(shape)
    if array.size != expected_size:
        raise ValueError(f"MATLAB variable {name} has {array.size} values; shape {shape} needs {expected_size}.")
    array = array.reshape(shape, order="F")
    if array_class not in (6, 7):
        raise ValueError(f"Expected a real numeric MATLAB matrix for {name}; class code {array_class}.")
    return name, array


def load_mat_v5(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 128:
        raise ValueError(f"MATLAB file is shorter than its header: {path}")
    marker = raw[126:128]
    if marker == b"IM":
        endian = "<"
    elif marker == b"MI":
        endian = ">"
    else:
        raise ValueError(f"Unsupported MATLAB endian marker {marker!r} in {path}")

    variables: dict[str, np.ndarray] = {}
    offset = 128
    while offset + 8 <= len(raw):
        element_type, _, element_payload, next_offset = read_data_element(raw, offset, endian)
        if element_type == 15:
            matrix_type, _, matrix_payload, _ = read_data_element(zlib.decompress(element_payload), 0, endian)
            if matrix_type != 14:
                raise ValueError("Compressed MATLAB element did not contain a matrix.")
            name, array = parse_matrix(matrix_payload, endian)
        elif element_type == 14:
            name, array = parse_matrix(element_payload, endian)
        else:
            offset = next_offset
            continue
        if name in variables:
            raise ValueError(f"Duplicate MATLAB variable name: {name}")
        variables[name] = array
        if next_offset <= offset:
            raise ValueError("MATLAB parser did not advance to the next element.")
        offset = next_offset
    return variables


def run_analysis(source_root: Path) -> dict[str, Any]:
    assignment = source_root / "assigenment2"
    train_vars = load_mat_v5(assignment / "data_train.mat")
    label_vars = load_mat_v5(assignment / "label_train.mat")
    test_vars = load_mat_v5(assignment / "data_test.mat")
    if set(train_vars) != {"data_train"} or set(label_vars) != {"label_train"} or set(test_vars) != {"data_test"}:
        raise ValueError("MATLAB variable names differ from the assignment specification.")

    x_train = np.asarray(train_vars["data_train"], dtype=float)
    y_raw = np.asarray(label_vars["label_train"])
    x_test = np.asarray(test_vars["data_test"], dtype=float)
    if x_train.shape != (577, 5) or y_raw.size != 577 or x_test.shape != (23, 5):
        raise ValueError(f"Unexpected matrix dimensions: Xtrain={x_train.shape}, y={y_raw.shape}, Xtest={x_test.shape}.")
    y_train = y_raw.reshape(-1).astype(int)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Training or test features contain NaN or infinity.")
    if not np.isin(y_train, (-1, 1)).all():
        raise ValueError(f"Unexpected labels: {np.unique(y_train).tolist()}")

    positive = x_train[y_train == 1]
    negative = x_train[y_train == -1]
    mu_positive = positive.mean(axis=0)
    mu_negative = negative.mean(axis=0)
    mean_difference = mu_positive - mu_negative
    centered_positive = positive - mu_positive
    centered_negative = negative - mu_negative
    within_scatter = centered_positive.T @ centered_positive + centered_negative.T @ centered_negative
    rank = int(np.linalg.matrix_rank(within_scatter))
    condition_number = float(np.linalg.cond(within_scatter))
    eigenvalues = np.linalg.eigvalsh(within_scatter)
    if rank != 5 or not math.isfinite(condition_number):
        raise ValueError(f"Within-class scatter is singular or invalid: rank={rank}, cond={condition_number}.")
    if not np.allclose(within_scatter, within_scatter.T, rtol=0.0, atol=1e-12) or eigenvalues.min() <= 0.0:
        raise ValueError("Within-class scatter is not symmetric positive definite.")

    weights = np.linalg.solve(within_scatter, mean_difference)
    weights_lstsq, _, least_squares_rank, _ = np.linalg.lstsq(
        within_scatter, mean_difference, rcond=None
    )
    if least_squares_rank != 5:
        raise ValueError(f"Least-squares cross-check is rank deficient: {least_squares_rank}.")
    bias = float(-0.5 * weights @ (mu_positive + mu_negative))

    train_scores = x_train @ weights + bias
    test_scores = x_test @ weights + bias
    positive_mean_score = float(mu_positive @ weights + bias)
    negative_mean_score = float(mu_negative @ weights + bias)
    train_predictions = np.where(train_scores >= 0.0, 1, -1)
    test_predictions = np.where(test_scores >= 0.0, 1, -1)
    scaled_scores = x_test @ (7.3 * weights) + 7.3 * bias
    scaled_predictions = np.where(scaled_scores >= 0.0, 1, -1)
    reversed_scores = x_test @ (-7.3 * weights) - 7.3 * bias
    reversed_predictions = np.where(reversed_scores <= 0.0, 1, -1)

    return {
        "source_root": str(source_root),
        "mat_variables": {
            "data_train": {"shape": list(x_train.shape), "stored_dtype": str(train_vars["data_train"].dtype), "finite": True},
            "label_train": {"shape": list(y_raw.shape), "stored_dtype": str(y_raw.dtype), "finite": bool(np.isfinite(y_raw).all())},
            "data_test": {"shape": list(x_test.shape), "stored_dtype": str(test_vars["data_test"].dtype), "finite": True},
        },
        "class_counts": {"-1": int(negative.shape[0]), "+1": int(positive.shape[0])},
        "mean_positive": mu_positive.tolist(),
        "mean_negative": mu_negative.tolist(),
        "mean_difference": mean_difference.tolist(),
        "within_class_scatter": within_scatter.tolist(),
        "within_scatter_rank": rank,
        "within_scatter_condition_number_2": condition_number,
        "within_scatter_eigenvalues": eigenvalues.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "positive_mean_decision_score": positive_mean_score,
        "negative_mean_decision_score": negative_mean_score,
        "decision_rule": "predict +1 when w^T x + w0 >= 0; otherwise predict -1",
        "solve_lstsq_relative_difference": float(np.linalg.norm(weights - weights_lstsq) / np.linalg.norm(weights)),
        "linear_system_relative_residual": float(np.linalg.norm(within_scatter @ weights - mean_difference) / np.linalg.norm(mean_difference)),
        "test_scores": test_scores.tolist(),
        "predictions": test_predictions.astype(int).tolist(),
        "test_count": int(x_test.shape[0]),
        "minimum_absolute_test_score": float(np.min(np.abs(test_scores))),
        "minimum_margin_record": int(np.argmin(np.abs(test_scores)) + 1),
        "positive_scale_7_3_preserves_predictions": bool(np.array_equal(test_predictions, scaled_predictions)),
        "negative_scale_with_reversed_rule_preserves_predictions": bool(np.array_equal(test_predictions, reversed_predictions)),
        "training_accuracy_internal_only": float(np.mean(train_predictions == y_train)),
    }


def write_outputs(source_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_analysis(source_root)
    (output_dir / "assignment2_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "assignment2_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("test_record", "decision_score", "predicted_class"))
        for index, (score, label) in enumerate(zip(result["test_scores"], result["predictions"]), start=1):
            writer.writerow((index, f"{score:.12g}", label))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_outputs(args.source_root, args.output_dir)
    print(json.dumps({
        "mat_variables": result["mat_variables"],
        "class_counts": result["class_counts"],
        "mean_positive": result["mean_positive"],
        "mean_negative": result["mean_negative"],
        "within_scatter_rank": result["within_scatter_rank"],
        "within_scatter_condition_number_2": result["within_scatter_condition_number_2"],
        "weights": result["weights"],
        "bias": result["bias"],
        "predictions": result["predictions"],
        "minimum_absolute_test_score": result["minimum_absolute_test_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
