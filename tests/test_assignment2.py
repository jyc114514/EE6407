from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assignment2_solution as solution


def data_element(element_type: int, payload: bytes, endian: str) -> bytes:
    padding = b"\x00" * ((-len(payload)) % 8)
    return struct.pack(endian + "II", element_type, len(payload)) + payload + padding


def matrix_element(name: str, values: np.ndarray, endian: str) -> bytes:
    array = np.asarray(values, dtype=endian + "f8")
    flags = struct.pack(endian + "II", 6, 0)  # Real numeric matrix, MATLAB double class.
    dimensions = np.asarray(array.shape, dtype=endian + "i4").tobytes()
    name_bytes = name.encode("ascii")
    real_values = array.tobytes(order="F")
    matrix_payload = b"".join(
        (
            data_element(6, flags, endian),
            data_element(5, dimensions, endian),
            data_element(1, name_bytes, endian),
            data_element(9, real_values, endian),
        )
    )
    return data_element(14, matrix_payload, endian)


def write_mat_file(path: Path, variables: dict[str, np.ndarray], endian: str = "<") -> None:
    description = b"MATLAB 5.0 MAT-file, synthetic unit-test fixture".ljust(116, b" ")
    subsystem_offset = b"\x00" * 8
    version = b"\x00\x01" if endian == "<" else b"\x01\x00"
    marker = b"IM" if endian == "<" else b"MI"
    header = description + subsystem_offset + version + marker
    path.write_bytes(header + b"".join(matrix_element(name, value, endian) for name, value in variables.items()))


class Assignment2SolutionTests(unittest.TestCase):
    def test_mat_v5_loader_supports_both_endian_orders(self) -> None:
        matrix = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            for endian in ("<", ">"):
                path = Path(temp_dir) / ("little.mat" if endian == "<" else "big.mat")
                write_mat_file(path, {"matrix": matrix}, endian)
                loaded = solution.load_mat_v5(path)
                np.testing.assert_array_equal(loaded["matrix"], matrix)

    def test_truncated_mat_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "truncated.mat"
            path.write_bytes(b"MATLAB")
            with self.assertRaisesRegex(ValueError, "shorter than its header"):
                solution.load_mat_v5(path)

    def test_fisher_solution_and_prediction_order_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(6407)
        negative = rng.normal(loc=-2.0, scale=0.35, size=(276, 5))
        positive = rng.normal(loc=2.0, scale=0.35, size=(301, 5))
        x_train = np.vstack((negative, positive))
        y_train = np.concatenate((-np.ones(276), np.ones(301))).reshape(-1, 1)
        x_test = np.vstack((
            rng.normal(loc=-2.0, scale=0.1, size=(9, 5)),
            rng.normal(loc=2.0, scale=0.1, size=(14, 5)),
        ))

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            root = Path(temp_dir)
            assignment_dir = root / "assigenment2"
            assignment_dir.mkdir()
            write_mat_file(assignment_dir / "data_train.mat", {"data_train": x_train})
            write_mat_file(assignment_dir / "label_train.mat", {"label_train": y_train})
            write_mat_file(assignment_dir / "data_test.mat", {"data_test": x_test})

            result = solution.run_analysis(root)
            self.assertEqual(result["mat_variables"]["data_train"]["shape"], [577, 5])
            self.assertEqual(result["mat_variables"]["label_train"]["shape"], [577, 1])
            self.assertEqual(result["mat_variables"]["data_test"]["shape"], [23, 5])
            self.assertEqual(result["class_counts"], {"-1": 276, "+1": 301})
            self.assertEqual(result["predictions"], [-1] * 9 + [1] * 14)
            self.assertLess(result["linear_system_relative_residual"], 1e-10)
            self.assertTrue(result["positive_scale_7_3_preserves_predictions"])
            self.assertTrue(result["negative_scale_with_reversed_rule_preserves_predictions"])


if __name__ == "__main__":
    unittest.main()
