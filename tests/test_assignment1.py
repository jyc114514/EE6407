from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assignment1_solution as solution


def write_workbook(path: Path, rows: list[tuple[object, ...]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    for row in rows:
        worksheet.append(row)
    workbook.save(path)


def synthetic_rows() -> tuple[list[tuple[object, ...]], list[tuple[float, ...]]]:
    training: list[tuple[object, ...]] = []
    for zero_based_row in range(120):
        class_label = zero_based_row // 40 + 1
        offset = zero_based_row % 10
        row: list[object] = [
            class_label * 10 + offset * 0.05,
            class_label * 8 + offset * 0.03,
            class_label * 6 + offset * 0.02,
            class_label * 4 + offset * 0.01,
            class_label,
        ]
        if zero_based_row == 4:
            row[0] = 100.0
        if zero_based_row == 38:
            row[0] = "?"
        if zero_based_row == 60:
            row[1] = "?"
        if zero_based_row == 82:
            row[3] = "?"
        training.append(tuple(row))

    test = [
        (
            class_label * 10 + offset * 0.025,
            class_label * 8 + offset * 0.015,
            class_label * 6 + offset * 0.01,
            class_label * 4 + offset * 0.005,
        )
        for offset in range(10)
        for class_label in (1, 2, 3)
    ]
    return training, test


class Assignment1SolutionTests(unittest.TestCase):
    def test_missing_outlier_cleaning_model_and_outputs(self) -> None:
        training, test = synthetic_rows()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            root = Path(temp_dir)
            assignment_dir = root / "assigenment1"
            assignment_dir.mkdir()
            write_workbook(assignment_dir / "TrainingData (2).xlsx", training)
            write_workbook(assignment_dir / "TestData.xlsx", test)

            result = solution.run_analysis(root)
            self.assertEqual(result["missing_rows"], [39, 61, 83])
            self.assertEqual(result["iqr_outlier_rows"], [5])
            self.assertEqual(result["deleted_rows"], [5, 39, 61, 83])
            self.assertEqual(result["retained_count"], 116)
            self.assertEqual(result["class_counts_after"], {"1": 38, "2": 39, "3": 39})
            self.assertEqual(result["model"]["test_count"], 30)
            self.assertTrue(all(label in (1, 2, 3) for label in result["model"]["predictions"]))
            self.assertTrue(all(value > 0 for row in result["model"]["variances"] for value in row))

            output_dir = root / "results"
            solution.write_outputs(root, output_dir)
            with (output_dir / "assignment1_predictions.csv").open(newline="", encoding="utf-8") as stream:
                prediction_rows = list(csv.reader(stream))
            self.assertEqual(len(prediction_rows), 31)
            self.assertEqual((output_dir / "assignment1_results.json").is_file(), True)


if __name__ == "__main__":
    unittest.main()
