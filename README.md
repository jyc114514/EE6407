# Assignment 1 — Gaussian Naive Bayes

This branch contains the Assignment 1 solution and synthetic regression tests. It reads the supplied workbook files from a local course-data directory and does not include course datasets or answer PDFs.

## Inputs

Pass the course-data root with `--source-root`. The directory must contain:

```text
assigenment1/
    TrainingData (2).xlsx
    TestData.xlsx
```

The source folder spelling `assigenment1` and workbook names match the local course files.

## Run

Python 3.10 or later is required.

```bash
python -m pip install -r requirements.txt
python src/assignment1_solution.py --source-root /path/to/course-data --output-dir results
```

The output directory receives `assignment1_results.json` and `assignment1_predictions.csv`.

## Tests

Tests construct synthetic workbooks in a temporary directory; they do not need the course data.

```bash
python -m unittest discover -s tests -v
```
