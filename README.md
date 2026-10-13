# Assignment 2 — Fisher Linear Discriminant

This branch contains the Assignment 2 solution and synthetic regression tests. It reads the supplied MATLAB v5 files from a local course-data directory and does not include course datasets or answer PDFs.

## Inputs

Pass the course-data root with `--source-root`. The directory must contain:

```text
assigenment2/
    data_train.mat
    label_train.mat
    data_test.mat
```

The solution includes a small MATLAB v5 numeric-matrix reader and uses only NumPy beyond the Python standard library.

## Run

Python 3.10 or later is required.

```bash
python -m pip install -r requirements.txt
python src/assignment2_solution.py --source-root /path/to/course-data --output-dir results
```

The output directory receives `assignment2_results.json` and `assignment2_predictions.csv`.

## Tests

Tests create synthetic MATLAB v5 numeric matrices in a temporary directory; they do not need the course data.

```bash
python -m unittest discover -s tests -v
```
