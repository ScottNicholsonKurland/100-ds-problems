# 100 Data Science Programming Problems

![Python checks](https://github.com/ScottNicholsonKurland/100-ds-problems/actions/workflows/tests.yml/badge.svg)

A structured collection of Python, data manipulation, statistics, modeling, SQL, plotting, and web-programming exercises for building practical data science and data analyst skills.

This repository is a deliberate practice project: each completed section should include problem statements, readable reference solutions, and pytest coverage.

## Status

| Section | Problems Drafted | Solutions Added | Tests Added |
|---|---:|---:|---:|
| General Programming | 13 | 13 | 13 |
| Data Manipulation - NumPy | 12 | 12 | 17 |
| Data Manipulation - pandas | 3 | 3 | 7 |
| Probability | 4 | 4 | 9 |
| Statistics | 2 | 2 | 9 |
| Algorithms | 7 | 7 | 14 |
| Plotting | 3 | 3 | 6 |
| SQL | 4 | 4 | 5 |
| Web Programming | 5 | 5 | 10 |
| **Total** | **53** | **53** | **90** |

## Extensions

| Extension | Implementations | Tests |
|---|---:|---:|
| scikit-learn algorithm variants | 4 | 10 |

## Skills Demonstrated

- Python fundamentals
- Data structures
- Algorithmic problem solving
- NumPy and pandas practice
- Statistics and probability exercises
- SQL query practice
- Test-driven development with pytest
- Code formatting and linting with Black and Ruff

## Current Quality Checks

This repository currently includes:

- Tested reference solutions for Python fundamentals, NumPy, pandas, probability, statistics, algorithms, plotting, SQL, and web programming
- scikit-learn extension implementations for logistic classification, thresholded profit optimization, and hurdle modeling
- GitHub Actions CI for Black, Ruff, and pytest

The goal is to build the repository gradually into a tested reference set for data science programming practice.

## Running the Checks Locally

```bash
python -m pip install -e ".[dev]"
python -m black --check solutions tests
python -m ruff check solutions tests
python -m pytest
