# 100 Data Science Programming Problems

A structured collection of programming, data manipulation, statistics, modeling, SQL, plotting, and web programming exercises for building practical data science and data analyst skills.

**Status:** 53 / 100 problems drafted. Reference solutions and pytest coverage have been started for the General Programming section.

## Progress

| Section | Problems Drafted | Solutions Added | Tests Added |
|---|---:|---:|---:|
| General Programming | 13 | 13 | 13 |
| Data Manipulation - NumPy | 12 | 0 | 0 |
| Data Manipulation - pandas | 3 | 0 | 0 |
| Probability | 4 | 0 | 0 |
| Statistics | 2 | 0 | 0 |
| Algorithms | 7 | 0 | 0 |
| Plotting | 3 | 0 | 0 |
| SQL | 4 | 0 | 0 |
| Web Programming | 5 | 0 | 0 |
| **Total** | **53** | **13** | **13** |

## Running the Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── problems/
│   ├── general_programming.md
│   ├── numpy.md
│   ├── pandas.md
│   ├── probability.md
│   ├── statistics.md
│   ├── algorithms.md
│   ├── plotting.md
│   ├── sql.md
│   └── web_programming.md
├── solutions/
│   ├── __init__.py
│   └── general_programming.py
├── tests/
│   └── test_general_programming.py
└── solutions_style_guide.md
```

## Problem Sections

- [General Programming](problems/general_programming.md)
- [Data Manipulation - NumPy](problems/numpy.md)
- [Data Manipulation - pandas](problems/pandas.md)
- [Probability](problems/probability.md)
- [Statistics](problems/statistics.md)
- [Algorithms](problems/algorithms.md)
- [Plotting](problems/plotting.md)
- [SQL](problems/sql.md)
- [Web Programming](problems/web_programming.md)

## Submitting Solutions

Solutions submitted to this repo should conform to the [solutions style guide](solutions_style_guide.md).
