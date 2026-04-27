# Solutions Style Guide

Submitted solutions should conform to the following standards.

## Code Style

- Format Python code with `black`.
- Lint Python code with `ruff`.
- Test solutions with `pytest`.
- Prefer readable solutions over overly clever one-liners.
- Handle reasonable edge cases explicitly.
- Raise clear exceptions when invalid input would otherwise produce misleading output.

## Documentation

- Use docstrings for public functions and classes.
- Prefer NumPy-style docstrings for larger functions and model classes.
- Include examples when they clarify expected behavior.

## Testing

Each solution should include pytest tests covering:

- the example shown in the problem statement
- at least one edge case
- invalid inputs when applicable

## File Organization

Solutions should be submitted in a module whose filename matches the problem section.

For example, solutions for the `General Programming` section should be submitted in:

```text
solutions/general_programming.py
