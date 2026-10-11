# Contributing to FlowInsight

> **NOTICE: SYNTHETIC DATA GENERATION**
> Synthetic data generated for demonstration. Not real EDA tool logs.

Thank you for contributing to FlowInsight! This document outlines coding standards, Git workflow, testing requirements, and quality checks.

---

## 1. Development Principles

- **No Hardcoded Metrics**: Every evaluation metric must be computed dynamically by model evaluation pipelines or API endpoints. Never hardcode, estimate, or invent numbers.
- **Explicit Synthetic Notice**: Synthetic data generated for demonstration. Never claim models work on real production logs without explicit genuine training data.
- **Simple, Readable Engineering**: Prefer standard, explainable algorithms (scikit-learn, regular expressions, SQLite) over opaque abstractions.

---

## 2. Git Workflow

- **Branching**: Develop features in isolated branches named `feature/<phase-name>` or `fix/<bug-name>`.
- **Commits**: Small, atomic commits with conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- **Pull Requests**: Every PR must include a comprehensive description of changes, real test output, and verification results.

---

## 3. Local Setup & Testing

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies and local package
pip install -r requirements.txt
pip install -e .

# 3. Run linting
ruff check flowinsight tests

# 4. Run test suite with coverage
pytest --cov=flowinsight --cov-report=term-missing
```

---

## 4. Code Standards

- **Python**: Python 3.11+.
- **Formatting & Linting**: `ruff format` and `ruff check`.
- **Typing**: Use standard Python type annotations on public functions.
