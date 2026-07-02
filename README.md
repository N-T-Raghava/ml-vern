# mlvern

[![PyPI Version](https://img.shields.io/pypi/v/mlvern)](https://pypi.org/project/mlvern/)
[![PyPI Downloads](https://img.shields.io/pepy/dt/mlvern)](https://pypi.org/project/mlvern/)
[![Documentation Status](http://readthedocs.org/projects/ml-vern/badge/?version=latest)](http://ml-vern.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/github/actions/workflow/status/N-T-Raghava/ml-vern/test.yml)](https://github.com/N-T-Raghava/ml-vern/actions)
[![Coverage Status](https://coveralls.io/repos/github/N-T-Raghava/ml-vern/badge.svg?branch=main)](https://coveralls.io/github/N-T-Raghava/ml-vern?branch=main)
[![codecov](https://codecov.io/gh/N-T-Raghava/ml-vern/graph/badge.svg?token=KZIEXD9ALC)](https://codecov.io/gh/N-T-Raghava/ml-vern)
[![Type Checked](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow)](https://github.com/PyCQA/bandit)
[![License](https://img.shields.io/github/license/N-T-Raghava/ml-vern)](LICENSE)

mlvern is a Python library for structuring machine learning workflows with consistent dataset handling, experiment tracking, and model management.

It provides a lightweight framework to organize ML projects by separating data processing, experimentation, and evaluation into reproducible units.

---

## Documentation

https://ml-vern.readthedocs.io/en/latest/

## Key Features

- Dataset registration with fingerprint-based identification  
- Metadata tracking for datasets and experiments  
- Structured experiment execution workflow  
- Model artifact storage and retrieval  
- Evaluation tracking and comparison across runs  
- Simple prediction interface for trained models  
- Utilities for dataset inspection and validation  

---

## Design Goals

mlvern is built around the following principles:

- Reproducibility: identical inputs produce identical tracked outputs  
- Traceability: datasets, experiments, and models are versioned and linked  
- Simplicity: minimal API surface with explicit behavior  
- Separation of concerns: data, training, and evaluation are decoupled  
- Lightweight structure: avoids unnecessary abstraction layers  

---

## Installation

```bash
pip install mlvern
```

## Quick Usage

```python
from mlvern import Forge

forge = Forge("your_project", "your_base_dir")
forge.init()
dataset_fp, _ = forge.register_dataset(df, "target")
run_id, metrics = forge.run(model, X_train, y_train, X_val, y_val, config, dataset_fp)

from mlvern import ModelComparator
ModelComparator(forge).compare_models([run_id])
```

## Requirements
Python 3.8+, NumPy, Pandas
