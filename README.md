# mlvern

[![PyPI Version](https://img.shields.io/pypi/v/mlvern)](https://pypi.org/project/mlvern/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlvern)](https://pypi.org/project/mlvern/)
[![Documentation Status](http://readthedocs.org/projects/auto-ml/badge/?version=latest)](http://auto-ml.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/github/actions/workflow/status/N-T-Raghava/ml-vern/test.yml)](https://github.com/<username>/ml-vern/actions)
[![Coverage Status](https://coveralls.io/repos/github/N-T-Raghava/ml-vern/badge.svg?branch=main)](https://coveralls.io/github/N-T-Raghava/ml-vern?branch=main)
[![codecov](https://codecov.io/gh/N-T-Raghava/ml-vern/graph/badge.svg?token=KZIEXD9ALC)](https://codecov.io/gh/N-T-Raghava/ml-vern)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]((https://img.shields.io/github/license/mashape/apistatus.svg))

mlvern is a lightweight Python framework for building reproducible and well-organized machine learning workflows. It provides clear tooling for dataset management, experiment tracking, model versioning, and evaluation reporting.

Project documentation: https://ml-vern.readthedocs.io/en/latest/

---

## Purpose

Machine learning projects often become difficult to maintain due to scattered datasets, untracked experiments, and inconsistent model artifacts. mlvern addresses these problems by offering a simple and deterministic project structure where heavy data inspection and analysis are performed only once per unique dataset fingerprint.

The framework is suitable for:
- Individual ML practitioners
- Research prototyping
- Academic projects
- Small to medium ML teams

---

## Core Capabilities

- Dataset registration and fingerprinting  
- Persistent metadata storage  
- Automated exploratory data analysis  
- Experiment run management  
- Model artifact registry  
- Standardized prediction interface  
- Evaluation and metric comparison  
- Cleanup and pruning utilities  

---

## Development Philosophy

mlvern follows these principles:

- Deterministic dataset fingerprinting
- One-time heavy data inspection
- Minimal and explicit APIs
- Clear artifact organization
- Easy comparison between runs
- Simple prediction and evaluation helpers

## Installation

Install the latest stable version from PyPI:

```bash
pip install mlvern
