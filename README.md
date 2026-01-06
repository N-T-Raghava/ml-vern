# mlvern

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
