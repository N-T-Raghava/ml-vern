# mlvern v0.2.0 Release Notes

**Release Date:** January 6, 2026 | **Status:** Stable

## What's New

mlvern v0.2.0 introduces **20+ new API functions** with advanced experiment management, model registry, and comprehensive project analytics.

### Key Features

**Dataset Management**
- `get_dataset_path()`, `load_dataset()`, `save_dataset()` - Full dataset lifecycle control

**Model Registry**
- `register_model()`, `list_models()` - Complete model versioning and retrieval

**Run Management**
- `tag_run()`, `get_run_tags()` - Organize runs with custom metadata tags
- `get_run()`, `get_run_metrics()`, `get_run_artifacts()`, `list_runs()` - Full run visibility
- `remove_run()` - Safe cleanup with confirmation

**Inference & Evaluation**
- `predict()` - Generate predictions from archived models
- `evaluate()` - Comprehensive multi-metric evaluation

**Project Analytics**
- `get_project_stats()` - Dataset count, storage usage, run metrics
- `prune_datasets()` - Smart dataset cleanup with retention policies

**Data Quality**
- Automated data drift detection and risk assessment
- Advanced statistical analysis (VIF, redundancy, interactions)
- EDA report generation with visualizations

### API Quick Reference

```python
from mlvern import Forge

forge = Forge(project="my_project")
forge.init()

# Datasets
info, is_new = forge.register_dataset(df, target="target")
path = forge.get_dataset_path(hash)

# Models
model_id = forge.register_model(model, {"framework": "sklearn"})
models = forge.list_models()

# Runs
runs = forge.list_runs()
metrics = forge.get_run_metrics(run_id)
artifacts = forge.get_run_artifacts(run_id)

# Tags
forge.tag_run(run_id, {"stage": "production"})
tags = forge.get_run_tags(run_id)

# Evaluation
predictions = forge.predict(run_id, X_test)
results = forge.evaluate(run_id, X_test, y_test)

# Analytics
stats = forge.get_project_stats()
pruned = forge.prune_datasets(older_than_days=30, confirm=True)
```

### What Changed

- Enhanced artifact storage and registry management
- Better error handling and input validation
- Full type hints and comprehensive docstrings
- Advanced data quality and statistical analysis functions
- Automated EDA with structured reports

### Backward Compatible

✅ **No breaking changes** - v0.2.0 is fully compatible with v0.1.x

### New Dependencies

None - uses existing: pandas, scikit-learn, matplotlib, joblib

### Installation

```bash
pip install --upgrade mlvern
```

### Limitations

- Large dataset EDA (>1M rows) may take longer
- Data drift detection optimized for <100K rows
- Tags support string, numeric, and boolean values

---

**License:** MIT | **Author:** Tanmai Raghava  
[Issues](https://github.com/N-T-Raghava/ml-vern/issues) | [Docs](https://ml-vern.readthedocs.io/)
