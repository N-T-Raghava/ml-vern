# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-01-06

### Added

#### API Functions & Dataset Management
- `get_dataset_path(dataset_hash)` - Retrieve filesystem path for datasets by hash
- `load_dataset(dataset_hash)` - Load dataset metadata, schema, and analytical reports
- `save_dataset(df, name, metadata)` - Persist processed datasets with full provenance tracking
- Enhanced dataset introspection and querying capabilities

#### Model Registry & Versioning
- `register_model(model, metadata, model_id)` - Register trained models with comprehensive metadata
- `list_models()` - Query all registered models in the project
- Full model artifact management and retrieval system
- Model metadata tracking with creation timestamps and artifact paths

#### Run Management & Tagging
- `tag_run(run_id, tags)` - Annotate experiment runs with custom metadata tags
- `get_run_tags(run_id)` - Retrieve tags for runs for easy filtering and organization
- `list_runs()` - Query all runs in the project with detailed metadata
- `get_run(run_id)` - Retrieve comprehensive run information and configuration
- `get_run_metrics(run_id)` - Extract metrics from completed runs
- `get_run_artifacts(run_id)` - Access model, configuration, and training artifacts
- `remove_run(run_id, confirm)` - Safe cleanup with confirmation prompts
- Run search and filtering by custom tags

#### Inference & Evaluation
- `predict(run_id_or_model, X_test)` - Generate predictions using registered or archived models
- `evaluate(run_id_or_model, X_test, y_test)` - Comprehensive model evaluation with multi-metric analysis
- Batch prediction support and structured evaluation reports
- Automatic evaluation artifact persistence

#### Project Management & Analytics
- `get_project_stats()` - High-level project health metrics
  - Total datasets, runs, and models
  - Storage utilization analysis
  - Run success/failure ratios
  - Dataset and artifact sizes
- `prune_datasets(older_than_days, confirm)` - Intelligent cleanup of old datasets
- Configurable retention policies and safe deletion with confirmation

#### Data Quality & Risk Assessment
- Automated data inspection and validation checks
- Risk assessment module for data drift detection (KS test, PSI)
- Missing value analysis and imputation recommendations
- Outlier detection and statistical anomaly reporting
- Data quality scoring and profiling

#### Advanced Statistical Analysis
- VIF (Variance Inflation Factor) calculation for multicollinearity detection
- Redundant feature detection with correlation thresholds
- Interaction pattern discovery via pairwise analysis
- Dimensionality assessment and reduction recommendations
- Comprehensive hypothesis testing (t-tests, correlation analysis)

#### Exploratory Data Analysis (EDA)
- Automated EDA report generation with structured insights
- Interactive visualizations and statistical summaries
- Multi-format report export (JSON, HTML)
- Dataset profiling with distribution analysis
- Feature relationship mapping and correlation matrices

### Changed
- Enhanced project initialization with robust directory structure
- Improved error handling and validation throughout API
- Better logging and debugging capabilities
- More comprehensive docstrings and type hints for all public APIs
- Strengthened artifact storage and retrieval system

### Improved
- Overall API design with consistent naming conventions
- Registry management with transaction safety
- Data handling for edge cases in statistical functions
- Error messages for better debugging and user guidance

### Security
- Enhanced validation of input data and parameters
- Better handling of edge cases and invalid inputs
- Improved artifact access controls

---

## [0.1.8] - 2025-12-15

### Added
- Initial release with core functionality
- `Forge` class for ML workflow management
- Dataset registration and management
- Basic experiment tracking with runs
- Model training and evaluation support
- Automated data inspection
- Extended EDA module with structured reports
- Data validation checks
- Project initialization and setup
- Registry-based tracking system

### Features
- Dataset fingerprinting and hashing
- Artifact storage and management
- Configuration tracking for reproducibility
- Basic run tagging system
- Dataset and run cleanup utilities

---

## Version History Summary

### v0.2.0 (Current)
- **Status:** Stable
- **Key Focus:** Advanced API functions, comprehensive experiment management, project analytics
- **New Methods:** 12+ new API functions
- **Breaking Changes:** None (fully backward compatible)

### v0.1.8
- **Status:** Stable
- **Key Focus:** Core framework and EDA foundation
- **Methods:** Core Forge class with basic operations

---

## Migration Guide

### From v0.1.8 to v0.2.0

No breaking changes! Your v0.1.8 code will continue to work without modifications.

**New features you can adopt:**

```python
from mlvern import Forge

forge = Forge(project="my_project")
forge.init()

# NEW: Tag your runs for better organization
forge.tag_run(run_id, {"stage": "production", "reviewed": True})

# NEW: Register and manage models
model_id = forge.register_model(model, metadata={"framework": "sklearn"})

# NEW: Get detailed run information
run_info = forge.get_run(run_id)
metrics = forge.get_run_metrics(run_id)
artifacts = forge.get_run_artifacts(run_id)

# NEW: Comprehensive evaluation
eval_results = forge.evaluate(run_id, X_test, y_test)

# NEW: Project analytics
stats = forge.get_project_stats()

# NEW: Smart dataset cleanup
pruned = forge.prune_datasets(older_than_days=30, confirm=True)
```

---

## Support

For issues, feature requests, or general feedback:
- **GitHub Issues:** [Report issues here](https://github.com/N-T-Raghava/ml-vern/issues)
- **Documentation:** [Full docs](https://ml-vern.readthedocs.io/)

---

## License

MIT License - See LICENSE file for details

---

**Last Updated:** January 6, 2026
