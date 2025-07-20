# Rethinking sktime documentation structure

In the current state, sktime documentation is comprehensive and explain most of the features of the library. However, it is worth rethinking its structure. Some negative feedbacks about sktime are focused on the documentation, since, for new users, they are really dense in information and can be confusing.

This step proposes rethinking sktime documentation with a focus on the user experience. The first exercise is defining the different personas that might be interested in the package, and how would be the best organization for their purpose. The current state is dense and might be best for the ones already familiar with Python and advanced features of dependencies, such as pandas and multiindex.

### Documentation Philosophy
The reorganization follows these key principles:

1. **Persona-oriented**: Addresses different user personas (beginners, researchers, scikit-learn users, contributors)
2. **Task-oriented**: Organizes content around what users want to accomplish
3. **Progressive disclosure**: Presents information in layers of increasing complexity
4. **Clear separation of concerns**: Distinguishes between tutorials, how-to guides, explanations, and reference material


## Personas

We can identify the following personas:

* **User A**: Student, Junior, new to timeseries
* **User B**: knows scikit-learn API, needs-production grade pipelines
* **User C**: Academic Researcher, wants to benchmark and create new models
* **User D**: OSS contributor, Python developer
* **User E**: familiar to other timeseries packages, want to compare and get to know sktime.

In general, the user will be interested in a specific timeseries task: Forecasting, Classification, Clustering, Detection, or Regression.

## Proposed information architecture

### Top-Level Information Architecture
```
docs/
├── index.md  # Home page with persona-based entry points
├── get_started/  # Quick installation and basic concepts
├── tutorials/  # Learning-oriented content (for User A primarily)
├── how_to/  # Task-oriented solutions (for Users B and E)
├── explanation/  # Understanding-oriented content (for Users C and E)
├── reference/  # Technical reference (for all users)
│   ├── api/  # API documentation
│   ├── estimator_overview.md  # Interactive estimator search tool
│   ├── glossary.md
│   └── data_formats.md
├── examples/  # Short, focused code examples (for all users)
│   └── case_studies/  # Real-world applications 
├── get_involved/  # Community and contribution
└── developer_guide/  # For contributors and developers (User D primarily)
```

### Detailed Structure with Notebook Organization
```
docs/
├── index.md  # Home page with persona-based entry points *
├── get_started/
│   ├── installation.md
│   ├── quick_start.md  # 5-min quick start for any of the 4 tasks
│   └── why_sktime.md  # Motivation and benefits

├── tutorials/  # Learning-oriented notebooks
│   ├── forecasting/
│   │   ├── forecasting_univariate_timeseries.ipynb  # 15 min *
│   │   │   # Content:
│   │   │   # - Load of a simple dataset (e.g. Airline), without exogenous variables
│   │   │   # - Brief explanation of pandas structure in sktime
│   │   │   # - Arguments for sktime forecasting API: y, fh
│   │   │   # - Show relative and absolute fh
│   │   │   # - Forecast with a simple model (e.g. exponential smoothing)
│   │   │
│   │   ├── forecasting_with_exogenous.ipynb  # 10 min *
│   │   │   # Content:
│   │   │   # - Forecast with a more advanced model (e.g. Chronos, to showcase versatility)
│   │   │   # - Forecasting with exogenous variables
│   │   │   # - Dataset with exogenous variables
│   │   │   # - all_estimators to get exogenous variables
│   │   │   # - Usage of a simple forecaster that uses exogenous variables: AutoREG
│   │   │
│   │   ├── transformations.ipynb  # 10 min *
│   │   │   # Content:
│   │   │   # - Context about transformations, motivation
│   │   │   # - Transformations for target variable (differencing, detrending)
│   │   │   # - Feature Engineering: FourierFeatures, Holidays
│   │   │
│   │   ├── pipelines.ipynb  # 10 min *
│   │   │   # Content:
│   │   │   # - Motivation: avoid data leakage, reproducibility...
│   │   │   # - Target transformations
│   │   │   # - Transformations for exogenous variables
│   │   │   # - Composition with both types of transformations
│   │   │   # - get_params and set_params for compositions
│   │   │
│   │   ├── cross_validation_and_metrics.ipynb  # 10 min *
│   │   │   # Content:
│   │   │   # - Splitters, and their plot_windows
│   │   │   # - Metrics
│   │   │   # - evaluate
│   │   │   # - Conclusion & call to click to other tutorials
│   │   │
│   │   ├── hyperparameter_tuning.ipynb  # 10 min *
│   │   │   # Content:
│   │   │   # - Tuners in sktime
│   │   │   # - Tuning of a simple model
│   │   │   # - Tuning of a composition
│   │   │   # - Cross-validation of a tuner
│   │   │
│   │   ├── probabilistic_forecasting.ipynb  # 20 min *
│   │   │   # Content:
│   │   │   # - Brief Motivation
│   │   │   # - showcase probabilistic forecasting with predict_interval using a simple forecaster
│   │   │   # - Enumerate other forecasting methods (quantiles, variance etc)
│   │   │   # - Detail briefly probabilistic forecasting behaviour with compositions
│   │   │   # - Metrics for probabilistic forecasting
│   │   │   # - Conformal wrappers and boostrapping
│   │   │
│   │   ├── forecasting_multiple_series.ipynb  # 15 min *
│   │   │   # Content:
│   │   │   # - Use a dataset with panel data, but not that large. 
│   │   │   # - Motivation
│   │   │   # - Details of pandas dataframe structure, and useful operations
│   │   │   # - Call of fit and predict with a simple univariate model
│   │   │   # - Demonstration of .forecasters_ attribute
│   │   │   # - Metrics for panel data (aggregation)
│   │   │   # - Conclusion and connection to global forecasting
│   │   │
│   │   ├── forecasting_with_sklearn.ipynb  # 15 min *
│   │   │   # Content:
│   │   │   # - dataset loading
│   │   │   # - window_summarizer, make_reduction
│   │   │   # - Why differencing can be useful: capturing the trend
│   │   │   # - Pipeline of transformations + reduction
│   │   │   # - get_params and its recursive behaviour
│   │   │
│   │   ├── global_forecasting.ipynb  # 15 min *
│   │   │   # Content:
│   │   │   # - Definition of global forecasting
│   │   │   # - Global forecasting with Reduction Forecasters
│   │   │   # - Global forecasting with Deep Learning
│   │   │   # - Zero-shot forecasting
│   │   │
│   │   └── hierarchical_forecasting.ipynb  # 15 min *
│   │       # Content:
│   │       # - Context of the problem
│   │       # - Reconciliation strategies
│   │       # - Reconciliation transformations (new API)
│   │       # - ReconcileForecaster and mint
│   │
│   ├── classification/
│   │   ├── introduction_to_classification.ipynb *
│   │   └── advanced_classification.ipynb *
│   │
│   ├── detection_segmentation/
│   │   ├── anomaly_detection.ipynb *
│   │   ├── change_point_detection.ipynb *
│   │   └── segmentation.ipynb *
│   │
│   ├── clustering/
│   │   └── time_series_clustering.ipynb *
│   │
│   └── data_types/
│       └── mtypes_and_scitypes.ipynb  # 10 min *
│           # Content:
│           # - What are mtypes and scitypes
│           # - How to use polars with sktime
│
├── how_to/  # Task-oriented guides
│   ├── forecasting/
│   │   ├── create_custom_forecaster.ipynb *
│   │   ├── tune_forecaster_parameters.ipynb *
│   │   ├── build_ensemble_forecaster.ipynb *
│   │   ├── use_probabilistic_forecasting.ipynb *
│   │   └── reconcile_hierarchical_forecasts.ipynb *
│   ├── classification/
│   │   ├── create_custom_classifier.ipynb *
│   │   ├── handle_multivariate_classification.ipynb *
│   │   └── use_deep_learning_classification.ipynb *
│   ├── transformations/
│   │   ├── create_custom_transformer.ipynb *
│   │   ├── extract_features.ipynb *
│   │   └── compose_transformers.ipynb *
│   ├── data/
│   │   ├── load_and_format_data.ipynb *
│   │   ├── handle_missing_values.ipynb *
│   │   └── convert_between_data_formats.ipynb *
│   ├── pipelines/
│   │   ├── build_forecasting_pipeline.ipynb *
│   │   ├── create_automl_pipeline.ipynb *
│   │   └── optimize_performance.ipynb *
│   ├── benchmarking/
│   │   ├── benchmark_forecasters.ipynb *
│   │   ├── benchmark_classifiers.ipynb *
│   │   └── create_custom_benchmark.ipynb *
│   └── advanced/
│       ├── use_clustering_with_forecasters.ipynb *
│       └── cross_validate_global_models.ipynb *
│
├── explanation/  # Understanding-oriented content
│   ├── core_concepts/
│   │   ├── time_series_data_types.ipynb *
│   │   ├── forecasting_framework.ipynb *
│   │   ├── classification_framework.ipynb *
│   │   └── transformation_framework.ipynb *
│   ├── algorithms/
│   │   ├── forecasting_algorithms.ipynb *
│   │   ├── classification_algorithms.ipynb *
│   │   └── transformation_algorithms.ipynb *
│   └── design/
│       ├── composability_principles.ipynb *
│       ├── reduction_approaches.ipynb *
│       └── scitype_design.ipynb *
│
├── reference/  # Information-oriented
│   ├── api/  # Keep existing API reference
│   │   └── ... (existing API structure)
│   ├── estimator_overview.md  # From source/estimator_overview.md - INTERACTIVE TOOL
│   ├── glossary.md  # From source/glossary.rst
│   └── data_formats.md  # From source/api_reference/data_format.rst
│
├── examples/  # Short, focused code examples
│   ├── forecasting/
│   │   ├── simple_univariate.py *
│   │   ├── arima_with_exogenous.py *
│   │   ├── probabilistic_forecast.py *
│   │   └── hierarchical_reconciliation.py *
│   ├──  case_studies/  # Real-world applications
│      │   ├── energy_forecasting.ipynb *
│      │   ├── financial_timeseries.ipynb *
│      │   ├── retail_forecasting.ipynb *
│      │   └── healthcare_applications.ipynb *
│   └── ... (other example directories)
├── get_involved/  # Community contribution
│   ├── contributing.md  # From source/get_involved/contributing.rst
│   ├── governance.md  # From source/get_involved/governance.rst
│   ├── mentoring.md  # From source/get_involved/mentoring.rst
│   └── enhancement_proposals.md  # From source/contributing/enhancement_proposals.rst
│
└── developer_guide/  # For contributors
    ├── developer_installation.md
    ├── git_workflow.md
    ├── add_estimators.md
    ├── testing_framework.md
    ├── documentation.md
    └── code_reviews.md
```



### Home

This is what users see first when they access sktime's documentation. We should highlight the versatility of the package. A suggestion would be highlighting how one can change `NaiveForecaster` to `Chronos`/`NBEATS` in 1 line of code to produce forecasts with a state-of-the-art model.

We could also have links for the different personas, e.g.:


* New to timeseries forecasting? Go to the univariate forecasting tutorial
* Researcher? Check-out how to benchmark your models
* Scikit-learn user? Learn how to use regressors for forecasting
* Want to contribute? Check-out the contributing guidelines

Also adding discord, linkedin urls.

### Get started

* Installation
* 5-min quick start for any of the 4 tasks (Forecasting, Classification, Clustering, Regression)
* Why sktime? Showcase the motivation of sktime, and why to use it, the benefits of its APIs and community



### Implementation Priorities

1. **First Phase: Forecasting Tutorials**

* This is the most developed area in the existing proposal
* Begin with the first 3-4 forecasting tutorials as they form a logical sequence
* This addresses User A (beginners) and provides immediate learning value

2. **Second Phase: How-To Guides**

* Focus on forecasting how-to guides first
* These provide practical problem-solving help for Users B and E
* Prioritize guides that answer common questions from the community

3. **Third Phase: Reference and Home Page**

* Reorganize the estimator overview, glossary, and data formats documentation
* Create the persona-based home page to improve navigation
* This improves the experience for all users by providing clear entry points

4. **Fourth Phase: Additional Tutorials and Explanations**

* Expand to classification, clustering, and detection tutorials
* Add explanation content to provide deeper understanding
* This addresses the needs of User C (researchers)

5. **Fifth Phase: Get Involved and Developer Guide**

* Reorganize contribution and development documentation
* This primarily benefits User D (contributors)

### Migration Approach

When migrating existing content:

1. **Preserve valuable content:** Existing tutorials and examples contain valuable information that should be preserved
2. **Restructure for clarity:** Break long notebooks into smaller, focused ones according to the detailed outline
3. **Use templates:** Create templates for each documentation type to ensure consistency
4. **Add cross-references:** Ensure connections between related content across sections

A branch `dev-docs` will be created and will be used as temporary  development branch for this documentation improvement. Contributors should create PRs from this branch, that will progressively incorporate such changes, until we reach a state that is good enough to be put live.

### Next Steps
1. Create the documentation develop branch
2. Create directories and reorder the navigation menu on that branch.
3. Create the umbrella issue and ask for contributions.
4. Start with the forecasting tutorials, following the detailed content plan
5. Continue implementing the other tutorials, and merge to master as soon as documentation is good enough.
6. Iterate to improve the documentation and add new content.
