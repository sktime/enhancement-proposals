These are the notes on API requirements for a benchmarking API from issue: https://github.com/alan-turing-institute/sktime/issues/141

---
## Evaluation 
- [ ]  Extend evaluation API and functionality to single dataset case, currently only multiple dataset case is supported

---
## Performance metrics 
A few implementation notes:
   - Vectorised vs iterative computations
   - Callable classes vs classes with methods for computation
   - Use of jackknife by default for non point-wise metrics
   - Computation of standard error as decorator/mix-in
   - Have separate classes for point-wise metrics which can be wrapped by aggregation functions (e.g. mean)

Also see https://github.com/JuliaML/LossFunctions.jl.

---
## Orchestration
### Should have
- [ ] Allow orchestrator to be persisted to replicate benchmarking studies
- [ ] add unit tests for `evaluator` methods
- [ ] update all methods on evaluator to work on new internal data representation, also see  https://www.statsmodels.org/stable/stats.html for some additional test implementations, e.g. the sign test, to improve readability, so that we can deprecate `_get_metrics_per_estimator_dataset` and `_get_metrics_per_estimator` methods
- [ ] for saving results inside the `orchestrator` and for loading results in results classes use `_ResultsWrapper` to simply/unify interface, `_ResultsWrapper` needs to have slots for at least: y_true, y_pred, y_proba, index, fit_time, predict_time, strategy_name, dataset_name, cv_fold, train_or_test
- [ ]  No timing of fit and predict available, see https://docs.python.org/3/library/time.html#time.perf_counter, potentially have new `save_timings` and `load_timings` method
- [ ] `orchestrator` cannot make probabilistic predictions, orchestrator tries to make probabilistic predictions using `predict_proba`, but (i) this will only works for some but not all classifiers and it won't work in regression, (ii) strategies currently don't even have a `predict_proba` (not even `TSCStrategy`), and (iii) current computation of `y_proba` fails if `y_pred` contains strings instead of integers which however is an accepted output format for classification I believe, add `predict_proba` to `TSCStrategy`
- [ ] handling of probabilistic metrics in `evaluator`
- [ ] no longer sure that saving results object as a master file is a good idea, as it may cause problems when multiple processes try to update it and because it needs to reflect the state of the directory somehow, maybe better to have a method on results object that allow to infer datasets, strategies and so on, something like a `register_results` method, instead of loading a fully specified dumped result object 
- [ ] separate `predict` method on `orchestrator` which loads and uses already fitted strategies
- [ ] fix UEA results class  

### Could have
- [ ] allow for pre-defined cv splits in files
- [ ] allow for pre-defined tasks in files 
- [ ] add `random_state` as input arg to orchestrator which is propagated to all strategies and cv 
- [ ]  perhaps also useful to catch exceptions and skip over them in `orchestrator` instead of breaking execution?
- [ ]  currently only works for ts data input format, add other use cases
- [ ]  better user feedback, logging, keeping track of progress 
- [ ] many docstrings still missing or outdated
- [ ] perhaps metrics shouldn't be wrapped in classes and the evaluator should take care of it internally, working with kwargs (e.g. `pointwise=True`)
- [ ] handling of multiple metrics in `evaluator`
- [ ] functionality for space filling parameter grids for large hyper-parameter search spaces (e.g. latin hypercube design), see this Python package: https://github.com/tirthajyoti/doepy
- [ ] monitoring and comparison of memory usage of different estimators 

Related issues/PRs: #132 
