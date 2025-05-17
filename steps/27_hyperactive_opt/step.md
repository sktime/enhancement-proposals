

## Motivation

* hyperactive currently has an abstract interface to multiple gfo optimizers
* with minor architectural changes, this could support a wide range of opt algorithms useful for ML
* a package like that does not exist yet, but would be hugely helpful for the sklearn ecosystem
* it would also improve user experience of applying GFO to ML/data problems


## High-level goals

* unified interface for objective functions (and constraints)
  * interoperable with all optimizer packages
* unified interface for optimizer packages (including gfo, optuna, "simple algos" such as grid search)
* sklearn integration for tuning where abstract optimizer can be used (via "sklearn objective function")
* sklearn-compatible interfaces throughout

## Requirements

* extensibility of unified API - users can write their own objective functions, optimizers easily
* extensibility of integrations - integrating with ML tuning packages is easy
* support for current gfo interfaces should be as complete as possible
* sklearn-like API throughout - get_params, nesting, composability
* tag system and soft dependency isolation (e.g., optuna, talos)


## Conceptual model

key objects in the conceptual model are:

#### "abstract" optimization algorithms (individual algorithms)

* examples: grid search, hill climbing, Bayesian optimization, random search

#### implmentations of optimization algorithms, they "live" in implementations, in different packages, with specific parametrization!
* examples: grid search as implemented in sklearn; gfo implements hill climbing and genetic algorithms; Bayesian optimization is implemented in gfo; optuna implements "tree of Parzen estimators", gfo also implements this!
  * there is an n:m relationship, the same algorithm can be implemented in multiple packages, and a given package can implement multiple algorihtms.
  * packages implement different parametrizations and API, even if implementing the "same" family of algorithm

#### objective functions and "experiments" to optimize
* abstract input-output with specified variable names and types for input and output; can be parametric
* examples:
  * Sphere function; parameter n_dim; objective variables in R^n, objective value (output) = objective
  * sklearn cross-validation experiment; parameters: data, estimator class, cross-validation setup; objective variables: parameter values of the estimator; objective value (output) = evaluated score on the cross-validation experiment
  * statistical estimation; parameter = histogram of empirical distribution, parametric shape of distirbution to be fitted to histogram; objective variables = distribution parameters (e.g., mean, variance of normal); objective value (output) = goodness-of-fit, e.g., likelihood of fit.

terminology:
* "objective input" for function input (variables) of the objective function/experiment
* "experiment parameters" for variable objects that are not directly input to the objective function
* "objective result" for output of the objective function/experiment, including metadata
* "objective value" for output of the objective function, numerical/scalar

#### search spaces, search instructions, and search constraints

* they could belong to an "experiment" but are typically not considered part of the objective function
* they could belong to the configuration of an optimization algorihtm
* examples:
  * hypersquare of numerical values for sphere function
  * for an neural-network, specifying a list of network layers to choose from during optimization.
  * for an sklearn estimator, specifying a specific grid or tree-of-grid in grid search, e.g., for SVC, we say: we optimize over kernel being poly and Gaussian; if poly, degree = [1, 2, 3]; if Gaussian, then we optimize over gamma
    * algorithm specific, to grid search and other discrete searches
  * for an sklearn estimator, specifying a distribution in random search, e.g., uniform or Beta in a rectangular range over C and gamma for SVC.
    * algorithm specific, to random search and other distribution based searches
  * in statistical estimation, search ranges are implied by constraints on the distribution parameters, e.g., variance is non-negative. Additional constraints could be added.
  * in statistical estimation, some searches can be given joint priors on the distribution parameters, which counts as a "search space".
* QUESTION: do we separate the following three concepts?
  * general search space, e.g., a set of parameters
  * algorithm specific search space which is a subste of general search space. E.g., general search space can be continuous, but grid or discrete algorithms require a finite or at least countable set of points.
  * search space related specifications, e.g., distributions over the search space.
  
**LACK OF CONCEPTUAL CLARITY HERE - NEED TO REVISIT?**

#### "property tags" abstract algorithms, implementations, and objective functions have property tags

* examples:
  * capability to use a gradient. gfo does not use gradient. Newton method uses gradient.
  * implementations may require python dependencies. gfo implementations requires gfo package, optuna tree of Parzen requires optuna, etc.
  * experiments may be deterministic or not. For instance, sphere function is deterministic; sklearn experiment has a randomized outcome in general.

### Key operations and interactions:

* evaluate an objective function on an input
* use the *implementation* of an optimization algorithm (any) to find a good solution to the experiment/objective
  * return the "best parameters"
  * report on the optimization process
    * examples: time elapsed; interations until convergence; top 10 "best parameters"
  * the algorithm is given instructions that may or may not be specific to both algorithm and the objective function
    * in the form of parameters of the algorithm or algorithm execution?
    * examples: number of iterations in random search (most if not all algos have this); grid size in grid search when given a continuous search space (only grid based algos); eps in Newton gradient descent, or other termination criteria (only gradient based algos)
  * these instructions may contain information on a search space, which refers to the experiment
* find all algorithms or objectives with a specific tag


## API design

### high-level design principles

* strategy pattern for unified API
* template pattern for extender API, `score` / `_score` or `fit` / `_fit`
* sklearn-like API, e.g., inheriting from skbase BaseObject
* objects per main concepts, "domain driven" design
  * optimization algo (concrete), experiment
  * search space? - maybe class, maybe leave algo specific?


### interfaces and classes

* optimizer - strategy/template pattern, skbase
  * base class for "all optimizers" is interface, concrete classes inherit, follows strategy pattern
  * optional intermediate classes for one package, e.g., gfo or other multi-optimizer packages
  * full template pattern applied at least to business logic (the algorithm implementation)
  * possible second layer of template pattern for multi-algorithm packages - template on package level
  * sklearn-like API added through skbase BaseObject
    * "parameters" and instructions map on get_params/set_params?
* experiment / objective
  * one base class interface, covers both classical objective functions and "experiments" like sklearn
  * concrete objectives/experiments inherit, follows strategy pattern
  * full template pattern with express requirement to make writing user defined experiments very easy
  * template pattern applied to the objective and parametrization
  * sklearn-like API added through skbase BaseObject
    * "parameters" map onto get_params/set_params
* search space does not need to be a separate class on "top design level"
  * only for the first design, could be changed later (design upwards compatible)
  * individual optimizers can handle this as they want
* optimizers and experiments are all tagged
  * tag system from skbase
  * need to decide on case-by-case basis which tags map onto differences in API contract  


### "sklearn compatibility"

#### What does this mean?

For all "unified objects", i.e., where strategy pattern is used:

* nested parameter inspection, `__init__`, `get_params`, `set_params` convention
* nested pretty printing via `__repr__` etc
* tag system via `skbase`

Consequence: `hyperactive` optimizers can be used seamlessly as components of `sklearn` estimators and beyond (e.g., the tuners)

#### What it does **not** mean:

* compatibility with `sklearn` interface APIs such as classifiers, regressors (for optimizers, objectives, etc)
* direct dependency on `sklearn`

It does not mean: "all `hyperactive` objects become classifiers/regressors"

(except for, of course, the tuner for `sklearn`)


### usage cases

* using and evaluating a pre-defined objective
* defining a new objective/experiment
* using an optimizer to solve an objective in unified API
    * grid search
    * gfo
    * optuna
* implementing a new optimizer (3rd party, power user)?
* using the sklearn integration, fit/predict
    * grid search
    * gfo
    * optuna
* searching for optimizers using tag system

### Design - "objective function" interface

#### user journey designs


##### objective function - usage

```python
from hyperactive.experiment.toy import Ackley

# defining an instance of Ackley with a fixed A
# A is an "experiment parameter", these are passed at the start
ack = Ackley(A=2)

# we can inspect the names of objective inputs
ack.paramnames()
# ["x0", "x1"]

# we can also inspect the experiment parameters with values
ack.get_params()
# {"A": 2}

# using "score" evaluates the objective - we pass a dict like in hyperactive currently
ack.score({"x0": 2, "x1": 42})
# (123456 (or whatever is the correct output), {"some": "metadata"})

# __call__ is syntactic sugar, quick-call
ack(x0=2, x1=42)
# 123456
# (this is the objective value only)
```

SB comments:
* `get_params` and `paramnames` - should we change names?
  * FK: `get_params` is fixed through sklearn expectation - same as `__init__` args
  * `paramnames` - perhaps a good idea to rename. `score_params`? `score_paramnames`


##### objective function - extension, implementing new objective

###### FK design

FK thinks there could be two options:

* option 1: user defines class, uses extension template
* option 2: user just passes a function (e.g., to optimizer), under the hood it gets wrapped in a class, so the user does not have to define a class.

option 1 - implement a class. User fills in an "extension template" which looks like this


```python
# FIXED PART start
from hyperactive.base import BaseExperiment


# TODO: change name of MyExperiment
class MyExperiment(BaseExperiment):
    """TODO: fill in docstring with parameters."""

    # TODO: fill in all experiment parameters. If none, delete the __init__
    def __init__(self, param1, param2=paramdefault):
        self.param1 = param1
        self.param2 = param2
        super().__init__()

    # TODO: fill in this function
    def _paramnames(self):
        # logic in this function can access the experiment params in self
        return [something]  # this must return a list of str

    # TODO: fill in this function
    def _score(params):
        param1 = params["param1"]
        param2 = params["param2"]
        obj_value, metadata = do_something_with(param1, param2)
        # metadata is a dict of metadata
        # obj_value is a float
        return obj_value, metadata
```


Option 2: passing a function (additive, on top of option 1 for quick specification)


```python
def objective_function(x1, x2):
    return do_something(x1, x2)


# usage is as follows, in any place where otherwise a class instance would be passed
my_optimizer(objective_function)

# internally this wraps objective_function in a class
#    _paramnames is obtained from inspecting objective_function
#    _score is same as executing objective_function
#    this way, no "experiment parameters" are possible
```


###### SB design



```python

```

```python
search_space = {
  "x0": (0, 10),
  "x1": [0,1,2,3],
}

# FIXED PART start
from hyperactive.base import BaseExperiment


# TODO: change name of MyExperiment
class MyExperiment(BaseExperiment):
    def setup(self, param1, param2):
      self.param1 = param1
      self.param2 = param2

    # TODO: _score name?
    def _score(params):
        return obj_value, metadata
        
        
my_experiment = MyExperiment(search_space)
```

`query`?

FK comments:
* `setup` content must be in `__init__` or we will lose `sklearn` compatibility
* `_score` content is still the same? `params` being a dict?



#### class API

### Design - "optimizer" interface

#### user journey designs
Example: grid search applied to scikit-learn parameter tuning:

1. defining the experiment to optimize:

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.datasets import load_iris
from sklearn.svm import SVC
>>>
X, y = load_iris(return_X_y=True)
>>>
sklearn_exp = SklearnCvExperiment(
    estimator=SVC(),
    X=X,
    y=y,
)
```
2. setting up the grid search optimizer:

```python
from hyperactive.opt import GridSearch
param_grid = {
   "C": [0.01, 0.1, 1, 10],
   "gamma": [0.0001, 0.01, 0.1, 1, 10],
}
grid_search = GridSearch(param_grid, experiment=sklearn_exp)
```


3. running the grid search :
```python
best_params = grid_search.run()

# Best parameters can also be accessed via the attributes:
best_params = grid_search.best_params_
```

#### class API

* `sklearn`-like, dataclass-like (`skbase` inheriting)
* `__init__` *must* always have arg `experiment=None` as last arg
* `run` without args runs the experiment

```python
def run(self):
    """Run the optimization search process.

    Returns
    -------
    best_params : dict
        dict with keys as experiment paramnames (or subste thereof)
```



### Design - sklearn integration

#### user journey designs
Tuning sklearn SVC via grid search

1. defining the tuned estimator:

```python
from sklearn.svm import SVC
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt import GridSearch

param_grid = {"kernel": ["linear", "rbf"], "C": [1, 10]}
tuned_svc = OptCV(SVC(), GridSearch(param_grid))
```

2. fitting the tuned estimator:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tuned_svc.fit(X_train, y_train)

y_pred = tuned_svc.predict(X_test)
```

3. obtaining best parameters and best estimator
```python
best_params = tuned_svc.best_params_
```

#### class API




## Implementation plan

### Phase 1 - unified API for optimizers and experiments

* added base API and tests for interface conformance
* new modules `experiment`, `opt`
* populating modules with concrete classes - experiments and optimizers
* test framework for experiments and optimizers

phase 1 does not include:

* rework of constraints or search space interfaces
* not necessary: major rework of `hyperactive` interface except where it helps 1b

##### PR 1a: API and test skeleton

1st PR adds API and a few concrete classes for testing. No other changes
* could be build on this: https://github.com/SimonBlanke/Hyperactive/pull/110
* FK to add: test classes

No other changes in this PR!

##### PR 1b: gfo class API

* all hyperactive optimizers get interfaces from individual classes
* one class per optimizer, inheriting from `BaseOptimizer`
* exact same interface as `HillClimbing(BaseOptiimzer)` in PR 110
    * following extension template or class
    * use `check_estimator` to check conformance

Conformance is tested using test framework from 1a (automatic).

IMPORTANT: this does not need to mean we need to move the entire logic to the classes.
We just need the classes as interfaces.

##### PR 1b.5: Scheduler and memory backend

* memory base class
* memory base class API tester
* 2 memory classes
  * hash dict
  * shared hash dict
* scheduler

Requires: 1a?

Open Q: should shared memory be modelled via memory class? Or should it be handled via multiton wrapper (in scheduler? wrapping experiment?)


##### PR 1c - `sklearn` integration with abstract optimizer

FK will do this


##### PR 1d - optional hyperactive/gfo API rework

* if desired, gfo interface can be refactored

Would strongly suggest to do this after 1b.

Can be based on https://github.com/SimonBlanke/Hyperactive/pull/101

##### PR 1e - simplification refactor of unified API

Optional.

The PR 1b, 1d will have likely led to some code duplication in gfo facing estimators.

Optimally, this can be refactored to a middle layer - or, 1d may have already don this.


##### PR 1f - optional: parallelization unified API

sktime parallelization API -> move to skbase

then refactor hyperactive native grid and random search to use this

from skbase

Requires: 1a (and a random search implementation)


#### "must do" from FK perspective

* new base classes `BaseExperiment` and `BaseOptimizer` (done by FK)
* test framework for API conformance (done by FK)
* gfo facing concrete classes inheriting from `BaseOptimizer`

#### optional (for FK)

* contents of folders other than `base`, `experiment`, `opt`
* name of module - `experiment` or `objectives`?
* name of module - `opt` or `optimizer` or `optimizers`?
* names of some methods, e.g., `paramnames`, `run`, `search`, `query`
    * this excludes `sklearn`-like methods which "need to be there": `get_params`, `set_params`
* granular structure of `optimizer` module, e.g., one `gfo` folder or "flat" structure
    * but should be "one class per py file"


### Phase 2 - adding integrations

* adding external optimizers the community considers important
* more integrations with popular ML/AI packages

phase 2 does not include:

* rework of constraints or search space interfaces

PR can be worked on in any order


##### PR 2a: optuna optimizer

* optuna optimizer in `BaseOptimizer` interface
* soft dependencies isolated
* we use this as proof-of-concept for soft dependency isolation


##### PR 2b: another backend? Optional

hyperopt? skopt?


##### PR 2c: sktime integration

new `sktime` estimators.

Proof-of-concept for tuning across mulitple machine learning tasks.

Requires:

* `SktimeForecastingExperiment`
* `SktimeClassificationExperiment`

Follows principle in 1c, e.g.,

```python
SktimeForecastingExperiment(
    all_params_of_evaluate,
)

ForecastingHyperactiveCV(
    tuner,
    all_params_of_evaluate_except_data,
)
```

Optional: `skpro`

##### PR 2d: surfaces integration

All problems from `surfaces` should be available in `hyperactive` as classes
inheriting from `BaseExperiment`.

This can be by moving, or simply by interfacing.

#### "must do" from FK perspective

* at least one, better two external optimizer integrations
* soft dependency handling patterns are consolidated
* at least one "experiment" integration, e.g., `surfaces`
* at least one "estimator" integration, e.g., `sktime`

#### optional (for FK)

same as in phase 1


### Phase 3 - revisiting search spaces

The above examples of optimizers will hopefully shed more light on how we can model
search spaces and search configurations.

I would rescommend this is the point where we revisit these questions and try to
develop a more stringent software API for search spaces.


## Open questions

* name of `score` - `query` (evaluate the optimization function)
* how and where to deal with search spaces
  * option 1: it is up to the definition of experiment and optimizer, where and if it needs to be passed
  * option 2: always must be passed or set in experiment (attached to objective)
  * option 3: always must be passed to optimizer, if optimizer requires; never set in experiment
  * question is also, which option is better to "leave our options open" to close this in the future and not now
  


