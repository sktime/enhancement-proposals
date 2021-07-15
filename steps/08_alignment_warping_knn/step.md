# alignment, time warping, and dtw distance based clustering/classification

Contributors: @fkiraly, 

## High-level summary 

### The Problem

Dynamic time warping is a popular technique used in time series classification and clustering.
There, it is used mainly via computation of the time warping distance matrix.

Natively, however, dynamic time warping solves the time series alignment task, a task not yet supported in `sktime`.

This STEP describes the natural reductions and required tasks for supporting all aspects of time warping.

### The Aim

We aim to support the three main incarnations of dynamic time warping (DTW) type algorithms:

* as algorithms for time series alignment
* as "DTW" style distances between time series
* as the "DTW" in clustering or classification algorithms

### The proposed solution

Our proposed solution follows the generic modularity/reduction and scityping design principles of `sktime`:

* "algorithms for time series alignment" will require a new task (alignment). Individual DTW aligners will be special cases of aligners.
* "DTW style distances" will be implemented as 2nd degree transformers that wrap certain aligners, plus shorthands
* "DTW" clustering or classification algorithms will be implemented as clusterers or classifiers that have distances as components, which can but don't need to be DTW distances.

Note that there is not just one "DTW distance" or "DTW alignment", in fact there are multiple algorithms (e.g. vanilla, gappy, incomplete, penalized, etc) under the DTW label. Further, there are aligners which are not DTW based, such as shapelet or moving window registration based.

### Related Work

There exist python packages which implement dtw algorithms such as `dtwpython` and `fastdtw`. To our knowledge there is no python package implementing sequence alignment as a learning task with `scikit-learn`-like interface - in particular not in the composable form that we would need.

`sktime` already has experimental support for time series distances and related composition patterns, and stable support for classifiers.

## User journey

We outline intended usage for two new use cases:
* specifying a DTW classifier in a modular fashion
* computing a time series alignment

Usage of classifiers or distances are not new use cases, these are as in the current `sktime` interfaces.

### User journey: specifying modular DTW classifier

A modular DTW classifier would be created as follows

```python
mydtwalign = DtwAligner(param1=1, param2=42)
mydtwdist = AlignDist(myDTWalign)
mydtwclassifier = KnnTsClassif(dist=mydtwdist, k=4)
```

where `DtwAligner` is some DTW based aligner, `mydtwalign` its instance, `AlignDist` makes distances from aligners, `mydtwdist` is a time warping distance, and `mydtwclassifier` is k-nearest neighbors using the distance `mydtwdist`, created via `KnnTsClassif`.

Of course, for popular combinations, the above could be replaced by a simple shorthands such as

```python
mydtwdist = DtwDist(param1=1, param2=42)
mydtwclassifier = KnnTsClassif(dist=mydtwdist, k=4)
```

or even

```python
mydtwclassifier = DtwKnnClassif(dist=mydtwdist, k=4, param1=1, param2=42)
```

### User journey: alignment

A time series alignment is computed between two or multiple time series in `pd.DataFrame` format.

Computing an alignment between two sequences `X`, `X2` would look as follows:

```python
mydtwalign = DtwAligner(param1=1, param2=42)
X = [X1, X2]
mydtwalign.fit(X)
alignment = mydtwalign.get_alignment()
X_aligned = mydtwalign.get_aligned()
```

Here, `DtwAligner` is an aligner class, `mydtwalign` is an instance.
`X1` and `X2` are time series in `pd.DataFrame` format (see below).
`alignment` is an index alignment, while `X_aligned` is an index aligned version of `X`, a list with two `pd.DataFrame` as elements.

## Scitype design: time series and sequence alignment

We proceed outlining the scitype design for aligners:

* a dedicated data container type for alignments
* key interface points for aligners (a scitype of estimator)

### Alignments: data container

An index alignment of `M` time series in a list `X` of `pd.DataFrame` is represented as follows:

* as a `pd.DataFrame`, referred to below as `align` (but necessarily called that)
* with $M$ columns, named `ind+str(i)` for any integer between `0` and `M-1` (inclusive)
* the `DataFrame.index` contains the alignment index
* the column `'ind'+str(i)` encodes mapping of alignment index to the index values of `X[i]` encoded by `int64` or `Int64` entries. The encoding is as follows: `align.index[j]` is mapped onto the index `X[i].index[j]`. Note this makes the entries of `align` reference to `iloc` indices, not `loc` indices.
* for partial and gappy alignments, the gap symbol is encoded by `pandas` `NA` of type `Int64` (important note: not `np.nan` which is always of type `float`
* for the alignment to be valid, all values of `X[i].index` must appear in the column `align['ind'+str(i)]` at least once, and monotonously. They can appear more often than once, which is interpreted as stretching.

Note that the alignment `align` itself does *not* store any information about the values in `X`, but will in general be computed from the values in `X`.

Example: consider two sequences, i.e., two `pandas.DataFrame`-s, `s1` and `s2` as follows: `s1` being

| index | `pressure` | `temperature` |
| --- | --- | --- |
| `1` | `2.2` |  `52.0` |
| `2` | `3.1` |  `53.4` |
| `3` | `3.4` |  `55.2` |

and `s2` being

| index | `pressure` | `temperature` |
| --- | --- | --- |
| `1` | `3.1` |  `53.4` |
| `2` | `3.4` |  `55.2` |
| `3` | `3.6` |  `55.3` |

We note that rows 2 and 3 of `s1` are identical with rows 1 and 2 of `s2`, and there is an increasing trend in all variables, so intuitively one would expect an alignment algorithm to identify `s1` as the "start" of a consensus sequence, and `s2` as the end of it, with an overlap in the middle.

A "reasonable" partial alignment of `s1` and `s2` (which is also a partial pairwise index alignment of `s1.index` and `s2.index`), is therefore

| index | `ind0` | `ind1` |
| --- | --- | --- |
| `1` |  `1` |  `NA` |
| `2` |  `2` |  `1` |
| `3` |  `3` |  `2` |
| `4` |  `NA` |  `3` |

Another possible (non-gappy), full pairwise alignment of `s1` and `s2` could be

| index | `ind0` | `ind1` |
| --- | --- | --- |
| `1` |  `1` |  `1` |
| `2` |  `2` |  `1` |
| `3` |  `3` |  `2` |
| `4` |  `3` |  `3` |

This "contracts" the unalignable ends into the aligned "middle piece" (the indices `2` and `3`).

The alignment algorithm also has freedom how to index the alignment, which could be meaningful. For instance,

| index | `ind0` | `ind1` |
| --- | --- | --- |
| `0` |  `1` |  `1` |
| `2` |  `2` |  `1` |
| `3` |  `3` |  `2` |
| `42` |  `3` |  `3` |

would also be a valid alignment, which could make more or less sense (depending on the application).

### Alignments: data container - discussion

Some design considerations:

* a possible choice for the entries of alignments would have been `loc` not `iloc` indices.
The decision for `iloc` is because the `loc` encoding can easily be recovered from the `iloc` alignment and the alignmed sequences, but this is not true the other way round. 

* allowing partial alignments is important - the only way to allow `NA`s in a `pd.DataFrame` of integer values is the `pandas` specific `Int64` type

* another option would have been to have the index as a separate column. This would have been redindant since the `pandas.Index` is the canonical index construct, and there is only one "overall" alignment index.

* one could have started counting column names at `1`, which may on first glance appear "nicer" in the frequent case of pairwise alignments (of two sequences). However, this seemed unpythonic and more error prone due to the plus-minus-`1` in indexing.

### Aligners: scitype base class design

Aligners inherit from a base class `BaseAligner`, which inherits from `sktime.BaseEstimator` (thus inheriting hyper-parameter handling and fitted state handling).

Aligners inheriting from `BaseAligner` has the following scitype defining methods:

`fit(X: listof[pd.DataFrame]) -> self`

fits the aligner to `X` - it is expected that the bottleneck computation happens here

`get_alignment() -> pd.DataFrame`

this returns the alignment of `X`. Requires `is_fitted` state.

`get_alignment_loc() -> pd.DataFrame`

this returns an alignment where entries are `loc` indices, not `iloc` indices - this is occasionally useful (e.g., for plotting or indexing). Requires `is_fitted` state.

`get_aligned() -> listof[pd.DataFrame]`

this returns an aligned version of `X`, indices in the return corresponding to indices of `X` in fit.
All elements in the return have the same index, also identical with the index of the `get_alignment` return.

`optional: get_distance() -> float`

this returns the alignment distance (if the aligner computes it). Requires `is_fitted` state.

Useful tags:

* whether the aligner can compute multiple alignments, or only pairwise alignments (the latter are more frequent). I.e., whether `X` can be of length `3` and above.
* whether the aligner computes a distance, i.e., whether `get_distance` is implemented.
* if yes, whether that distances is invariant under permutation of the time series (e.g., switching two)


### Aligners: scitype base class design - discussion

There are three/four methods that can be called post `fit`.

This could have also been solved as three/four return argument of `fit` or an equivalent function.

There are a number of reasons not to handle it this way:

* the number of the returns would depend on whether the equivalent of `get_distance` is implemented, which leaves a number of options, all ugly
* `get_alignment_loc` can easily implement default functionality in the base class which converts an alignment obtained from `get_alignment`, which allows implementation of the logic in `get_alignment_loc` to become optional
* similar for `get_aligned`, this also allows default functionality based on `get_alignment` (and remembering `X` in `fit`)
* the split allows to move the computational bottleneck into `fit`, which allows multiple calls to retrieve results to be highly efficient. This is useful in algorithms that depend on such multiple retrieval, or can make use of pre-computation, for instance composites (see user journey).

### Alignments: alignment distances including DTW distances

Given the above, an alignment distance can now be easily obtained from a dedicated wrapper `AlignDist` which is a pairwise panel transformer - i.e., inherits from `BasePairwiseTransformerPanel`.

Usage is as outlined above:

```python
mydtwalign = DtwAligner(param1=1, param2=42)
mydtwdist = AlignDist(myDTWalign)
```

Here, `AlignDist` simply loops over `get_distance` for all pairs of sequences.

In addition, the above can be used to easily factory-create a number of shorthands, every time a new DTW alignment algorithm is added.