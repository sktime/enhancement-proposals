# Time Series Annotation
Contributors: @fkiraly, @mloning 

## Introduction
### What's time series annotation?
Time series annotation describes the learning task in which we want to annotate the observations of a given time series. 

More formally, for given observations $\textbf{x} = (x(t_1)\dots x(t_T))$ of a single time series at $T$  time points $t_1 ... t_T$, the task is to learn an annotator that accurately predicts a series of annotations $\hat{\textbf{y}} = (\hat{y}(a_1)\dots \hat{y}(a_A))$ for the observed series $\textbf{x}$, where $a_1\dots a_A$ denotes the time indices of the annotations. The task varies by value domain and interpretation of the annotations $\hat{\textbf{y}}$ in relation to $\textbf{x}$. For example,
* in change-point detection, $\hat{\textbf{y}}$ contains change points and the type of change point; 
* in anomaly detection, the $a_j$ are a sub-set of the $t_j$ and indicate anomalies, possibly with the anomaly type; 
* in segmentation, the $a_j$ are interpreted to subdivide the series $\textbf{x}$ into segments, annotated by the type of segment. 

Time series annotation is also found in supervised form, with partial annotations within a single time series, or multiple annotated i.i.d. panel data training instances.

### What's the goal of sktime? 
With sktime, our goal is to design a uniform interface across time series related learning tasks (e.g. time series classification/regression, forecasting or time series annotion), so that users can easily build composite models and apply reduction techniques. For more details, see [our paper](http://learningsys.org/neurips19/assets/papers/sktime_ml_systems_neurips2019.pdf).

This requires that frameworks for specific learning tasks, like time seris annotation, have a uniform interface which is inter-operable with other frameworks. All annotation algorithms (or annotators) need to have methods for at least the following functionality: 

1. Model specification (constructor, hyper-parameters)
2. Training/estimation
3. Inspection (hyper-parameters, but ideally also fitted parameters)
4. Application (prediction/annotation)
5. Updating (online learning)
6. Persistence (save, load)


## Table of contents 
**I.** [**Example use cases**](#I.-Example-use-cases)

**II.** [**Key concepts and questions**](#II.-Key-concepts-and-questions)

**III.** [**Related software**](#III.-Related-software)

**IV.** [**Proposal 1: Single series annotation**](#IV.-Proposal-1:-Single-series-annotation)

**V.** [**Future development**](#V.-Future-development)



## I. Example use cases
Before looking at concrete proposals, we describe a few use cases we want to cover with sktime and discuss related key concepts and questions below. 

* outlier/anomaly detection
* time series segmentation (e.g. piecewise linear approximation based on segments which were found through minimising some approximation error, i.e. a transformation which also produces a segmentation)

## II. Key concepts and questions
To start, we give an overview of time series annotation. 

 
### Taxonomy of time series annotation tasks
There are multiple time series annotation related learning tasks. They can be distinguished as follows: 

* by the kind of annotation, we distinguish between two kinds of annotations: points and segments,
* by whether the point or segment is labelled,
* by the interpretation of the anotation (e.g. outlier, change point, etc.).

The table below gives an overview.

| Name | Kind | Labelled? | Interpretation | Return type |
|---|---|---|---|---|
| change points | point | no | "here something changes" | collection of time stamps |
| anomalies/outliers | point | no | "here is an anomaly" | collection of time stamps |
| unlabelled points of interest | point | no |"this is an event of interest" | collection of time stamps |
labelled points of interest | point | yes |"this is an event of type X" | collection of labelled time stamps |
change segments | segment | no |"something changes over the duration of this segment" | collection of segments |
anomalous segments | segment | no | "this is an anomalous segment" | collection of segments with the "anomalous" label |
unlabelled segments | segment | no |"this is a segment of interest" | collection of segments |
 labelled segments | segment | yes | "this is a segment with label X" | labelled segment |


In addition, other key distinctions include:
* Unsupervised vs semi-supervised vs supervised - are there labels available during training?
* i.i.d. replicates of time series vs only a single time series
* Type of label, e.g. category or number
* Deterministic vs probabilistic annotation, multiple variants can exist of probabilistic variants, e.g. distribution over segments or only over end points (marginals),
* Co-occurrence of multiple different annotation types can complicate the problem (multi-label problems)
* Are segments mutually exclusive or can they overlap (fuzzy segments)? Is the covering by segments jointly exhaustive?
* Are there multiple kinds of events, if training data is unlabelled?
* If training data is labelled, have we seen all labels in the training set? 
* Which parts of the data can we use for training? Entire samples only, or "past" data plus other samples (online vs offline learning)?

### Return type
* `pd.Series` for points
* `pd.DataFrame` for segments
* time index of annotations may be different from time index of time series, but should generally be a subset
* some variants thereof for probabilistic annoations

## III. Related software
  
 | Name | Description | Time series? | Labelled? | Single series? | Online? | 
 |---|---|---|---|---|---|
 | [ruptures](https://github.com/deepcharles/ruptures) | change point detection | yes | yes | yes | no |
 | [seqlearn](http://larsmans.github.io/seqlearn/index.html) | labelled points (Hidden Markov and structured Perceptron) | yes | yes | panel data | no |
 | [PyOD](https://github.com/yzhao062/pyod/issues) | outlier detection | tabular | - | - | - |
 | [gluon-ts](https://gluon-ts.mxnet.io) | forecasting and anomaly detection | yes | no | both | yes |
| [adtk](https://github.com/arundo/adtk/tree/master) | anomaly detection | yes | yes | yes | yes |

For other related software, also see [our wiki entry](https://github.com/alan-turing-institute/sktime/wiki/related-software).

:::info
* adtk is an open-source toolbox from Arundo, a company that focuses on time series analysis and signal processing, they have implemented rule based algorithms and composite functionality, we should get in touch with them to see if they have any ideas for a joint projects and re-use their code whenever possible
:::


## IV. Proposal 1: Single series annotation

### Scope
* start with simplest setting: the single series offline annotation


### Core API

```python
class SingleSeriesUnsupervisedAnnotator:
    
    def fit(self, x):
        # x is single time series
        pass
    
    def predict(self, x):
        pass
        
class SingleSeriesSupervisedAnnotator:

    def fit(self, x, y):
        # x is single time series, y is series of labels
        pass
        
    def predict(self, x):
        pass
```

### Algorithms
:::info
Which algorithms should we include?
:::

#### Atomic
* naive rule-based annotators (e.g. thresholding)

#### Composite
* for supervised annotation, reduction to time series classification with bidirectional sliding windows
* for unsupervised annotation, reduction to forecasting via thresholding residuals


## V. Future development

### Panel data annotators
```python
class UnsupervisedAnnotator:

    def fit(self, X):
        # X is dataframe of multiple series
        pass
        
    def predict(self, X):
        pass
        
class SupervisedAnnotator:
    
    def fit(self, X, Y):
        # X is dataframe of multiple series, Y dataframe of labels for each series in X
        pass
        
    def predict(self, X):
        pass
```

### Online learning and forecasting
* For online learning, elements from sktime's forecasting API can be used for keeping track of time and dynamically updating annotations
