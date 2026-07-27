# Recursive composite and native serialization

Contributors: geetu040, patelchaitany, fkiraly

## Introduction

STEP 27 proposes one serialization design for `sktime` objects which:

- preserves the existing `save` and `load` user API;
- serializes composite estimators recursively;
- stores deep-learning and foundation-model state using framework-native formats;
- uses the same recursive container contract in `sktime`, `skpro`, and other
packages built on the common base-object interface.

The central design rule is that every serialized `BaseObject` is a
self-contained **serialization node**. A node contains ordinary Python state,
may contain framework-native artifacts, and may contain child serialization
nodes. The same node format is applied recursively, so a child may itself own
native artifacts and further children.

STEP 27 supersedes
[STEP 20](../20_saveload/step.md). STEP 20 established the public `save`/`load`
workflow and the `_metadata`/`_obj` archive, but it did not specify a common
recursive format or a general native-artifact mechanism.

Preliminary discussions and implementations are:

- `skpro` [issue #1072](https://github.com/sktime/skpro/issues/1072) and
`skpro` [PR #1073](https://github.com/sktime/skpro/pull/1073), on recursive
serialization;
- `sktime` [issue #10450](https://github.com/sktime/sktime/issues/10450) and
`sktime` [PR #10453](https://github.com/sktime/sktime/pull/10453), on native
serialization;
- `sktime` [issue #10582](https://github.com/sktime/sktime/issues/10582), on
making the two designs homogeneous.



## Contents

1. [Problem statement](#problem-statement)
2. [Goals and non-goals](#goals-and-non-goals)
3. [Description of proposed solution](#description-of-proposed-solution)
4. [Serialization-node format](#serialization-node-format)
5. [Save and load semantics](#save-and-load-semantics)
6. [Developer interface](#developer-interface)
7. [Cross-package serialization](#cross-package-serialization)
8. [Backward compatibility](#backward-compatibility)
9. [Motivation](#motivation)
10. [Alternatives considered](#discussion-and-comparison-of-alternative-solutions)
11. [Implementation plan](#detailed-description-of-design-and-implementation-of-proposed-solution)
12. [Testing and security](#testing-and-security)



## Problem statement

The current `sktime` serialization format has a useful universal API:

```python
saved = estimator.save()
restored = sktime.base.load(saved)
```

or:

```python
estimator.save("estimator")
restored = sktime.base.load("estimator")
```

For an ordinary estimator, the file form is a ZIP archive containing a pickled
class in `_metadata` and a pickled estimator in `_obj`. This works when the
complete estimator graph is safely handled by pickle.

Two classes of objects require more structure.

First, a composite estimator contains other estimators, possibly nested inside
lists, tuples, dictionaries, or fitted attributes. Treating the entire graph as
one opaque pickle:

- prevents a child from using its own serialization policy;
- prevents recursive composition across `sktime` and `skpro`;
- makes the archive hard to inspect;
- does not allow a native model owned by a nested child to be externalized.

Second, many deep-learning and foundation-model objects should not be pickled.
Their libraries provide formats designed to store weights, configuration, and
framework state, such as:

- `save_pretrained`/`from_pretrained`;
- Keras `.keras` files;
- Lightning checkpoints;
- PyTorch state dictionaries.

Some large objects are reconstructable caches and should not be persisted at
all. Existing estimator-specific `save`, `__getstate__`, and `__setstate__`
implementations solve parts of this problem, but do not provide one composable
contract.

The required design must therefore answer both questions at once:

1. how is an estimator graph decomposed into recursively loadable components?
2. how does any node in that graph store attributes using a native backend?



## Goals and non-goals



### Goals

The proposed format and implementation must:

- keep the public `BaseObject.save` and `sktime.base.load` workflow unchanged;
- preserve fitted state and predictive behaviour after a round trip;
- make an ordinary non-composite archive a valid minimal serialization node;
- serialize child `BaseObject` instances recursively;
- apply native serialization independently at every node;
- support the same logical bundle in memory and on disk;
- be readable and debuggable after unpacking the ZIP;
- be usable by `sktime`, `skpro`, and compatible packages without
package-specific cases in the top-level `load` function;
- retain compatibility with existing plain `_metadata`/`_obj` archives;
- fail clearly when a selected object cannot be serialized.



### Non-goals

STEP 27 does not promise:

- loading archives from untrusted sources safely; `_obj` and `_metadata` use
pickle-compatible serialization;
- loading under arbitrary Python, package, or framework versions;
- a language-independent representation of estimator state;
- automatic native serialization of arbitrary objects not selected by an
estimator;
- support for cyclic estimator ownership graphs in the first implementation.

Non-estimator third-party objects, including scikit-learn objects, remain part
of the parent's `_obj` unless an owning estimator selects them as native
artifacts. If such an object is not pickleable and has no selected native
backend, saving raises an error.

## Description of proposed solution



### Public API

No new user-facing workflow is introduced:

```python
from sktime.base import load

model.fit(y)

# in memory
serial = model.save()
restored = load(serial)

# on disk
model.save("model")
restored = load("model")
```

`serialization_format="pickle"` and `"cloudpickle"` continue to control
serialization of `_metadata` and `_obj`. Native backends always use their
framework-specific formats.

### The serialization-node abstraction

Every `BaseObject` is serialized as a node with this contract:

```text
node/
├── _metadata
├── _obj
├── _artifacts/        # optional
│   ├── index.json
│   └── ...
└── _components/       # optional
    ├── index.json
    └── ...
```

The archive root is the root object's node. There is no additional `root/`
directory.

The two optional directories are orthogonal:

- `_artifacts` contains selected attributes of this node, written with native
framework serializers;
- `_components` contains child `BaseObject` nodes, each of which follows the
same contract.

Consequently, native serialization is not a special case at the archive level.
It is one possible part of any node. A composite with a deeply nested Keras or
Transformers estimator works because the nested estimator's node owns its
native model files.

### State-classification precedence

Before a node's `_obj` is serialized, its state is classified in this order:

1. attributes selected by `serialization:skip` are omitted;
2. attributes selected by `serialization:native_artifacts` are removed from
  `_obj` and stored under `_artifacts`;
3. remaining child `BaseObject` instances are replaced by component references
  and stored under `_components`;
4. all remaining state is serialized into `_obj`.

An attribute must not be listed in both serialization tags. Conflicting tags
raise a validation error instead of relying on implicit precedence.

Native-artifact selection applies to the complete selected attribute. The
serializer does not recursively search inside that attribute for components.
This makes estimator declarations authoritative and backend dispatch
predictable.

## Serialization-node format



### `_metadata`

`_metadata` identifies the class responsible for loading the node.

Readers must accept the existing representation, which is a pickled class
object. A versioned writer may store a pickled metadata dictionary:

```python
{
    "format_version": 2,
    "class": type(self),
    "serialization_format": "pickle",
}
```

`format_version == 2` denotes the recursive-node format in STEP 27. Keeping
the class object, rather than only a qualified class string, preserves the
current top-level dispatch model and cloudpickle support for non-importable
classes.

The metadata in a child node has the same meaning as metadata at the archive
root. The child's `_metadata`, not a parent-side duplicate, is authoritative.

### `_obj`

`_obj` is a pickle-compatible serialization of the node after skipped,
native-artifact, and child-component state has been externalized.

Component references should use pickle persistent IDs or an equivalent
internal reference mechanism. Conceptually, a reference is:

```python
("sktime-component", "component-0000")
```

Using object references in `_obj`, rather than encoded attribute paths such as
`estimators_[0][1]`, has important properties:

- children can occur inside arbitrary supported Python containers;
- reconstruction does not need an attribute-path parser;
- refactoring a container layout does not change the archive protocol;
- multiple references from one parent to the same child can reuse one
component ID and preserve identity.

The root object itself is never converted to a component reference while its
node is being written.

### `_artifacts`

The native-artifact layout from `sktime` PR #10453 is retained:

```text
_artifacts/
├── index.json
├── model_/
│   ├── config.json
│   └── model.safetensors
└── network_/
    └── state_dict.pt
```

`_artifacts/index.json` maps an estimator attribute to the backend, class, and
relative path:

```json
{
  "model_": {
    "backend": "pretrained",
    "class": "transformers.models.bert.modeling_bert.BertModel",
    "path": "model_"
  },
  "network_": {
    "backend": "torch_state_dict",
    "class": "package.networks.Network",
    "path": "network_"
  }
}
```

The initially supported backend contracts are:


| Backend                | Save form               | Load form                                                        |
| ---------------------- | ----------------------- | ---------------------------------------------------------------- |
| `pretrained`           | `save_pretrained(path)` | `class.from_pretrained(path, **kwargs)`                          |
| `keras`                | `model.keras`           | `keras.models.load_model`                                        |
| `lightning_checkpoint` | `model.ckpt`            | `class.load_from_checkpoint`                                     |
| `torch_state_dict`     | CPU `state_dict.pt`     | estimator constructs the module, then loads the state dictionary |


Backend selection is based on the selected object's type or supported native
protocol. The backend field in the index controls loading. Framework-specific
load variations are supplied by small estimator hooks; they are not added to
the generic `load` function.

An attribute listed as a native artifact is absent from `_obj`. If its value is
`None` or the attribute is absent, no artifact entry is written. If no entries
exist, `_artifacts` is omitted.

### `_components`

`_components` stores immediate child nodes:

```text
_components/
├── index.json
├── component-0000/
│   ├── _metadata
│   ├── _obj
│   └── _artifacts/
│       ├── index.json
│       └── model_/
│           └── model.keras
└── component-0001/
    ├── _metadata
    ├── _obj
    └── _components/
        ├── index.json
        └── component-0000/
            ├── _metadata
            └── _obj
```

`_components/index.json` maps local component IDs to directories:

```json
{
  "component-0000": {
    "path": "component-0000"
  },
  "component-0001": {
    "path": "component-0001"
  }
}
```

Component IDs are opaque and local to the parent node. Directory names do not
encode Python attribute paths. The corresponding persistent references in
`_obj` describe where loaded objects are inserted naturally through unpickling.

No explicit global topological load order is stored. Recursive loading already
defines the order: a referenced child is fully loaded before it is returned to
the parent unpickler. This removes duplicated graph information from the
format.

Within one parent node, repeated references to the same child identity use the
same component ID. The first implementation may reject ownership cycles and
cross-branch estimator aliases with a clear error. It must not recurse
indefinitely or silently produce an ambiguous graph. These cases can be added
later by extending references with archive-global object IDs.

### Complete example

A composite whose root owns a pretrained model, and whose child owns a Keras
model, is represented as:

```text
composite.zip
├── _metadata
├── _obj
├── _artifacts/
│   ├── index.json
│   └── model_/
│       ├── config.json
│       └── model.safetensors
└── _components/
    ├── index.json
    └── component-0000/
        ├── _metadata
        ├── _obj
        └── _artifacts/
            ├── index.json
            └── network_/
                └── model.keras
```

There is deliberately no archive-global `manifest.json`. `_metadata` and the
two node-local indexes are the manifest for that node. Applying the same small
contract recursively keeps ownership explicit and allows a developer to
understand a subtree by inspecting its directory alone.

## Save and load semantics



### Saving a node

The node writer performs:

1. validate serialization tags and capture the object's serializable state;
2. omit attributes selected by `serialization:skip`;
3. externalize selected native artifacts;
4. assign local IDs to immediate child `BaseObject` identities;
5. serialize `_obj` with component references in place of those children;
6. write `_metadata`;
7. write `_artifacts` and its index, if any;
8. recursively write each child under `_components` and write its index;
9. finalize the root directory as a ZIP or in-memory container.

Saving must not mutate the source object after the method returns, including
when serialization raises. An implementation may temporarily alter state, but
restoration must occur in a `finally` block. Prefer preparing detached state or
using a custom pickler over mutating `self.__dict__`.

### Loading a node

The node reader performs:

1. read `_metadata` and identify the node class and format version;
2. start creating the ordinary object from `_obj`;
3. when `_obj` requests a component ID, locate it through
  `_components/index.json`;
4. recursively load the child using the child's own `_metadata` and node
  loader;
5. return the loaded child to the parent unpickler and finish constructing the
  parent object;
6. restore the parent's native artifacts from `_artifacts/index.json`;
7. return the fully reconstructed object.

Component loading is therefore depth first. By the time a parent's native load
hooks run, the parent's ordinary serialized state and child objects are
available. The returned root has all native attributes and children restored.

The loose top-level `load` function remains limited to input validation,
opening the outer container, reading `_metadata`, and delegating to the
identified class. It must not contain estimator-, framework-, or
package-specific logic.

### In-memory form

The current two-element return contract is retained:

```python
cls, payload = estimator.save()
```

For a node with neither native artifacts nor child nodes, `payload` may remain
the lightweight pickle/cloudpickle byte stream.

If a node has native artifacts or child nodes, `payload` is an in-memory ZIP
containing the same logical structure as the disk archive. Callers must treat
the payload as opaque and pass the complete tuple to `load`.

Path-oriented native frameworks may use a temporary directory while producing
an in-memory ZIP.

## Developer interface



### Serialization mixin

The common implementation belongs in a private serialization mixin used by
`BaseObject`, as proposed in `sktime` PR #10453:

```python
class BaseObject(
    _SerializationMixin,
    _HTMLDocumentationLinkMixin,
    _BaseObject,
):
    ...
```

The mixin owns:

- `save`;
- `load_from_serial`;
- `load_from_path`;
- node writing and reading;
- native backend dispatch;
- component reference handling.

This avoids adding framework-specific branches to `BaseObject` or the loose
`load` function.

### Tags

The native design uses two tags:

```python
_tags = {
    "serialization:native_artifacts": ("model_",),
    "serialization:skip": ("trainer_",),
}
```

- `serialization:native_artifacts` lists attributes which must be saved using
a native backend.
- `serialization:skip` lists reconstructable caches, runtime wrappers,
sessions, or other attributes intentionally absent from persistence.

The tags may be dynamic when the policy depends on fitted state. For example, a
zero-shot model cache may be skipped if it can be recreated from a model
identifier, while fine-tuned weights must be stored natively.

No tag is required to enumerate components. After tag-selected attributes have
been handled, child `BaseObject` instances encountered by the component-aware
pickler are serialized recursively.

### Native load hooks

Small hooks handle cases where a native format is not self-constructing:

- `_create_torch_artifact(name)` constructs a compatible module before loading
a state dictionary;
- `_get_native_artifact_load_kwargs(name)` supplies arguments needed by
`from_pretrained`;
- `get_custom_objects()` supplies custom Keras objects.

Additional hooks or backends should be added at the native-backend layer, not
to the generic recursive traversal.

## Cross-package serialization

The node format is package-neutral. A composite may contain, for example, an
`sktime` forecaster with an `skpro` probabilistic-regression component.

Cross-package loading works as follows:

1. the parent loader encounters a component reference;
2. it reads the child node's `_metadata`;
3. the metadata resolves the child's actual class;
4. the common node reader loads the child, irrespective of which package
  defines that class;
5. the child is returned to the parent's unpickler.

This requires participating packages to implement the same node contract and
metadata interpretation. The implementation should be shared in `skbase` once
the protocol is stable. During migration, `sktime` and `skpro` may carry
compatible implementations with cross-package conformance tests.

A package must not assume that all child classes derive from its own local
`BaseObject` symbol. Component recognition should use the common `skbase`
base-object type or a narrowly defined shared serialization protocol.

## Backward compatibility

STEP 27 deliberately makes the old archive a subset of the new design:

```text
legacy-or-minimal-node/
├── _metadata
└── _obj
```

A new reader must support:

- existing archives whose `_metadata` is a pickled class object;
- existing in-memory `(class, pickle_bytes)` tuples;
- minimal archives without `_artifacts` or `_components`;
- native-artifact archives produced by the design in PR #10453;
- version-2 recursive nodes.

The public signatures and path conventions remain unchanged. Existing
estimators with no serialization tags and no child nodes retain the lightweight
path.

Old readers are not expected to load new recursive archives. Versioned metadata
allows new readers to reject unsupported future formats with a useful error
instead of failing later during unpickling.

Estimator-specific serialization overrides should be migrated to the mixin,
tags, and backend hooks. Temporary overrides remain possible, but they must
produce or consume the common node format if they participate in recursive
serialization.

## Motivation

The design composes two independently useful mechanisms.

Native serialization solves the persistence needs of a single estimator node.
Recursive serialization solves estimator ownership. Keeping the mechanisms
orthogonal means their combination follows automatically:

- an ordinary leaf has only `_metadata` and `_obj`;
- a native-model leaf adds `_artifacts`;
- a classical composite adds `_components`;
- a composite with native state may have both;
- a child repeats exactly the same possibilities.

The explicit directory tree follows object ownership and is straightforward to
inspect after unpacking an archive. Node-local indexes contain only local
relationships, so no global manifest and no separately maintained topological
order can become inconsistent with the directory structure.

The design also preserves the successful parts of the current API and PR
#10453: tagged opt-in, native backend dispatch, unchanged user workflow, and
the `_artifacts/index.json` contract.

## Discussion and comparison of alternative solutions



### Pickle the complete estimator graph

This is the current default and remains the fast path for minimal objects. It
is insufficient as the universal format because nested estimators cannot apply
their own serialization policy, and native model objects may be inefficient,
fragile, or impossible to pickle.

### One flat archive and one global `manifest.json`

The initial recursive implementation in `skpro` PR #1073 places the root in a
special `root/` directory, places all other components under a flat
`components/` directory, and records parents, attribute paths, children, and a
topological load order in a global manifest.

This has the advantage that the complete graph can be inspected in one JSON
file. It also permits a loader to process leaves in an explicit order.

It is not selected because:

- the archive root already represents the root object;
- a flat directory hides nesting that then has to be reconstructed in the
manifest;
- attribute-path strings require a second object-navigation language;
- parent, child, path, and load-order fields duplicate relationships and can
disagree;
- a component is not represented by the same self-contained format as the
root;
- it diverges unnecessarily from existing `sktime` archives and the native
layout in PR #10453.



### Recursive directories plus a global manifest

A global manifest could index an otherwise recursive tree. It makes a complete
inventory available without walking directories, but duplicates local metadata
and introduces two sources of truth.

STEP 27 instead uses `_metadata`, `_artifacts/index.json`, and
`_components/index.json` at each node. Tools that need a global view can derive
one by walking the explicit tree.

### Encoded attribute paths

An index may record locators such as `estimators[0][1]` and use them to reattach
children. This is readable for simple cases but requires escaping, parsing, and
mutation rules for every supported container type. It is also brittle for
immutable containers.

Persistent component references let the serializer itself preserve placement
inside lists, tuples, dictionaries, and object state, so they are preferred.

### Estimator-specific `save` and `load` overrides

Overrides are flexible but do not compose automatically. A parent must know
that a child has a special implementation, and multiple framework-specific
formats proliferate.

Tags, backend strategies, and narrow reconstruction hooks keep estimator code
declarative while retaining an extension point for genuine framework
differences.

### Automatic native-object discovery

The serializer could inspect every object and apply native serialization
whenever it recognizes a Keras, Torch, or Transformers type.

This is rejected because ownership and reconstruction requirements are
estimator-specific. Explicit tags document which attributes are authoritative
fitted state, which are reconstructable caches, and which should remain in
ordinary state.

## Detailed description of design and implementation of proposed solution

The implementation can proceed in independently testable stages.

### Stage 1: establish native serialization at one node

Complete the design from `sktime` PR #10453:

- move serialization into `_SerializationMixin`;
- add and validate the two serialization tags;
- implement `_NativeArtifactStore`;
- implement native backend strategies;
- support native bundles for both memory and disk;
- migrate estimator-specific deep-learning serialization.

This stage already produces the node's `_metadata`, `_obj`, and optional
`_artifacts`.

### Stage 2: add component-aware node writing

Add a component-aware pickler or equivalent state transformer which:

- recognizes child objects through the shared base-object protocol;
- excludes the current node itself;
- assigns deterministic local component IDs;
- returns persistent component references;
- records each child once per parent identity;
- recursively writes child nodes;
- writes `_components/index.json`;
- detects unsupported ownership cycles.

The native and skip classification must occur before component discovery.

### Stage 3: add recursive node loading

Add the matching component-aware unpickler which:

- validates persistent IDs;
- resolves IDs only through the current node's component index;
- validates that resolved paths remain inside the node;
- reads the child `_metadata`;
- delegates to the common node reader;
- caches loaded component IDs to preserve repeated references.

The existing loose `load` entry point should require only metadata-version
compatibility changes.

### Stage 4: adopt the protocol in `skpro`

Replace the experimental flat recursive format in `skpro` PR #1073 with the
same node writer and reader. Add mixed composites to the test matrix. Once both
packages agree on the protocol and error semantics, move the package-neutral
parts into `skbase`.

### Stage 5: documentation and migration

Document:

- the public memory and file workflows;
- the complete node layout and JSON schemas;
- serialization tags and native hooks;
- compatibility and environment requirements;
- the pickle trust warning.

Migrate custom serializers incrementally and retain compatibility tests for
archives produced before STEP 27.

## Testing and security



### Required tests

Tests should cover:

- ordinary object round trips in memory and on disk;
- pickle and cloudpickle;
- skipped attributes without mutation of the source;
- each native backend;
- native artifacts at the root, a child, and a grandchild;
- composites using direct attributes, lists, tuples, and dictionaries;
- repeated references to one child from the same parent;
- mixed `sktime`/`skpro` composites;
- fitted and unfitted composites;
- missing, `None`, and unsupported native artifacts;
- unsupported estimator cycles;
- malformed metadata, component indexes, artifact indexes, and persistent IDs;
- legacy `_metadata`/`_obj` and PR-#10453-style native archives;
- behavioural equivalence before and after serialization, not only equality of
stored attributes.

The source estimator must remain usable and unchanged after successful and
failed save attempts.

### Security

Serialization archives contain pickle-compatible data. Loading may execute
arbitrary code and must be documented as safe only for trusted archives.

Archive readers must additionally:

- reject absolute paths and `..` traversal in indexes and ZIP members;
- resolve all artifact and component paths beneath their declared node;
- reject duplicate or unknown component IDs;
- avoid following symlinks outside the extraction directory;
- provide clear errors for unsupported format versions and backends;
- clean up temporary files after success or failure.

Resource limits and cryptographic signing are outside the first implementation,
but the format does not prevent them from being added later.