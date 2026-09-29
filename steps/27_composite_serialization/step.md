# Recursive composite and native serialization

Contributors: [geetu040](https://github.com/geetu040), [patelchaitany](https://github.com/patelchaitany), [fkiraly](https://github.com/fkiraly)

## Introduction

This step lays out a serialization design for `sktime` objects that keeps the existing `save`/`load` API exactly as users know it, but changes what happens underneath: composite estimators get serialized recursively, deep-learning and foundation-model state gets written out in whatever native format the framework actually uses, and the same recursive container contract is shared across `sktime`, `skpro`, and anything else built on the common base-object interface.

The core idea is simple to state: every serialized `BaseObject` is a self-contained **serialization node**. A node holds ordinary Python state, and it may also hold framework-native artifacts and child serialization nodes. Because the same node format applies at every level, a child node can itself contain native artifacts and further children of its own — the format doesn't need a separate rule for "one level down."

This step supersedes the earlier `save`/`load` step. That step established the public workflow and the `_metadata`/`_obj` archive, but it never specified a general recursive format or a mechanism for native artifacts, which is the gap this step fills.

Preliminary discussions and implementations:

- `skpro` issue [#1072](https://github.com/sktime/skpro/issues/1072) and PR [#1073](https://github.com/sktime/skpro/pull/1073), recursive serialization
- `sktime` design document [hackmd](https://hackmd.io/rShrR00UQrKcNo33P6dJBA), issue [#10450](https://github.com/sktime/sktime/issues/10450) and PR [#10453](https://github.com/sktime/sktime/pull/10453), native serialization
- `sktime` issue [#10582](https://github.com/sktime/sktime/issues/10582), unifying the two designs

## Problem statement

Right now, `estimator.save()` / `sktime.base.load()` just produce a ZIP with a pickled class in `_metadata` and a pickled estimator in `_obj`. That's fine as long as the whole estimator graph can be safely pickled, but two situations break it.

The first is **composite estimators**. These contain other estimators, sometimes nested inside lists, tuples, dicts, or fitted attributes. Treating the entire graph as one opaque pickle means a child can't apply its own serialization policy, recursive composition across `sktime`/`skpro` is blocked, the resulting archive is hard to inspect, and a nested child's native model has no way to be pulled out and saved separately.

The second is **deep-learning and foundation-model objects**. These really shouldn't be pickled at all — they already have native formats of their own (`save_pretrained`/`from_pretrained`, Keras's `.keras`, Lightning checkpoints, PyTorch state dicts), and some of their larger attributes are just reconstructable caches that don't need to be persisted in the first place.

So the design has two questions to answer: how does an estimator graph decompose into pieces that load back recursively, and how does any given node store its attributes through a native backend when it needs to?

## Goals and non-goals

What we want to achieve: the public `save`/`load` workflow stays unchanged; fitted state and predictive behavior survive a round trip; an ordinary non-composite archive is still a valid, minimal node; children are serialized recursively; native serialization can be applied independently per node; the same bundle works whether it's in memory or on disk; archives stay inspectable once unzipped; the whole thing works across `sktime`, `skpro`, and compatible packages without `load` needing package-specific branches; existing `_metadata`/`_obj` archives keep working; and anything genuinely unserializable fails with a clear error rather than silently.

What we're explicitly not trying to do here: make loading safe for untrusted archives (it's still pickle-based, and that doesn't change); guarantee cross-version compatibility; define a language-independent state representation; automatically apply native serialization to attributes nobody selected; or support cyclic ownership graphs in this first version.

Third-party, non-estimator objects — including plain scikit-learn objects — stay inside the parent's `_obj` unless someone explicitly marks them as native artifacts. If such an object can't be pickled and has no native backend, saving simply raises an error.

## Proposed solution

There's no new user-facing workflow here. `model.save()` / `load(serial)` still works in memory, and `model.save("model")` / `load("model")` still works on disk, exactly as before. The `serialization_format="pickle"`/`"cloudpickle"` option still controls what happens to `_metadata`/`_obj`; native backends always use their own formats regardless of that setting.

### The serialization-node abstraction

Every `BaseObject` gets written out like this:

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

The archive root is simply the root object's own node — there's no extra `root/` wrapper directory around it. `_artifacts` (attributes selected for native serialization) and `_components` (child `BaseObject` nodes, following the exact same contract recursively) are orthogonal to each other. That means native serialization never becomes a special case at the archive level; it's just one optional part that any node might have.

### State classification order

When a node is written, its attributes get sorted in this order:

1. anything tagged `serialization:skip` gets dropped entirely
2. anything tagged `serialization:native_artifacts` gets pulled out into `_artifacts`
3. whatever `BaseObject` children remain become component references in `_components`
4. everything left over goes into `_obj`

An attribute can't carry both tags at once — that raises a validation error. Also worth noting: native-artifact selection applies to the whole attribute that got selected. The serializer doesn't then go digging around inside it looking for further components.

## Serialization-node format

**`_metadata`** is what identifies the loader class. Existing archives store a pickled class object directly; a versioned writer may instead store something like `{"format_version": 2, "class": type(self), "serialization_format": "pickle"}`. We keep the actual class object rather than just a qualified name string, because that's what preserves cloudpickle support for classes that aren't otherwise importable. A child always has authority over its own `_metadata` — a parent never duplicates or overrides it.

**`_obj`** holds whatever pickle-compatible state is left after the skip/native/component extraction has happened. Components inside it are referenced by persistent IDs — things like `("sktime-component", "component-0000")` — rather than by attribute paths such as `estimators_[0][1]`. Using IDs instead of paths means children can live inside arbitrary container structures, there's no path parser to build, the references survive container refactors down the line, and if the same child is referenced twice, both references can point at one shared component ID.

**`_artifacts`** keeps the same layout introduced in PR [#10453](https://github.com/sktime/sktime/pull/10453):

```text
_artifacts/
├── index.json
├── model_/
│   ├── config.json
│   └── model.safetensors
└── network_/
    └── state_dict.pt
```

`index.json` just maps each attribute name to its backend, class, and path, for example:

```json
{
  "model_": {"backend": "pretrained", "class": "transformers.models.bert.modeling_bert.BertModel", "path": "model_"},
  "network_": {"backend": "torch_state_dict", "class": "package.networks.Network", "path": "network_"}
}
```

The backends supported so far:

| Backend                | Save form               | Load form                                              |
| ---------------------- | ----------------------- | ------------------------------------------------------ |
| `pretrained`           | `save_pretrained(path)` | `class.from_pretrained(path, **kwargs)`                |
| `keras`                | `model.keras`           | `keras.models.load_model`                              |
| `lightning_checkpoint` | `model.ckpt`            | `class.load_from_checkpoint`                           |
| `torch_state_dict`     | CPU `state_dict.pt`     | estimator builds the module, then loads the state dict |

If an artifact attribute is `None` or missing, nothing gets written for it, and if `_artifacts` ends up empty overall it's left out of the archive entirely. Anything backend-specific and quirky about loading stays in small estimator-level hooks rather than leaking into the generic loader.

**`_components`** is where immediate children live:

```text
_components/
├── index.json
├── component-0000/
│   ├── _metadata
│   ├── _obj
│   └── _artifacts/...
└── component-0001/
    ├── _metadata
    ├── _obj
    └── _components/...
```

`index.json` here just maps opaque local IDs to their directories — again, no attribute-path encoding involved. There's no need for a separate global load order either: because loading is recursive (a child is always fully loaded before it's handed back to the parent's unpickler), the order falls out naturally from the recursion itself. If the same child gets referenced more than once inside one parent, both references reuse the same component ID. Ownership cycles and cross-branch aliases aren't supported in this first version — they raise a clear error rather than looping forever or producing an ambiguous graph.

### Complete example

Here's what a composite looks like with a pretrained model at the root and a Keras model tucked inside a child:

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

Notice there's deliberately no top-level `manifest.json` anywhere. `_metadata` plus the two node-local index files already act as the manifest for that node, so anyone can understand a subtree just by looking inside its own directory — no need to cross-reference something at the root.

### Multiple serialization formats

Currently `serialization_format` supports only `"pickle"` and `"cloudpickle"`, but this is independent of native or composite serialization. The format only decides how `_metadata` and `_obj` are written at a single node; recursion, component references, and native backends never look at it, and `_artifacts` is unaffected since native backends always use their framework formats.

So more formats can be added later without touching any of that. A format is just a name mapped to a dumps/loads pair, and each node uses one format for both its `_metadata` and its `_obj`, recording the name in `_metadata` so a reader knows what wrote the node. Children inherit the parent's format by default. The one requirement is that `_metadata` stays readable before the format is known, which holds as long as the name is stored as a plain string and `_metadata` itself is written in a form plain `pickle` can read.

## Save and load semantics

**Saving** goes: validate tags, capture state, strip out skipped attributes, externalize native artifacts, assign local IDs to child identities, serialize `_obj` with component references in place, write `_metadata`, write `_artifacts` if there are any, recursively write each child under `_components`, and finally package it all up as a ZIP or an in-memory container. Saving must never mutate the source object — even if something fails partway through, any temporary state changes get restored in a `finally` block.

**Loading** goes the other direction: read `_metadata`, start reconstructing from `_obj`, and whenever a component reference is hit, resolve it through that node's `_components/index.json`, recursively load the child (depth-first), and hand the fully-loaded child back to the parent's unpickler. Once `_obj` and all children are in place, native artifacts get restored, and the fully reconstructed object is returned. The top-level `load` function itself stays deliberately thin — it just opens the container, reads `_metadata`, and delegates everything else. No estimator- or framework-specific logic lives there.

**In memory,** nothing changes from the caller's perspective: `cls, payload = estimator.save()` works exactly as before. If a node has no artifacts and no children, `payload` stays the lightweight pickle/cloudpickle bytes it's always been. Otherwise, `payload` becomes an in-memory ZIP with the same internal structure as an on-disk archive — callers just treat it as opaque either way.

## Developer interface

All of this lives inside a private `_SerializationMixin` that `BaseObject` uses. It owns `save`, `load_from_serial`, `load_from_path`, the node reading/writing logic, native backend dispatch, and component reference handling — which keeps framework-specific code out of `BaseObject` itself and out of the loose `load` function.

Two tags drive how attributes get classified:

```python
_tags = {
    "serialization:native_artifacts": ("model_",),
    "serialization:skip": ("trainer_",),
}
```

These tags can be dynamic too — for instance, skipping a zero-shot cache only when it's actually reconstructable from an identifier. Components don't need any tagging at all: any `BaseObject` child left over after tag-based extraction gets picked up automatically by the component-aware pickler.

A handful of small native-load hooks cover the cases where a format can't just reconstruct itself: `_create_torch_artifact(name)` builds a module before a state dict gets loaded into it, `_get_native_artifact_load_kwargs(name)` supplies whatever `from_pretrained` needs as arguments, and `get_custom_objects()` supplies custom Keras objects. Any new backend belongs at this layer — not inside the generic traversal logic.

## Cross-package serialization

It's entirely possible for a composite to mix packages — an `sktime` forecaster with an `skpro` probabilistic-regression child, say. Loading handles this by reading each child's own `_metadata` to figure out its real class, then delegating to the shared node reader no matter which package actually defines that class. For this to work, every participating package needs to implement the same contract — ideally shared through `skbase` once things stabilize, with `sktime` and `skpro` running compatible implementations plus conformance tests in the meantime. Component recognition has to rely on the common `skbase` base-object type here, not on any assumption that children derive from one specific package's local `BaseObject`.

## Backward compatibility

The old archive format is just a strict subset of the new one:

```text
legacy-or-minimal-node/
├── _metadata
└── _obj
```

New readers need to support all of the following: existing pickled-class `_metadata`, existing `(class, pickle_bytes)` in-memory tuples, minimal archives with neither `_artifacts` nor `_components`, the native archives introduced in PR [#10453](https://github.com/sktime/sktime/pull/10453), and the new version-2 recursive nodes. Public signatures and paths don't change at all, and estimators with no tags and no children keep using the same lightweight path they always have. Old readers aren't expected to handle new archives — that's fine, since versioned metadata lets new readers reject unsupported future formats up front with a clear error, instead of failing halfway through an unpickle. Estimator-specific overrides are expected to migrate over to the mixin/tags/hooks approach over time, though temporary overrides can stick around as long as they still produce and consume the common format.

## Motivation

Native serialization solves the persistence problem for a single node. Recursive serialization solves the ownership problem. Keeping the two orthogonal means their combination just falls out naturally rather than needing special-casing: a plain leaf has only `_metadata`/`_obj`; a leaf with a native model adds `_artifacts`; a composite adds `_components`; a composite that also has native state has both; and a child can repeat any of these same possibilities on its own. Because indexes are node-local, there's no global manifest that can ever drift out of sync with the actual directory structure.

## Alternatives considered

**Pickling the whole graph** is what happens today, and it's still the fastest path for simple objects. But it doesn't generalize — nested estimators can't apply their own policies, and native models can be fragile or outright impossible to pickle.

**One flat archive plus a global manifest** was the approach taken in `skpro` PR [#1073](https://github.com/sktime/skpro/pull/1073): a special `root/` directory, a flat `components/` directory, and a manifest tracking parents, attribute paths, children, and load order. This was rejected for several reasons: the root is already the root node, so a separate wrapper is redundant; a flat layout hides the actual nesting, which the manifest then has to reconstruct; attribute-path strings require a whole second navigation language; the parent/child/path/order fields in the manifest can end up disagreeing with each other; a component under this scheme isn't self-contained the way the root is; and it diverges from both existing `sktime` archives and the native layout from PR [#10453](https://github.com/sktime/sktime/pull/10453).

**Recursive directories plus a global manifest** would give a full inventory of the archive without needing to walk the directory tree, but it duplicates local metadata as a second source of truth that can drift. This step sticks to node-local `_metadata`, `_artifacts/index.json`, and `_components/index.json` instead — a global view can always be derived by walking the tree if one is actually needed.

**Encoded attribute paths** (things like `estimators[0][1]`) read nicely for simple cases, but need escaping, parsing, and mutation rules that vary per container type, and they're brittle against immutable containers. Persistent component references let the serializer preserve placement without any of that extra machinery.

**Estimator-specific save/load overrides** are flexible, but they don't compose well — a parent has to know in advance that a particular child is special, and framework-specific formats end up proliferating. Tags plus backend strategies keep estimator code declarative while still leaving room for genuine exceptions when needed.

**Automatic native-object discovery** — inspecting every object and applying native serialization based on recognized type — was rejected because ownership and reconstruction needs are estimator-specific. Explicit tags make it clear which attributes are authoritative fitted state versus which are just reconstructable caches.

## Implementation plan

1. **Native serialization at one node.** Finish out the PR [#10453](https://github.com/sktime/sktime/pull/10453) design: move serialization logic into `_SerializationMixin`, add and validate the two tags, implement `_NativeArtifactStore` plus the backend strategies, turn `_get_serializer` into a small format registry and record the chosen format in `_metadata`, support both memory and disk, and migrate existing deep-learning serialization over. This produces `_metadata`/`_obj` plus an optional `_artifacts`.
2. **Component-aware node writing.** Build a pickler/state transformer that recognizes children through the shared base-object protocol, excludes the current node itself, assigns deterministic local IDs, returns persistent references, records each child once per identity, writes children recursively, writes `_components/index.json`, and detects unsupported cycles. Native/skip classification has to happen before component discovery runs.
3. **Recursive node loading.** Build a matching unpickler that validates persistent IDs, resolves them only through the current node's own index, checks that resolved paths stay inside the node, reads each child's `_metadata`, delegates to the common reader, and caches loaded IDs so repeated references are preserved correctly. The loose `load` entry point only needs a metadata-version compatibility check.
4. **Adopt in `skpro`.** Replace the flat recursive format from PR [#1073](https://github.com/sktime/skpro/pull/1073) with the same writer/reader, add mixed composites to the test matrix, and then move the package-neutral parts into `skbase` once both packages are in agreement.
5. **Documentation and migration.** Document the memory and file workflows, the node layout and JSON schemas, the tags and hooks, the compatibility requirements, and the pickle trust warning. Migrate custom serializers incrementally, keeping compatibility tests running against pre-this-step archives throughout.
