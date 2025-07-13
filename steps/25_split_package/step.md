# Modularisation of sktime into Independent Component Packages

Contributors: @yarnabrina

## Introduction

As the sktime codebase continues to expand, the current monolithic structure is
presenting increasing challenges in terms of maintenance, dependency management, and
release coordination. This proposal outlines a plan to modularise sktime into
independent, interoperable component packages, each with its own dependencies and
release process. The intention is to improve maintainability and scalability, while
allowing for more flexible development and release cycles.

## Contents

1. Problem statement
2. Description of proposed solution
3. Motivation
4. Discussion and comparison of alternative solutions
5. Detailed description of design and implementation
    - Package structure
    - Migration strategy
    - Interoperability and testing
    - Release strategy
    - Documentation and user support
6. Potential concerns and areas to monitor

## Problem statement

The current monolithic structure of sktime presents several challenges:

- Rapid growth in codebase size and complexity.
- Increasing difficulty in managing and resolving dependency conflicts, especially with
  soft dependencies.
- Release process bottlenecked by a single release manager and the need to synchronise
  all modules.
- Inconsistent dependency bound management and insufficient testing of all documented
  bounds.
- Limited and non-exhaustive interoperability testing across modules, particularly for
  estimators with soft dependencies.

## Description of proposed solution

This proposal suggests to split sktime into a set of independent, interoperable
packages, each corresponding to a major module (e.g., forecasting, classification,
transformation). A lightweight `sktime-core` package will provide shared base classes
and utilities, ensuring interoperability and a unified interface. Pipeline and
composition logic will initially reside in `sktime-core`, with the option to split into
a dedicated `sktime-pipeline` package if complexity warrants. Each component package
will manage its own dependencies, extras, and release cadence, following semantic
versioning. A clear migration path and enhanced interoperability testing will be
established.

## Motivation

- **Maintainability:** Smaller, focused packages are easier to maintain, test, and
  document.
- **Scalability:** Modularisation allows the project to scale with new contributions and
  features.
- **Dependency management:** Isolating dependencies per package reduces conflicts and
  installation issues.
- **Release flexibility:** Independent release cycles enable faster bug fixes and
  feature delivery.
- **User experience:** Users can install only what they need, reducing bloat and
  complexity.

## Discussion and comparison of alternative solutions

- **Status quo:** Retaining the monolithic structure avoid the overhead of managing
  multiple packages, but would continue to exacerbate maintenance and dependency issues
  as the project grows.
- **Partial modularisation:** Splitting only some modules would not fully address
  dependency and release bottlenecks and coordination challenges.
- **Meta-package (`sktime-all`):** While useful for transition, maintaining a
  meta-package long-term increases maintenance overhead and can reintroduce dependency
  conflicts.

The proposed full modularisation, with a shared core and clear migration strategy,
offers a balance of maintainability, flexibility, scalability, and user experience.

## Detailed description of design and implementation

### Package structure

- **sktime-core:** Contains all base class definitions, shared utilities, and
  (initially) pipeline/composing logic.
- **Component packages:**
  - `sktime-forecasting`
  - `sktime-classification`
  - `sktime-transformations`
  - ...and others as needed.
- **Optional:** If pipeline logic grows in complexity, introduce `sktime-pipeline` as a
  separate package.
- **Meta-package:** `sktime` (which essentially is `sktime-all`) would be retained only
  for transition, to be deprecated and removed post-migration.

### Migration strategy

- Announce the modularisation plan and timeline to the community.
- Provide migration guides and automated scripts where possible.
- Deprecate monolithic imports with clear warnings and documentation.
- Maintain `sktime-all` as a transitional meta-package, to be deprecated after a defined
  period.
- Ensure all new features and fixes are developed in the new packages post-split.

### Interoperability and testing

- Develop a dedicated integration test suite to verify interoperability across component packages and pipelines.
- Include real estimators with soft dependencies in CI, not just dummy ones.
- Use a CI matrix to test combinations of packages, Python versions, and operating
  systems.
- Regularly test all documented dependency bounds, both upper and lower, using automated
  tools.
- Consider a “smoke test” meta-package for CI-only integration testing.

### Release strategy

- Each package follows semantic versioning:
  - **Major:** Breaking changes.
  - **Minor:** New features.
  - **Patch:** Bug fixes and small enhancements.
- Independent release cadence per package, allowing for more agile and targeted
  releases.
- Use the modularisation as an opportunity for a major (`1.0.0`) release, signalling
  stability and the new structure.
- Release management can be distributed among maintainers familiar with specific
  packages.

### Documentation and user support

- Update documentation to reflect the new package structure and installation
  instructions.
- Provide migration guides, FAQs, and clear guidance on selecting and installing
  component packages.
- Clearly document the stability and support status of each package.

## Potential concerns and look out areas

- **Interoperability:** Must ensure pipelines and workflows remain seamless across
  packages; integration testing is critical.
- **User confusion:** Clear documentation and migration support are essential to prevent
  confusion during and after the transition.
- **Maintenance overhead:** More packages mean more CI, releases, and documentation to
  manage; governance and maintainership must scale accordingly.
- **Fragmentation:** Risk of ecosystem fragmentation if not managed carefully; maintain
  a strong shared core and community engagement.
- **Meta-package deprecation:** Plan and communicate the deprecation of `sktime` to
  avoid long-term maintenance burden.
