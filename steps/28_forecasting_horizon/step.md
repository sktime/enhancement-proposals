# Design Document for ForecastingHorizon

Contributors: [RecreationalMath]

## Table of Contents
[TOC]

## Problem statement & Motivation
`ForecastingHorizon` is strongly coupled with `pandas.Index` and often leads to breakages as `pandas` makes inconsistent updates to their `Index` in every subsequent major/minor release. They neither seem to be following a consistent pattern for the updates nor going in the direction of a single ideal. So decoupling it is the pragmatic solution to keep the maintenance load to a minimum going forward.

[Github sktime/Issue#7617](https://github.com/sktime/sktime/issues/7617) 

<!-- draft PR [sktime/#]() -->

## Discussion and comparison of alternative solutions

<!--
Independent internal representation of ForecastingHorizon
- Can be acheived. A new base class `ForecastingHorizonValues` handles the functionality currently outsourced to pandas
- `ForecastingHorizon` relatively remains same, with same functions and signatures. Minor changes.
-->

A lot of the below points are interlinked and might have some repetition, but considering them under different topics is important to know their impact on different aspects of the class.

1. Interface Compatibility & Backward Compatibility

    - Q1.1. Whether the new class should be a drop-in replacement that maintains 100% API compatibility with current implementation, because end users already use it and expect it to behave in a certain way? And changing it might need following deprecation patterns and introduce complexities. Or a refactor of the interface itself can also be considered? 
    - Consideration1: Assuming some (many?) users would be using the exposed methods of `ForecastingHorizon` in their workflows and preserving them from breaking by having the same methods still exist after the rework, is of importance. 
    - Consideration2: Changing the interface would require maintaining both classes for a while and a complicated deprecation period.
    - Conflict: 
        - Keeping same interface means `ForecastingHorizon` accepts the pandas objects as inputs. Which puts the responsibility to manage the `freq` mnemonics changes by `pandas` on `sktime`. Same problem as before.
        - If we want to ensure absolutely no breakage with regards to `freq` mnemonics change, the onus of converting to a `ForecastingHorizon` compatible input type needs to be transfered to the end-user. `sktime` can obviously help by provide a utility for conversion. But if we do two things will happen. 
            1. current working workflows of end-users will break. We can use deprecation and handle it gracefully.
            2. yt will add one extra step before/after using the sktime forecasters -> to convert the forecasting horizon from pandas to fh_compatible input types. Seems tedious, need inputs and thoughts.
    

2. Internal Data Representation

    - Q2.1 Should the class still support all input types (int, list, np.ndarray, pd.Index, timedelta, date offsets)?
    - Consideration: Same concern as above. If we support the pandas objects as input types, there would always be a chance of breakages whenever pandas makes changes as it currently happens. (When our convertor tries to convert an unseen freq type to map it to the internal representation of the custom `fh`)
    - Q2.2: Instead of wrapping pd.Index, what should be the primary internal representation?
    - Option A: numpy arrays for numeric values + custom metadata objects (least coupling with pandas)
    - Option B: WILD IDEA!! what if instead of de-coupling we embrace the pandas fully and don't write converters but use pandas objects directly? Need investigation on what changes would that wrrant.

    
3. Pandas Interoperability (between different parts of sktime)

    - Q3.1 sktime forecasters return pandas Series/DataFrames with pandas Index objects? Does that mean the new class still needs to provide `to_pandas()` and `to_numpy()` methods for integration?
    - Accept pandas Index objects as inputs but convert them immediately to internal representation? This would result in same concerns.

4. Type Handling & Frequency Management

    - Q4.1 For relative vs. absolute distinction, what are the tradeoffs between keeping the current type-based approach (inferring from value types) v/s using explicit metadata/flags instead?
    - Q4.2 For frequency management (the complex freq property with setter) since a lot of breakages happen here, is there a way to make it optional instead of required? Will it break anything?

5. MultiIndex support
    - This is a new feature request. Can be done but the approach will depend on answers to other questions, i.e. how much away or close the picked end-state is to pandas.

6. Performance & Caching

I'm a bit rusty on my threading/caching/memory-leakages concepts and have used AI coding assistant's help in deliberating this part. Would appreciate inputs here. 
Please feel free to validate/invalidate below points.

- Hashability concerns: 
    - aren't pandas Index objects hashable? 
    - current code uses _HashIndex wrapper, this is also source of coupling with pandas. 
- Cache and thread safety concerns: 
    - if many instances are created - is this implementation memory safe? 
    - does lru_cache exhibit thread-safety?

7. Maintenance load

it seems edge-cases (and corresponding workarounds) will always be needed as long as there exists a code internal to `ForecastingHorizon` which converts to or from pandas objects. Unless that piece is specifically moved out and exposed to end-user and forcing them to make it part of their forecasting pipeline. Even after doing so, it will still break, but the fault localisation for the end-user might be easier and some heavy users might choose to write their own wrappers on top of `sktime` provided convertor utility. Giving end-users a way to fix their broken pipelines and also maybe sharing the blame for breakages with pandas. 

## Description of proposed solution

Being deliberated upon.

## Detailed description of design and implementation of proposed solution 
TBD.