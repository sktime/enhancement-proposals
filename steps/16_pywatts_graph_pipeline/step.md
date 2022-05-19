# Integrating the pyWATTS' Graph Pipeline

Contributors: @kphipps, @fkiraly, @smeisen, @aiwalter, @benheid

## Introduction

Workflows in ML projects are often non-sequential. E.g., a regressor gets as input statistical features, calendar 
features, and historical values. All of these information are extracted from the same series or dataarray.

To realise such mappings via a pipeline, the pipeline has to be non-sequential.

[concise introduction to problem and overview of proposed solution]

For preliminary discussions of the proposal presented here, see issue: [links to issues/pull requests]

## Contents
[table of contents]

## Problem statement
[concise problem statement]
Workflows in ML projects are often non-sequential. E.g., a regressor gets as input statistical features, calendar 
features, and historical values. All of these information are extracted from the same series or dataarray.

To realise such mappings via a pipeline, the pipeline has to be non-sequential.

## Description of proposed solution

pyWATTS solves this problem by proposing a graph pipeline. Based on this solution, we can aim to enable the usage of 
such pipelines in pyWATTS.

* Pipeline
  * Tasks
* Step
  * Tasks:
* Module
  * Tasks:
* StepInformation
  * Task

* How are the modules added to the pipeline
  * Functional API

* How is the data flow be realised
  * Step fetches the data and maintains a buffer

## Motivation

## Discussion and comparison of alternative solutions

## Detailed description of design and implementation of proposed solution 


[prototypical implementation if applicable]
