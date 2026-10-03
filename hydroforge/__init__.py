# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
hydroforge: Generic framework for GPU-accelerated hydrological modelling.

Subpackages, from the lowest layer to the highest
-------------------------------------------------
core        Dependency-free validation, errors, calendars, arrays, and expressions.
contracts   Immutable field, input, and kernel contracts.
io          Atomic files, NetCDF and binary formats, and rank-partitioned output.
parallel    Launcher environment, device selection, process meshes.
platform    Kernel backend facts and selection, environment variables.
kernels     Kernel registration, toolchains, code generation and Torch/Triton/CUDA/
            Metal backends.
declare     Module and model declarations, tensor fields, and their frozen specs.
data        Model inputs, forcing datasets, and input-pipeline utilities.
mapping     Sparse spatial mapping tables and offline mapping builders.
statistics  Statistics IR, windows, kernel plans, backend programs, and runtime.
compiler    Pure compilation of a model declaration into its frozen plan.
execution   Model runtime: input binding, materialization, step orchestration,
            structural updates, checkpoints, and backend capture.
model       AbstractModel and the public declarative-model API.
testing     Module construction helpers for tests (same layer as model).
"""

__all__: list[str] = []
