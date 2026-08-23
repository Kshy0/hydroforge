# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
hydroforge: Generic framework for GPU-accelerated hydrological modelling.

Subpackages
-----------
contracts   Immutable field, input, and kernel contracts.
model       Declarative AbstractModule/AbstractModel API.
compiler    Cold-path model specialization and immutable plans.
data        Datasets, distributed loading, and spatial mappings.
serialization Shared file-format contracts and atomic serialization primitives.
output      Checkpoint, NetCDF, and multi-rank output facilities.
statistics  Statistics IR, emitters, and runtime.
kernels     Kernel registration and Torch/Triton/CUDA/Metal backends.
execution   Compiled step orchestration, input staging, and backend capture.
"""

__all__: list[str] = []
