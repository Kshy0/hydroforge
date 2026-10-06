# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Device predicates shared by kernel binding and execution."""

import torch


def devices_match(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    if actual.type == "cpu":
        # PyTorch accepts cpu:N but CPU tensors have no selectable index.
        return True
    if actual.type == "mps":
        # PyTorch commonly reports an MPS tensor as ``mps:0`` even when the
        # requested model device was the equivalent singleton spelling
        # ``mps``.  MPS exposes no selectable multi-device index today.
        return actual.index in {None, 0} and expected.index in {None, 0}
    return expected.index is None or actual.index == expected.index
