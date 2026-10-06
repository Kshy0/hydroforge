# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Public driver-side model inputs and input-pipeline utilities."""

from hydroforge.data.input import InputProxy
from hydroforge.data.transfers import prefetch_to_device

__all__ = [
    "InputProxy",
    "prefetch_to_device",
]
