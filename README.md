# HydroForge

A framework for GPU-accelerated hydrological modelling with Torch, Triton,
native CUDA, and Metal backends.

## Packages

| Package | Contents |
|---|---|
| `hydroforge.model` | Model and module declarations, tensor fields, execution and output configuration |
| `hydroforge.execution` | Managed model steps and substep execution |
| `hydroforge.data` | Eager and lazy model inputs |
| `hydroforge.data.datasets` | Streaming forcing datasets and dataset exports |
| `hydroforge.mapping` | Spatial grids, mapping tables, and offline mapping builders |
| `hydroforge.io` | Construction inputs and multi-rank output readers |
| `hydroforge.kernels` | Kernel specifications and backend implementations |
| `hydroforge.parallel` | Distributed setup and ensemble partitioning |
| `hydroforge.platform` | Device and backend selection |

## Installation

Python 3.11+ is required. Install the appropriate PyTorch build for your device,
then install HydroForge:

```bash
pip install git+https://github.com/Kshy0/hydroforge.git
```

For local development:

```bash
git clone https://github.com/Kshy0/hydroforge.git
cd hydroforge
pip install -e .
```

## Backend selection

The default backend follows the model device: Triton for CUDA/ROCm and XPU,
Metal for MPS, and Torch otherwise. Override it with `HYDROFORGE_BACKEND`:

```bash
export HYDROFORGE_BACKEND=triton  # triton, cuda, metal, or torch
```

Backend availability also depends on the model's kernel implementations.
Models can set a backend-specific mixed-precision default through
`BackendRequirement(default_mixed_precision=...)`; an explicit
`mixed_precision=True` or `False` at construction takes precedence.

On Metal, high-precision values use two FP32 components rather than native FP64.
NetCDF floating outputs default to FP32 independently of mixed precision; set
`OutputConfig(save_precision="float64")` for FP64 files, or `save_precision=None`
to preserve each output's logical dtype. See [Metal precision](docs/METAL_FLOAT32X2.md).

Tensor declarations can use `depends_on="options.forcing.par"` to select fields
from declared Boolean options at construction. Tuples combine module and option
conditions with AND; disabled fields stay `None`. See [option field conditions](docs/OPTIONS_FIELD_GATES.md).

## Usage

```python
from hydroforge.model import (
    AbstractModel,
    AbstractModule,
    OutputConfig,
    TensorField,
    computed_tensor_field,
    module_ref,
)
from hydroforge.execution import between_steps, managed_step
from hydroforge.data import InputProxy
from hydroforge.data.datasets import (
    DailyBinDataset,
    ERA5LandAccumDataset,
    ExportedDataset,
    MultiVariableDataset,
    NetCDFDataset,
    open_multivariable,
)
from hydroforge.io import MultiRankStatsReader
from hydroforge.kernels import BackendRegistry, KernelSpec
from hydroforge.parallel import setup_distributed
```

`InputProxy.from_nc("parameters.nc", lazy=True)` loads model inputs. Models
use `with model:` to materialize and release resources; `@managed_step` methods
advance them. Set execution options directly on the model with
`execution_mode="auto"` or `"eager"`, `block_size=128`, and an optional
`parallel=mesh`. Configure statistics through `OutputConfig`, with `mean`, `sum`,
`min`, `max`, `first`, and `last` reductions. Results can be kept in memory or
written to NetCDF.

## License

Apache 2.0
