# HydroForge

A framework for GPU-accelerated hydrological modelling with Torch, Triton,
native CUDA, and Metal backends.

## Packages

| Package | Contents |
|---|---|
| `hydroforge.model` | `AbstractModel`, module declarations (`AbstractModule`, `TensorField`, `module_ref`, ...) and `OutputConfig` |
| `hydroforge.contracts` | Typed declarations used by models: `OptionsConfig`, conditions (`module`, `opt`), `option`, `SimulationSchedule`, `SpinupSchedule`, `StatisticsPlan` and its windows, `ParameterChange`, `BackendRequirement`, `ModuleRequirement`, step fields |
| `hydroforge.execution` | Managed model steps and substep execution |
| `hydroforge.data` | Eager and lazy model inputs |
| `hydroforge.data.datasets` | Streaming forcing datasets and dataset exports |
| `hydroforge.mapping` | Spatial grids, mapping tables, and offline mapping builders |
| `hydroforge.io` | Construction inputs and multi-rank output readers |
| `hydroforge.kernels` | Kernel specifications and backend implementations |
| `hydroforge.parallel` | Distributed setup and ensemble partitioning |
| `hydroforge.platform` | Device and backend selection |
| `hydroforge.statistics` | Statistics programs, windows and runtime behind `OutputConfig` |
| `hydroforge.testing` | Single-module construction and kernel interception for tests |
| `hydroforge.core` | Dependency-free validation, errors, calendars and arrays |

Internal layers: `hydroforge.declare` holds the declarations re-exported by
`hydroforge.model`, and `hydroforge.compiler` compiles a declaration into the
frozen plan exposed as `model.plan`.

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
`hydroforge.contracts.BackendRequirement(default_mixed_precision=...)`; an explicit
`mixed_precision=True` or `False` at construction takes precedence.

On Metal, high-precision values use two FP32 components rather than native FP64.
NetCDF floating outputs default to FP32 independently of mixed precision; set

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
from hydroforge.contracts import OptionsConfig, SimulationSchedule, StatisticsPlan
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

### Kernels, modules and options

Kernel specs never read options. Every module receives the model's root
options as its `options` field (a model-owned field like `opened_modules`,
never read from inputs or written to checkpoints and outputs) and derives
kernel flags and constants from them; specs bind module fields by exact name
(`module_flag(...)`, `constant(...)` served by a `kernel_field`):

```python
class Soil(AbstractModule):
    options: MyOptions                      # typed access to the model options

    @kernel_field
    def SOIL_SOLVER(self) -> int:
        return self.options.option_code("soil.solver")
```

Registered kernels bind their arguments from the model automatically, and
launch eagerly, wherever the model runs them: inside `@managed_step` bodies,
inside `initialize_model_state()` (cold starts, derived state) and inside
`@between_steps` methods (for example re-deriving parameters after a setter
copied new values). Kernels that read step fields should be called from
managed steps, where the step's values are prepared. Elsewhere a registered
kernel call is an error.

## License

Apache 2.0
