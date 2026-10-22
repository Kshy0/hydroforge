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
pip install hydroforge==0.3.0
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

Metal has no native FP64. With mixed precision, Metal stores high-precision
values as two FP32 components (`metal_emulation="auto"` or `"float32x2"`);
without mixed precision they are FP32.
Statistics outputs default to FP32 independently of mixed precision; set
`OutputConfig(save_precision="float64")` to keep FP64 values, or
`save_precision=None` to keep each statistic's own precision. Integer and
boolean outputs are never converted.

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
never read from inputs; checkpoints and `parameters.nc` record only a
fingerprint of the options they were built with) and derives
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
inside `initialize_model_state()` (initialization) and inside
`@between_steps` methods (for example re-deriving parameters after a setter
copied new values). Step fields hold the current step's values inside
managed steps and the initial clock inside `initialize_model_state()`, where
`step_seconds` is the first scheduled step's width (unavailable without a
`simulation_schedule`). Elsewhere a registered kernel call is an error.
A spec parameter declared `step_field("step_seconds")` (or another calendar
value) binds the current outer step's value; one declared
`substep_field("dt")` or `substep_field("index")` binds the enclosing fixed or
adaptive substep's width in seconds (a fixed loop's is the step width divided
by the substep count) or its int32 index, and is available only inside a
substep body.

### Initialization

1. The plan tabulates the fields of the opened modules; the input fields the
   model input lacks are `model.missing_inputs` (`"module.field"`).
2. Construction reads the inputs (`param`, `topology`, `init_state`; defaults
   where absent) and evaluates the computed `topology`, `derived_param` and
   `state` tensors; buffer virtuals declaring `materialize=` conditions are
   allocated. Inputs stay as read, so a checkpoint saves the parameters and
   topology it was built from; prefer adapting a kernel to rewriting one.
3. `initialize_model_state()` updates state in place, in explicit order, and
   may run registered kernels. It branches on `missing_inputs`: a checkpoint
   supplies every `init_state` field, a cold start lacks some.
4. Computed state declared `after_init=True` is evaluated from the
   initialized state, for values that simply follow from it.

```python
class Model(AbstractModel):
    def initialize_model_state(self) -> None:
        if self.dam is not None and "base.storage" in self.missing_inputs:
            self.dam.fill()                     # cold start only


class Base(AbstractModule):
    @computed_tensor_field("Depth", ("n",), category="state", after_init=True)
    @cached_property
    def depth(self) -> torch.Tensor:
        return self.storage / self.area
```

Computed tensors are made contiguous unless they declare `contiguous=False`.
An input declared `mode="discard"` is released after construction; derived
parameters read from it keep their value and cannot be refreshed by a
parameter change.

## License

Apache 2.0
