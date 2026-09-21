# HydroForge

HydroForge is a framework for building GPU-accelerated hydrological models
with Torch, Triton, native CUDA, and Metal backends.

## Installation

Python 3.11+ is required. Install the appropriate PyTorch build first, then
install HydroForge:

```bash
pip install git+https://github.com/Kshy0/hydroforge.git
```

For local model development:

```bash
git clone https://github.com/Kshy0/hydroforge.git
cd hydroforge
pip install -e .
```

## Model API

Model declarations are available from `hydroforge.model`, and execution helpers
from `hydroforge.execution`:

```python
from hydroforge.execution import ManagedStep, between_steps, managed_step
from hydroforge.model import (
    AbstractModel,
    AbstractModule,
    TensorField,
    computed_tensor_field,
    kernel_field,
)
```

- `AbstractModel`: base class for a complete model.
- `AbstractModule`: base class for optional model components.
- `TensorField`: declares model and module tensors.
- `computed_tensor_field`: declares tensors computed during initialization.
- `kernel_field`: exposes a precomputed value to kernel argument inference.
- `managed_step`: manages one public model step.

Tensor storage can depend on optional modules:

```python
import torch

diagnostic: torch.Tensor | None = TensorField(
    "Per-cell diagnostic",
    shape=("base.num_cells",),
    category="state",
    default=0,
    depends_on="log",
)
```

`depends_on` requires every named module to be open; `required_by` requires at
least one listed consumer to be open or request output. `output_only` allocates
storage only for directly requested fields. Inactive tensors are `None`, and
virtual expressions do not allocate storage. Use `materialized_outputs` to
request resident fields without writing them.

Among state categories, only `init_state` tensors are checkpointed;
reconstruction also retains parameters and topology. Distributed checkpoints
support unequal or empty spatial partitions. Ensemble checkpoint saving is
unsupported.

Models stage forcing and declare physical execution order directly:

```python
@between_steps
@torch.inference_mode()
def set_inputs(self, *, runoff: torch.Tensor) -> None:
    self.base.runoff.copy_(runoff)

@managed_step
def step_advance(self, step: ManagedStep) -> None:
    for _substep in step.fixed():
        route_flow()
        update_storage()
```

```python
model.set_inputs(runoff=runoff)
model.step_advance()
```

Without a simulation schedule, pass `time_step=timedelta(...)` to the managed
step. Fixed-step models may also pass `num_sub_steps`. Execution methods must be
synchronous. Consume compiled scope iterators normally; use device predicates
for runtime termination rather than Python `break` or `return`.

Model-wide options use immutable `OptionsConfig` objects:

```python
from enum import StrEnum

from hydroforge.contracts import OptionField, OptionsConfig

class RoutingOption(StrEnum):
    KINEMATIC = "kinematic"
    DIFFUSIVE = "diffusive"

class ModelOptions(OptionsConfig):
    routing: RoutingOption = OptionField(
        RoutingOption.KINEMATIC,
        codes={RoutingOption.KINEMATIC: 0, RoutingOption.DIFFUSIVE: 1},
        requires_modules={
            RoutingOption.KINEMATIC: (),
            RoutingOption.DIFFUSIVE: ("floodplain",),
        },
        description="River-routing formulation.",
    )
```

Selected options activate required modules and validate declared backend and
forcing requirements. Kernels bind compile-time values through `option_code`,
`config_value`, and `output_requested`; `KernelSpec.step_fields` binds read-only
device scalars for time-dependent values.

## Inputs and datasets

`InputProxy` loads model parameters eagerly or lazily:

```python
from hydroforge.data import InputProxy

parameters = InputProxy.from_nc("parameters.nc", lazy=True)
```

Streaming forcing datasets are available from `hydroforge.data.datasets`:

```python
from hydroforge.data.datasets import (
    DailyBinDataset,
    ERA5LandAccumDataset,
    ExportedDataset,
    NetCDFDataset,
    SourceDataset,
    open_multivariable_exported,
    open_multivariable_netcdf,
)
```

Gridded datasets use `build_local_mapping()` for source selection and
`shard_forcing()` for device-side mapping. Dataset chunks are not padded; use
`DataLoader(dataset, batch_size=None, ...)` to yield chunks directly.
Time keys accept `datetime` and `cftime.datetime` and preserve their calendars.

Distributed drivers select devices explicitly:

```python
from hydroforge.data import setup_distributed

distributed = setup_distributed(allowed_devices=("cuda", "cpu"))
device = distributed.device
rank = distributed.rank
world_size = distributed.world_size
```

Ensemble-enabled models support `EnsembleParallel` for member and spatial
partitioning. Input arrays use global member axes; model buffers use local
member slices. Collectives default to spatial scope. Close models before
leaving the parallel context.

## Model and statistics clocks

Dataset model steps and statistics windows are explicit:

```python
from datetime import timedelta

from hydroforge.contracts import CalendarWindow, StatisticsPlan

runoff_dataset = DailyBinDataset(
    ...,
    model_step=timedelta(days=1),
)

model = Model(
    ...,
    statistics_plan=StatisticsPlan(
        inner=CalendarWindow(period="day"),
        outer=CalendarWindow(period="year"),
    ),
)
```

## Statistics and NetCDF output

Models select variables and aggregation operations with `variables_to_save`.
Supported reductions are `mean`, `sum`, `max`, `min`, `first`, and `last`.
Declared tensors use `output="auto"` to apply an available `SelectionField`,
`output="full"` to save the full domain, or `output="disabled"` to reject direct
output. Statistics expressions retain numeric precision and member layout.

The default NetCDF profile is lossless Blosc-Zstd level 5 with byte shuffle,
with zlib level 4 as a fallback. Set zlib explicitly for files that must be
readable without the Blosc plugin:

```python
output_netcdf_options={
    "compression": "zlib",
    "complevel": 4,
}
```

Quantization is disabled by default. Streaming output selects approximately
4 MiB chunks when `chunksizes` is omitted. Dataset exports use the same
`netcdf_options` mapping. Boolean variables use `u1` storage with
`hydroforge_dtype="bool"`.

Output uses two workers by default; `output_workers=0` writes synchronously.
Closing the model waits for pending output. Multi-rank output can be read with:

```python
from hydroforge.output.multirank import MultiRankStatsReader
```

The reader exposes the common committed time prefix across rank files.

## Backend selection

Set `HYDROFORGE_BACKEND` to select a backend:

```bash
export HYDROFORGE_BACKEND=triton
export HYDROFORGE_BACKEND=cuda
export HYDROFORGE_BACKEND=metal
export HYDROFORGE_BACKEND=torch
```

Backend availability depends on the model and device. Native CUDA compilation
uses `HYDROFORGE_PRECOMPILE_JOBS` for build processes and `MAX_JOBS` for Ninja
jobs within each process.

## License

Apache 2.0
