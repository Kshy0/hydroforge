# HydroForge

HydroForge is a framework for building GPU-accelerated hydrological models
with Torch, Triton, native CUDA, and Metal backends.

## Installation

Install the appropriate PyTorch build first, then install HydroForge:

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

Model declarations and execution helpers are separated by namespace:

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

Use a tuple for multiple required modules. `required_by` declares storage used
when any listed consumer is open. Only `init_state` tensors are checkpointed.

Conditional tensor storage follows three rules:

- `depends_on`: every named module must be open.
- `required_by`: at least one named consumer must be open or request output.
- `output_only`: storage exists only when the field is requested directly.

Inactive computed tensors are exposed as `None`. Virtual expression fields are
symbolic and do not allocate storage.

`variables_to_save` directly requests its declared fields. Alias expressions
observe their source fields without activating storage. For a conditional field
with a same-named alias, the tensor is used when active and the alias expression
is used when inactive. Use `materialized_outputs=("module.field", ...)` when a
field must be resident without being written by the statistics system.

Kernels can bind output-dependent features with
`output_requested(module, field)`. The feature is fixed for the compiled model
specialization across all backends.

A model defines forcing staging and physical execution directly:

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
step. Fixed-step models may also pass `num_sub_steps`.

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

Distributed drivers select devices explicitly:

```python
from hydroforge.data import setup_distributed

distributed = setup_distributed(
    allowed_devices=("cuda", "cpu"),
)
device = distributed.device
rank = distributed.rank
world_size = distributed.world_size
```

## Model and statistics clocks

Model schedules and statistics windows are explicit:

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
For a declared tensor, `output="auto"` applies the matching `SelectionField`
when one exists and otherwise saves the full logical domain; `output="full"`
always saves the full domain, while `output="disabled"` rejects direct output.
Output policy controls serialization; field dependencies and output demand
control storage allocation.

The default NetCDF profile is lossless Blosc-Zstd level 5 with byte shuffle:

```python
output_netcdf_options={
    "compression": "blosc_zstd",
    "complevel": 5,
    "blosc_shuffle": 1,
}
```

HydroForge falls back to zlib level 4 when Blosc is unavailable or a fixed
chunk is too small. Set zlib explicitly for files that must be readable without
the Blosc plugin. Quantization is not enabled by default.

When `chunksizes` is omitted, streaming output chooses an approximately 4 MiB
layout aligned with the write batch. Dataset export methods use the same
`netcdf_options` mapping.

Multi-rank model output can be read with:

```python
from hydroforge.output.multirank import MultiRankStatsReader
```

Boolean variables use `u1` storage with `hydroforge_dtype="bool"`.

## Backend selection

Set `HYDROFORGE_BACKEND` when an explicit backend is required:

```bash
export HYDROFORGE_BACKEND=triton
export HYDROFORGE_BACKEND=cuda
export HYDROFORGE_BACKEND=metal
export HYDROFORGE_BACKEND=torch
```

## License

Apache 2.0
