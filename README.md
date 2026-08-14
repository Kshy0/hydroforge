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

The main model-building interfaces are available from `hydroforge.model`:

```python
from hydroforge.model import (
    AbstractModel,
    AbstractModule,
    TensorField,
    computed_tensor_field,
    kernel_field,
    managed_step,
)
```

- `AbstractModel`: base class for a complete model.
- `AbstractModule`: base class for optional model components.
- `TensorField`: declares model and module tensors.
- `computed_tensor_field`: declares tensors computed during initialization.
- `kernel_field`: exposes a precomputed value to kernel argument inference.
- `managed_step`: manages one public model step.

Tensor storage can depend on optional modules. A conditional field remains
`None` and is excluded from input loading, partitioning, runtime namespaces,
kernel ownership, and checkpoints unless every named module is open:

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

Use a tuple when storage requires multiple modules, for example
`depends_on=("log", "routing_diagnostics")`. Supplying an inactive conditional
field explicitly is an error.

Use `required_by=("bifurcation", "reservoir")` when storage is required if
at least one listed consumer module is open. Only `init_state` tensors are
checkpointed; runtime workspaces are reconstructed after restoration.

A model defines its physical execution order directly:

```python
@managed_step
def step_advance(self):
    for substep in self.substeps.fixed():
        route_flow()
        update_storage()
```

With a simulation schedule, the driver stages forcing and advances the model:

```python
model.set_inputs(runoff)
model.step_advance()
```

Without a schedule, pass `time_step=timedelta(...)` when calling the method.
`time_step`, `num_sub_steps`, and `output_enabled` are framework call options,
not parameters of the decorated method. Fixed-step models may use
`model.step_advance(num_sub_steps=360)`; omitting it lets the model choose its
default. Adaptive models must omit `num_sub_steps`.

Adaptive models use `self.substeps.adaptive(...)` and call
`substep.resolve_dt()` between timestep proposal and physical routing.

## Inputs and datasets

`InputProxy` loads model parameters eagerly or lazily:

```python
from hydroforge import InputProxy

parameters = InputProxy.from_nc("parameters.nc", lazy=True)
```

Streaming forcing datasets are available from `hydroforge.data.datasets`:

```python
from hydroforge.data.datasets import (
    AbstractDataset,
    DailyBinDataset,
    ERA5LandAccumDataset,
    ExportedDataset,
    GriddedDataset,
    MultiVariableDataset,
    NetCDFDataset,
    open_multivariable_exported,
    open_multivariable_netcdf,
)
```

Gridded datasets provide mapping and export helpers such as
`select()`, `export_climatology()`, and `export_catchment_data()`.

## Model and statistics clocks

Model schedules and statistics windows are explicit:

```python
from datetime import timedelta
from hydroforge import (
    CalendarWindow,
    StatisticsPlan,
)

runoff_dataset = DailyBinDataset(
    ...,
    model_step=timedelta(days=1),
)
schedule = runoff_dataset.simulation_schedule

statistics_plan = StatisticsPlan(
    inner=CalendarWindow("day"),
    outer=CalendarWindow("year"),
)
```

Dataset chunks are not padded; the final chunk may be shorter than `chunk_len`.
Use `DataLoader(dataset, batch_size=None, ...)` to yield chunks directly.

## Statistics and NetCDF output

Models select variables and aggregation operations with `variables_to_save`.
Supported reductions are `mean`, `sum`, `max`, `min`, `first`, and `last`.
NetCDF variable options are passed through validated mappings:

```python
model = Model(
    ...,
    variables_to_save={
        "mean": ["discharge"],
        "max": ["water_depth"],
    },
    output_netcdf_options={
        "compression": "zlib",
        "complevel": 4,
        "chunksizes": (24, 1024),
    },
    checkpoint_netcdf_options={
        "compression": "zlib",
        "complevel": 4,
    },
)
```

Dataset export methods use the same `netcdf_options` mapping.

Multi-rank model output can be read with:

```python
from hydroforge.output.multirank import MultiRankStatsReader
```

Statistics output uses contract version 3 and one `hydroforge_run_id` across
ranks and split years. The reader rejects legacy output and mixed-run files.
Background output failures stop subsequent model steps.

Boolean NetCDF variables use `u1` storage with
`hydroforge_dtype="bool"`; values must be `0` or `1`.

## Backend selection

Set `HYDROFORGE_BACKEND` when an explicit backend is required:

```bash
export HYDROFORGE_BACKEND=triton
export HYDROFORGE_BACKEND=cuda
export HYDROFORGE_BACKEND=metal
export HYDROFORGE_BACKEND=torch
```

The selected backend and model precision are validated during initialization.

## License

Apache 2.0
