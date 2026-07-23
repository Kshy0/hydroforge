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
    copy_input,
)
```

- `AbstractModel`: base class for a complete model.
- `AbstractModule`: base class for optional model components.
- `TensorField`: declares model and module tensors.
- `computed_tensor_field`: declares tensors computed during initialization.
- `kernel_field`: exposes a precomputed value to kernel argument inference.
- `managed_step`: manages one public model step.
- `copy_input`: copies caller data into stable model storage.

A model defines its physical execution order directly:

```python
@managed_step
def step_advance(self, runoff, time_step, current_time=None):
    copy_input(self.base.runoff, runoff, name="runoff")

    for substep in self.substeps.fixed(count=self.num_sub_steps):
        route_flow()
        update_storage()
```

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

## Model and forcing clocks

Schedules, forcing resampling, and statistics windows are explicit:

```python
from datetime import timedelta
from hydroforge import (
    CalendarWindow,
    ForcingPlan,
    ForcingSource,
    SimulationSchedule,
    StatisticsPlan,
)

schedule = SimulationSchedule.from_contract(
    runoff_dataset.temporal_contract(),
    step=timedelta(hours=1),
)

forcing_plan = ForcingPlan.bind(
    schedule=schedule,
    runoff=ForcingSource(
        runoff_dataset.temporal_contract(),
        semantics="mean_rate",
        resampling="hold",
    ),
)

statistics_plan = StatisticsPlan(
    schedule=schedule,
    inner=CalendarWindow("day"),
    outer=CalendarWindow("year"),
)
```

Use `ForcingPlan.bundle(...)` to stream synchronized inputs into successive
`step_advance` calls.

## Statistics and NetCDF output

Models select variables and aggregation operations with `variables_to_save`.
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
