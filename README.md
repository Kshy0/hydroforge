# HydroForge

A framework for GPU-accelerated hydrological modelling with Torch, Triton,
native CUDA, and Metal backends. Use it through a model package such as
[CaMa-Flood-GPU](https://github.com/Kshy0/CaMa-Flood-GPU) or
[WRF-GPU](https://github.com/Kshy0/WRF-GPU); that package supplies the physics,
input preparation tools, forcing interfaces, and model options.

## Installation

Python 3.11+ is required. Install a PyTorch build for your device, then
HydroForge and your chosen model package:

```bash
pip install git+https://github.com/Kshy0/hydroforge.git
```

The Triton backend also needs a compatible Triton installation. Follow the
model package's installation instructions for supported devices, backends,
and any additional runtime data.

For a local HydroForge checkout:

```bash
git clone https://github.com/Kshy0/hydroforge.git
cd hydroforge
pip install -e .
```

## Backend and precision

Set the backend before starting Python:

```bash
export HYDROFORGE_BACKEND=cuda  # cuda, triton, metal, or torch
```

Without an override, CUDA/ROCm and XPU devices use Triton, MPS uses Metal,
and CPU uses Torch. The model must implement the selected backend.

Pass precision settings when constructing the model:

| Setting | Use |
|---|---|
| `precision="float32"` | FP32 computation |
| `precision="float64"` | FP64 computation, where supported |
| `storage_precision="float64"` | Higher-precision storage with FP32 computation |
| `execution_mode="eager"` | Run without captured execution; the default is `"auto"` |

Storage precision must be at least the compute precision. If it is omitted,
`mixed_precision` and the model's backend defaults choose it. Metal has no
native FP64; supported models can use `metal_emulation="float32x2"` for
higher-precision storage. `tf32`, `bfloat16`, and `float16` are available only
where the selected model and backend support them.

## Run a model

Prepare inputs with the model package's tools. `InputProxy.from_nc` opens
prepared NetCDF inputs; it does not convert arbitrary source data into a
model's input format.

This WRF example requires the WRF-GPU package, a CUDA device, and a prepared
`parameters.nc` matching the options and start time below. Other models have
their own step methods and forcing setters; use their examples for those
parts.

```python
from contextlib import closing
from datetime import datetime, timedelta

import torch
from hydroforge.contracts import SimulationSchedule
from hydroforge.data import InputProxy
from hydroforge.model import OutputConfig
from wrfgpu.config import WRFOptions
from wrfgpu.models import WRF

options = WRFOptions(mp_physics=1, time_step=2.0)
schedule = SimulationSchedule.regular(
    start=datetime(2000, 6, 1),
    end=datetime(2000, 6, 1, 0, 1),
    step=timedelta(seconds=options.time_step),
)

with closing(InputProxy.from_nc("parameters.nc", lazy=True)) as inputs:
    with WRF(
        input_proxy=inputs,
        options=options,
        device=torch.device("cuda"),
        precision="float32",
        simulation_schedule=schedule,
        output=OutputConfig(
            dir="out",
            experiment="forecast",
            variables={"last": ("u", "v", "moist")},
        ),
    ) as model:
        for _ in schedule:
            model.step_advance()
        model.export_html("model.html")
```

Keep model calls inside `with model:` and close input proxies when finished.
The schedule supplies step times and durations; WRF requires its step duration
to equal `options.time_step`. Update time-varying forcing through the model's
public setter before each step when the model requires it.

## Outputs

`OutputConfig` selects fields and reductions such as `mean`, `sum`, `min`,
`max`, `first`, and `last`. With no `statistics_plan`, outputs are sampled
every model step. The example writes NetCDF files beneath `out/forecast`.
Use the model's documented field names and units.

Floating-point output defaults to FP32. Set `save_precision="float64"` to
save FP64, or `save_precision=None` to preserve each result's precision.
Integer and boolean outputs retain their types.

To keep results in memory, replace the output configuration with:

```python
OutputConfig(sink="memory", variables={"last": ("u", "v")})
```

Read them before leaving the model context:

```python
u = model.results.get("u", op="last")  # tensor stacked over output times
```

For a subset of a structured grid, select the model's named global axes:

```python
from hydroforge.contracts import Out, OutputView, Range

OutputConfig(
    dir="out",
    views={"box": OutputView(bottom_top=Range(10), west_east=Range(100, 200))},
    variables={"last": (Out("u", view="box"),)},
)
```

Paired grid points can be configured without model fields:
`OutputView(points={"y": [4, 1, 4], "x": [2, 8, 2]}, point_dimension="gauge")`.
This selects (4, 2), (1, 8), (4, 2), preserving order and repeated locations,
and requires every logical grid axis of the output. Each rank writes its
owned points, with their positions in the complete list as the output
coordinate. Axis-range views retain their Cartesian shape; point views have
one point dimension.

Choose ranges within your domain. Distributed outputs contain owned rows
only. For point-based models, `hydroforge.io.MultiRankStatsReader` reads
saved point series across rank files by point ID.

## Checkpoints and restart

Save between steps while the model context is open:

```python
with closing(model.save_state()) as checkpoint:
    checkpoint_path = checkpoint.file_path
    restart_time = model.current_time
    print(checkpoint_path, restart_time)
```

`save_state()` requires `OutputConfig(dir=...)` and currently supports
non-ensemble models only. A checkpoint contains construction inputs and
physical state. Record the continuation time and keep the same model options
and opened modules.

To restart, open `checkpoint_path` with `InputProxy.from_nc(..., lazy=True)`
and construct the same model with that input. Supply a new
`SimulationSchedule` starting at `restart_time`, the desired end time, and
compatible forcing. The checkpoint does not choose the continuation schedule
for you.

## Coupling modules

One model schedules its modules; a component such as an atmosphere or an ocean
is a module. `spatial_domains` names one slab decomposition per grid, over the
same spatial ranks, and may accompany a `partition_key`. A module states its
domain with `spatial_domain`, and a field belongs to the domain its shape names.

Modules declare what they exchange. `exports` and `imports` are checked when
the model class is defined, and `exchange_` runs an import through a weighted
link table of global point IDs, held as ordinary fields of a coupling module
so that they and any accumulators are checkpointed. A slab field's IDs index
its `Port.dims` row-major over their global extents (default: every
dimension, so `(lat, lon)` cells); a partition field's are coordinate IDs and
located entities' their place in the complete table. Other dimensions are
carried along each point and must match between the two ends:

```python
class Atmosphere(AbstractModule):
    spatial_domain = "air"
    exports = {"heat": Port("heat_mean"),  # (num_ja, num_ia) cells
               "tracer": Port("tracer", dims=("num_ja", "num_ia"))}  # carries ch

class Ocean(AbstractModule):
    spatial_domain = "sea"
    imports = {"heat": Port("heat_in", source="atmosphere.heat"),
               "tracer": Port("tracer_in", source="atmosphere.tracer",
                              dims=("num_jo", "num_io"))}

class Coupled(AbstractModel):
    spatial_domains = {"air": SlabAxis(dim="num_ja", halo=2),
                       "sea": SlabAxis(dim="num_jo", halo=1)}

    @managed_step
    def step_advance(self, step):
        ...  # component scopes
        exchange_(self, "ocean.heat", target_ids=links.target,
                  source_ids=links.source, weights=links.weight)
```

The result does not depend on the rank count. The exchange runs between
scopes: each rank receives, point to point over the halo links, only the
source points its own links need, so its memory and messages scale with its
share of the grids rather than with the table. Under CUDA graphs it is
captured at its first call.

## Physical substep time

Kernels inside `step.fixed()` or `step.adaptive()` can declare read-only
framework time inputs in their `KernelSpec` parameters:

```python
from hydroforge.contracts import SUBSTEP_TICK_SECONDS, substep_field

clock_inputs = {
    "dt_ptr": substep_field("dt"),
    "outer_ticks_ptr": substep_field("outer_ticks"),
    "elapsed_ticks_ptr": substep_field("elapsed_ticks"),
    "dt_ticks_ptr": substep_field("dt_ticks"),
}
```

`dt` is the physics width in model precision. The three tick inputs are
`int64`: `outer_ticks` is the duration of previously completed outer steps,
`elapsed_ticks` is the current substep's start within its outer step, and
`dt_ticks` is its clock width. Multiply ticks by `SUBSTEP_TICK_SECONDS`
(`2**-30` seconds) when a calculation needs seconds. A model-specific
initial time offset can be added before conversion; do not accumulate a
second floating clock in the model.

A step has at most one sampled `step.fixed()` or `step.adaptive()` scope, which
feeds statistics. Components that subcycle at other rates over the same outer
interval run in further sequential scopes that never sample:

```python
for _ in step.fixed(count=atmosphere_steps, sampled=False):
    advance_atmosphere()
for _ in step.fixed(count=land_steps, sampled=False):
    advance_land()
for _ in step.fixed(count=ocean_steps):  # the sampled scope
    advance_ocean()
```

`span=(index, parts)` runs an unsampled scope over one sub-interval, so a step
can alternate components across coupling windows. Every window of one call site
shares one compiled program, and `outer_ticks` starts at the window. Statistics
still come from the one sampled scope: when every component subcycles inside
the windows, end the step with a sampled scope that publishes the diagnostics.

```python
for window in range(windows):
    for _ in step.fixed(count=atmosphere_steps, sampled=False, span=(window, windows)):
        advance_atmosphere()
    for _ in step.fixed(count=ocean_steps, sampled=False, span=(window, windows)):
        advance_ocean()
for _ in step.fixed(count=1):
    publish_diagnostics()
```

The outer clock accumulates successful step durations as integer
microseconds, with each cumulative endpoint rounded to the nearest tick.
Adjacent outer intervals therefore share exactly the same endpoint. Fixed
substeps partition those ticks to within one tick while retaining their
original physics width, `duration / count`, in model precision. Adaptive
substeps expose the controller's accepted ticks directly. These inputs work
without a calendar and retain stable device addresses during graph replay.
They reset to zero on a new runtime or `reset_clock()`; they are runtime
metadata, not checkpoint fields.

## Multiple GPUs

Use a model that supports spatial decomposition. Before constructing it,
initialize distributed execution in your driver script:

```python
from hydroforge.parallel import setup_distributed

context = setup_distributed(allowed_devices=("cuda",))
```

Pass `device=context.device` to the model instead of a fixed CUDA device.
Run the script with one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 run.py
```

All ranks must construct the same model configuration and execute the same
step and checkpoint calls. They can share the output directory. Close models
on every rank before destroying the process group:

```python
import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()
```

Supported modules, ensemble settings, and minimum partition sizes depend on
the model. To inspect the available devices and communication support:

```bash
torchrun --standalone --nproc-per-node=2 -m hydroforge.parallel.probe
```

## License

Apache 2.0
