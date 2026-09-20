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
when any listed consumer is open. Among state categories, only `init_state`
tensors are checkpointed; reconstruction also retains parameters and topology.
Distributed checkpoints include each partitioned coordinate and permit unequal
or empty local partitions, while non-partition dimensions must match across
ranks. Ensemble checkpoint saving remains unsupported.

Scheduled `ParameterChange` arrays are validated against the actual prepared,
rank-local parameter shape, including an explicit member axis when present.
Validation uses the model's `shard_param()` and the module's
`prepare_module_input()` hooks, not the raw input-file dimensions. These hooks
must perform data preparation without depending on a materialized backend or
execution runtime: parameter schedules can invoke them during model
construction. Their prepared inputs and needed default-factory results are
reused for module initialization, without allocating unrelated default state.
Target IDs retain global membership validation and follow the prepared local
coordinate order. A failed initialization discards prepared inputs; a retry
prepares and validates fresh inputs rather than reusing mutated tensors.

Conditional tensor storage follows three rules:

- `depends_on`: every named module must be open.
- `required_by`: at least one named consumer must be open or request output.
- `output_only`: storage exists only when the field is requested directly.

Inactive computed tensors are exposed as `None`. Virtual expression fields are
symbolic and do not allocate storage.

`variables_to_save` directly requests its declared fields. Alias expressions
activate their dependencies without forcing the alias-named tensor to exist.
For a conditional field with a same-named alias, the tensor is used when active and the alias expression
is used when inactive. Use `materialized_outputs=("module.field", ...)` when a
field must be resident without being written by the statistics system.

Ad-hoc statistics expressions retain the inferred numeric field precision and
output coordinate/selection, including `hpfloat` in mixed-precision models.
Boolean conditions do not replace the numeric reference field's precision.
Shared parameter statistics retain their unbatched shape in multi-member models;
indexed level outputs are selected along the spatial axis and sampled once, not
once per member. Member-batched state statistics keep their separate member axis.
Expressions combining shared parameters with member-batched fields broadcast the
parameters without expanding their storage, including nested aliases and selected
level outputs. Direct and intermediate scatter buffers retain the expression's
actual member layout rather than inheriting the model's global member count.
Numeric literal validation also applies inside `scatter_sum` and `scatter_mean`:
nonzero literals that underflow Python float64 or the output dtype are rejected,
while explicit zero remains valid. Multiline scatter values retain their literal
spelling and matching AST source positions.

`ReferenceIndexField("reference_id")` is aligned with the referencing rows;
`inverse=True` aligns it with the target coordinate, leaving unmatched rows at
`-1`. Both its symbolic shape and coordinate metadata follow that axis, including
qualified cross-module dimensions. Derived indices follow the source field's
activation: inactive sources have no index storage or active namespace entry.
Statistics dependencies on an index also request its source and target fields;
output demand never overrides a closed `depends_on` module.

`ModuleType.get_tensor_schema("index", opened_modules=(...))` resolves optional
target modules. Alongside `opened_modules`, pass a `FieldDemandPlan` as
`field_demand` to restrict derived
metadata to a particular output specialization. Without that plan, schema
queries retain output-activatable candidates; an ambiguous target is rejected
rather than guessed. Output discovery does not resolve unused inactive indices.

Model-wide options and constants are immutable, nested configuration objects
rather than module fields or ad-hoc integer flags:

```python
from enum import StrEnum

from hydroforge.contracts import OptionField, OptionsConfig, option_code

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

HydroForge automatically adds modules required by selected options, validates
backend availability declared on `OptionField`, and checks forcing combinations
declared with `ForcingOptionField` at `@between_steps` boundaries.

Kernels declare semantic compile-time bindings with `compile_time_sources`.
Use `option_code("routing")` for a stable device integer,
`config_value("path.to.scalar")` for a scalar option or constant, and
`output_requested(module, field)` for output-dependent specialization. These
values are fixed for the compiled model specialization across all backends.

For Triton, a compile-time value declared with kind `precision` is lowered to
a typed `tl.float32` or `tl.float64` scalar so Python `float` defaults cannot
silently force fp32. Integer and boolean compile-time values remain
`tl.constexpr`. Compound Triton programs must launch inner kernels through
`hydroforge.kernels.launch_triton_kernel`; the program dispatcher exposes the
resolved model precision while preparing and replaying the program.

Native CUDA operator recording batches the validated specializations actually
used by a scope and its nested predicate bodies. It builds them through the
shared process pool, then publishes executable launch closures without changing
physical execution order. `HYDROFORGE_PRECOMPILE_JOBS` controls build processes;
`MAX_JOBS` controls Ninja jobs inside each process. Direct specialization outside
operator recording remains lazy and synchronous.

`make_spec_cuda_dispatcher(source=Path(...), include_root=Path(...))` follows
quoted local headers within that root. Only reachable source contents enter the
cache identity. Explicit `CudaExtensionSpec.inline_includes` remain supported.
Spec templates expose typed constants and `hydroforge::KernelParameters` aliases
for device functions with explicit compile-time configuration parameters; they
do not translate model physics.

A model defines forcing staging and physical execution directly:

Both `@between_steps` and `@managed_step` require synchronous, non-generator
functions. Coroutine, generator, and async-generator implementations are
rejected at declaration, including those in a `functools.wraps` chain, because
deferred bodies would run outside the runtime's transaction and failure guards.

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

Compiled scope iterators must be consumed normally: do not use Python `break`
or `return` to leave a `step.fixed()`, `step.adaptive()`, or nested
`step.predicate()` body. Incomplete predicate recording is rejected before
the fixed parent can execute or cache a body with the child loop missing.
Use the predicate frame's device `continue_flag` for runtime termination.
Explicit host `specialization` values retain exact scalar types and finite
float identity, including the distinction between `+0.0` and `-0.0`.

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

Spatial mapping keeps projected coordinates in their declared units; longitude
wrapping applies only to geographic grids. Geographic latitude centers must lie
within `[-90, 90]`. Inferred cell bounds, like explicit bounds, must have positive
representable widths and contain their centers; valid polar centers remain
supported with bounds clipped at the poles.

`MappingTable` rejects missing target IDs, coordinates, coverage and sparse
components instead of discarding their masks. Direct and archived CSR inputs
are structurally checked before native sparse operations, and canonical storage
is verified independently of caller-cached flags. `LocalMapping` still coalesces
valid duplicate entries. Source-mask `nearest` repair chooses a minimum Euclidean
distance in grid-index space, including the periodic longitude distance where
applicable, rather than stopping at the first nonempty square search ring.

The public daily, monthly, yearly and single-file time-key helpers require
`datetime` or `cftime.datetime` values. They preserve the supplied calendar;
date-only values, strings and date-like objects are not accepted implicitly.

`MultiRankStatsReader` validates one captured set of rank files. Files added
during or after construction are not incorporated into that reader; construct
a new reader to include them. Rank identities must be complete and unique,
including each rank/year pair for split output. Only the common committed
time prefix is exposed, including for uneven or empty spatial partitions.
Point queries reject missing IDs/XY coordinates rather than discarding their
mask. Real-valued grid fills retain overflow/underflow checks for NumPy extended
precision, while explicit NaN and infinite fills remain supported.

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

Ensemble-enabled models can additionally pass an active
`hydroforge.data.EnsembleParallel(ensemble_size=10, ensemble_partitions=2,
spatial_partitions=2)` context as `parallel=...`, with the same global
`ensemble_size=10` in the model constructor. Launch four processes for this
example. Input arrays use global member axes; modules, forcing buffers and kernels
use local member slices. `parallel=None` preserves the original spatial layout.
The model retains global `rank/world_size` for control-plane coordination and
exposes `spatial_rank/spatial_world_size` for domain ownership.

Framework collectives default to `scope="spatial"`; use `scope="ensemble"` only
for intentional communication between member groups, such as a shared CFL limit.
Close models before leaving the mesh context. Subgroup output directories contain
an `ensemble` coordinate with global member IDs; existing readers operate on one
such directory at a time. See [implementation and local validation boundaries](docs/ensemble-parallel-2026-09-11.md).

## Model and statistics clocks

File-backed datasets capture source identity before inspecting storage and verify
it throughout schema discovery and streamed reads. Replacing a file during
construction requires constructing a new dataset; an existing schema is never
silently rebound to the replacement. Grouped `DailyBinDataset` files must contain
every requested absolute frame, including spin-up support, even for an empty
spatial selection. `ExportedDataset` accepts explicit `window_length` and
`window_starts` together; starts must be a nonempty, strictly increasing integer
vector whose windows fit the main source axis.

`build_local_mapping()` materializes the device matrix before publishing its
selection and rolls back selection/read-plan metadata for every participating
source if installation fails. Multi-variable exports require distinct output
filenames; read/progress failures still close writers and discard their temporary
files. Explicit NetCDF input alignment indices must be a complete permutation
without missing positions, not merely a vector with the expected length.

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

Output workers start asynchronously during statistics runtime installation,
overlapping initialization with model work. Their readiness tasks count as no
output rows or payload bytes; existing checks observe failures, and close still
waits for owned work. The default remains two workers; `output_workers=0` writes
synchronously. See [startup controls and tradeoffs](docs/early-output-workers-2026-09-14.md).

Models select variables and aggregation operations with `variables_to_save`.
Supported reductions are `mean`, `sum`, `max`, `min`, `first`, and `last`.
For a declared tensor, `output="auto"` applies the matching `SelectionField`
when one exists and otherwise saves the full logical domain; `output="full"`
always saves the full domain, while `output="disabled"` rejects direct output.
Output policy controls serialization; field dependencies and output demand
control storage allocation.

For declared `scatter_sum` and `scatter_mean` statistics, a destination outside
`[0, target_extent)` contributes nothing, including the `-1` sentinel produced
by an inverse reference index. Such contributors do not affect sums or mean
counts even if their values are NaN or infinite. A destination with no valid
contributors has sum zero and mean NaN. This rule also applies when indices
change between captured executions and when the output domain is empty.

The default NetCDF profile is lossless Blosc-Zstd level 5 with byte shuffle:

```python
output_netcdf_options={
    "compression": "blosc_zstd",
    "complevel": 5,
    "blosc_shuffle": 1,
}
```

HydroForge falls back to zlib level 4 when Blosc is unavailable, a chunk is too
small, or an unlimited-axis variable leaves its chunk layout to NetCDF's
implicit heuristics. Explicit safe chunk sizes retain the preferred compressor.
Set zlib explicitly for files that must be readable without the Blosc plugin.
Quantization is not enabled by default.

HydroForge detects the installed `netCDF4` distribution from its installer
metadata. Pip-installed wheels retain their bundled HDF5 filter plugins; only
Conda-installed distributions use standard Conda HDF5 plugin directories.

When `chunksizes` is omitted, streaming output chooses an approximately 4 MiB
layout aligned with the write batch. Dataset export methods use the same
`netcdf_options` mapping.

Multi-rank model output can be read with:

```python
from hydroforge.output.multirank import MultiRankStatsReader
```

Invalid multi-rank query arguments are rejected before loading an optional data
cache. Export methods validate NetCDF options before preparing output, and retain
both the original operation error and any cleanup errors. A failed yearly export
aborts its current temporary file; previously committed years remain published.

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

## 0.3.0 migration

Python 3.11+ is required. See [the migration guide](docs/migration-0.3.0.md) for native Torch forcing-copy semantics, lifecycle/compiler ownership, backend validation limits, and the coordinated eight-model migration.

Calendar-dependent kernels can now declare read-only device scalars with
`KernelSpec.step_fields`; see [step-field bindings](docs/step-fields.md) for
compiled GPU calendar advancement, fused custom expressions, and graph-replay lifetime rules. Builtins require no Python provider or per-step value upload.

## Code review

The [file-by-file review](docs/simplicity-review-2026-09-11.md) records the
validation boundaries, simplifications, regression results, and platform limits
for the current implementation. Earlier reports retain their historical results.
