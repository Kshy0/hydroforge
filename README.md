# HydroForge

A framework for GPU-accelerated hydrological modelling with Torch, Triton,
native CUDA, and Metal backends.

## Packages

| Package | Contents |
|---|---|
| `hydroforge.model` | `AbstractModel`, module declarations (`AbstractModule`, `TensorField`, `module_ref`, ...) and `OutputConfig` |
| `hydroforge.contracts` | Typed declarations used by models: `OptionsConfig`, conditions (`module`, `opt`), `option`, `SimulationSchedule`, `SpinupSchedule`, `StatisticsPlan` and its windows, output views (`GridAxis`, `OutputView`, `Range`, `Indices`, `Out`), `ParameterChange`, `BackendRequirement`, `ModuleRequirement`, step fields |
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
from hydroforge.contracts import (
    GridAxis,
    Out,
    OptionsConfig,
    OutputView,
    Range,
    SimulationSchedule,
    StatisticsPlan,
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

### Output axes and views

Every logical axis of an output has a name. The leading axis of a field
declaring `dim_coords` is its coordinate, whose rows are integer IDs; a
symbolic axis takes the name the model gives it in `grid_axes` (otherwise
the symbol itself); a literal axis is `dim_{position}`. A staggered axis
names the cell axis it bounds:

```python
class Atmosphere(AbstractModel):
    grid_axes = {
        "num_k": GridAxis(name="bottom_top"),
        "num_i": GridAxis(name="west_east"),
        "num_is": GridAxis(name="west_east_stag", staggers="west_east"),
    }
```

Outputs select by global identity, never by a rank's local position:
coordinate rows by ID through their `SelectionField`, grid axes through a
named `OutputView` of global logical indices (`Range`, spelled like
`range`, or strictly increasing `Indices`). A view applies to the axes a
variable has; a staggered axis it does not name keeps the faces bounding a
selected unit-step range (otherwise the same indices). One variable has one
view, so alias a field to write it under another:

```python
OutputConfig(
    views={"box": OutputView(bottom_top=Range(10), west_east=Range(100, 300, 2))},
    variables={
        "mean": ["t2", Out("u", view="box"), Out("u_col", expression="u", view="box")],
        "max3_mean": [Out("t2_box", expression="t2", view="box")],
    },
)
```

Fields and views of any rank support every statistic, including compound,
top-k and arg reductions. A regular range (or evenly spaced indices) is
addressed as `start + step * i`; only irregular `Indices` read a table. A
scatter (`scatter_sum`/`scatter_mean`) read only through one selection
aggregates just the selected targets and the sources reaching them; that
topology is rebuilt when a structural update edits the index or selection. Output dimensions take the axis names (a coordinate
axis is `saved_points`), and every subset grid axis is written with an int64
coordinate of the global indices it holds. `MultiRankStatsReader.get_series`
takes one `level` index per value axis after the saved points (an int, or a
tuple for several). Ranks write only the rows they own: halo ghost rows are
never part of an output.

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

### Splitting connected structure: halo exchange (CUDA)

Partition groups normally keep connected structure (for example a river
basin) on one rank. HydroForge checks this from the partition itself: every
reference to `partition_key` rows that does not define a coordinate's
ownership may cross ranks, and a group is closed when all of its rows'
targets stay on its rank. When every group is closed (always on one rank),
nothing below exists: no ghost rows, symmetric memory or exchange kernels.

Groups are placed greedily, largest first on the least loaded rank. When a
rank then exceeds the mean by more than `partition_imbalance` (default 5 %,
`None` keeps groups whole) and the model exchanges a halo, groups that do
not fit are cut along their references to the partition key: each is
spanned by its tree references (`downstream`), and pieces, sub-basins or
reaches between confluences, move to a rank, the ones crossing the fewest
references per row first, until it is full. On the Amazon at 03min
(1.5 million cells, 184 thousand bifurcations) 2 ranks cut 11 references
and 4 ranks 45 (METIS: 7 and 43). The partition ids are derived before any
input is sliced, alike on every rank; the `model.partition` event reports
the pieces, the imbalance and the references cut.

A group that is not closed (such as METIS parts written into the group field)
is accepted only when the model exchanges the halo of every open module,
declared as `module_requirements={"base": ModuleRequirement(halo=True), ...}`,
with one ensemble member on the `cuda` or `triton` backend; otherwise
construction names the crossing references and stops. Each rank then also
holds read-only ghost rows of the remote rows its rows reference, after its
owned rows: fields aligned to the coordinate are `owned + ghosts` long, while
size scalars return `owned_rows(coordinate)`, the extent kernels iterate.
Checkpoints and outputs keep owned rows only.

Inside a managed step, `model.halo_exchange_(*fields, mode=...)` moves up to
three fields: `"mirror"` copies owner values to ghosts, `"accumulate"` adds
ghost values to their owners and zeroes the ghosts (`clear=` zeroes one more
field there). It is a registered kernel, recorded and CUDA-graph capturable;
`model.halo_enabled` tells whether any group is split, and a rank without
messages in an exchange launches nothing.

The transport is chosen once, from facts every rank gathers alike (node,
WSL, CUDA driver version, RDMA devices, NVSHMEM): the fastest one all ranks
support, or the one `HYDROFORGE_HALO_TRANSPORT` names (`auto`, `peer`,
`nvshmem`, `nccl`); without any, construction lists why each is unusable.

| Transport | Moves rows by | Needs |
|---|---|---|
| `peer` | direct stores into peers' memory and a released arrival signal (CUDA C++ over CUDA IPC on `cuda`, NVSHMEM memory on `triton`) | one node outside WSL |
| `nvshmem` | NVSHMEM puts with signals (`triton`) | NVSHMEM and one CUDA driver version on every rank; across nodes a Mellanox InfiniBand/RoCE or EFA device on every node |
| `nccl` | packing, one NCCL send/recv group and unpacking, peer by peer | an NCCL process group |

Every transport is a registered kernel, CUDA-graph capturable. Buffers that
peers write use the `remote` kernel access mode, which capture rollbacks
never restore. `model.halo` events report the transport and why faster ones
were skipped.

### All-reductions

`all_reduce_` of model tensors on CUDA ranks, recorded inside a managed
step, runs as a registered kernel rather than a process-group collective
behind a per-launch handshake, so a substep loop that reduces (an adaptive
sub-step count) stays CUDA-graph captured. Its transport is chosen like the
halo's, once per process group from the gathered rank facts, or named by
`HYDROFORGE_REDUCE_TRANSPORT` (`auto`, `peer`, `nvshmem`, `nccl`,
`process_group`):

| Transport | Reduces by | Serves | Needs |
|---|---|---|---|
| `peer` | stores into every rank's slots, a released signal, rank-order reduction (CUDA C++ on `cuda`, Triton on `triton`) | min, max, sum of ≤ 64 values | one node outside WSL |
| `nvshmem` | the same with NVSHMEM puts and signals (`triton`) | min, max, sum of ≤ 64 values | as the halo's `nvshmem` |
| `nccl` | NCCL all-reduce | any | an NCCL process group |

`peer` and `nvshmem` give every rank the same bits. Reductions to one
destination, temporaries and other devices keep the process-group path; a
`model.reduce` event reports the transports and why others were skipped.

## License

Apache 2.0
