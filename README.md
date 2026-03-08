# Hydroforge

Generic framework for GPU-accelerated hydrological modelling.

## Packages

| Package | Contents |
|---|---|
| `hydroforge.modeling` | `AbstractModule`, `AbstractModel`, `InputProxy`, distributed utilities, model utilities |
| `hydroforge.io.datasets` | `AbstractDataset`, `MixedDataset`, `StaticParameterDataset`, `DailyBinDataset`, `NetCDFDataset`, `ERA5LandAccumDataset`, `ExportedDataset` |
| `hydroforge.aggregator` | `StatisticsAggregator`, streaming NetCDF output, kernel codegen |
| `hydroforge.runtime` | Kernel backend selection (`cuda` / `metal` / `triton` / `torch`) |

## Installation

### 1. Install PyTorch manually

`torch` is not listed as a dependency and must be installed before this package. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your environment. For example, with CUDA 13.0:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> `triton` ships automatically with PyTorch on supported systems and does not need to be installed separately.

### 2. Install Hydroforge

From source:

```bash
pip install git+https://github.com/Kshy0/hydroforge.git
```

Or in editable mode (for development):

```bash
git clone https://github.com/Kshy0/hydroforge.git
cd hydroforge
pip install -e .
```

## Backend Selection

Set `HYDROFORGE_BACKEND` to choose the kernel backend (default: `triton`):

```bash
export HYDROFORGE_BACKEND=triton   # default
export HYDROFORGE_BACKEND=torch    # pure-PyTorch fallback
```

## Usage

```python
# Core abstractions
from hydroforge.modeling.module import AbstractModule, TensorField, computed_tensor_field
from hydroforge.modeling.model import AbstractModel
from hydroforge.modeling.input_proxy import InputProxy

# Dataset utilities
from hydroforge.io.datasets import (
    AbstractDataset,
    DailyBinDataset,
    NetCDFDataset,
    ERA5LandAccumDataset,
    ExportedDataset,
    MixedDataset,
    StaticParameterDataset,
)

# Statistics aggregation
from hydroforge.aggregator.aggregator import StatisticsAggregator

# Backend selection
from hydroforge.runtime.backend import KERNEL_BACKEND
```

## License

Apache 2.0
