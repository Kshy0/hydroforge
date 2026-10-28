# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""What each rank can use to reach the others, as the halo decides it.

Run under the launcher of the job to check a machine before a model does::

    torchrun --nproc-per-node=4 -m hydroforge.parallel.probe

Every rank reports its node, GPU, driver, RDMA devices and whether GPUDirect
RDMA is loaded, the NCCL torch uses and the NCCL HydroForge loads for peer
memory (``HYDROFORGE_NCCL``), what that NCCL reports (device API, multimem,
GPU-initiated networking), and which peers it can load from and store to
through an NCCL window. Nothing here depends on a known machine: NCCL itself
decides which peers are reachable (NVLink, PCIe peer-to-peer, multi-node
NVLink) and which network carries the rest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _gpudirect_rdma() -> str | None:
    """The kernel module that lets NICs reach GPU memory, if loaded."""

    modules = Path("/proc/modules")
    if not modules.is_file():
        return None
    loaded = {line.split()[0] for line in modules.read_text().splitlines()}
    for name in ("nvidia_peermem", "nv_peer_mem"):
        if name in loaded:
            return name
    return None


def _rdma_devices() -> list[str]:
    root = Path("/sys/class/infiniband")
    if not root.is_dir():
        return []
    found = []
    for device in sorted(root.iterdir()):
        driver = device / "device" / "driver"
        found.append(
            f"{device.name}:{driver.resolve().name if driver.exists() else '?'}"
        )
    return found


def local_facts(device: torch.device) -> dict[str, Any]:
    """This rank's node, GPU, driver and NCCL, without communication."""

    from hydroforge.parallel.nccl import nccl_library
    from hydroforge.parallel.symmetric import _driver_version
    from hydroforge.parallel.topology import node_label

    library, problem = nccl_library()
    properties = torch.cuda.get_device_properties(device)
    torch_nccl = torch.cuda.nccl.version() if torch.cuda.is_available() else None
    return {
        "node": node_label()[:12],
        "host": os.uname().nodename,
        "gpu": properties.name,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
        "driver": _driver_version(),
        "wsl": "microsoft" in os.uname().release.lower(),
        "rdma": _rdma_devices(),
        "gpudirect_rdma": _gpudirect_rdma(),
        "torch_nccl": None if torch_nccl is None else ".".join(map(str, torch_nccl)),
        "nccl": None if library is None else library.version_text,
        "nccl_path": None if library is None else library.path,
        "nccl_problem": problem,
    }


def probe(group: Any = None) -> list[dict[str, Any]]:
    """Every rank's facts and reachable peers (collective)."""

    import torch.distributed as dist

    from hydroforge.parallel.nccl import PeerWindow, nccl_library
    from hydroforge.parallel.topology import node_label

    if group is None:
        group = dist.group.WORLD
    device = torch.device("cuda", torch.cuda.current_device())
    facts = local_facts(device)
    library, _problem = nccl_library()
    world = dist.get_world_size(group)
    usable: list[Any] = [None] * world
    dist.all_gather_object(usable, library is not None, group=group)
    if all(usable):
        nodes: list[Any] = [None] * world
        dist.all_gather_object(nodes, node_label(), group=group)
        window = PeerWindow(library, group, device, 4096, nodes)
        facts.update(window.properties)
        if window.node_properties:
            facts["node"] = window.node_properties
        facts["window_scope"] = window.scope
        facts["load_store_peers"] = [
            peer for peer, address in enumerate(window.peers) if address
        ]
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        window.close()
    else:
        facts["load_store_peers"] = None
    everyone: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(everyone, facts, group=group)
    return everyone


def main() -> None:
    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    report = probe()
    if dist.get_rank() == 0:
        for rank, facts in enumerate(report):
            print(json.dumps({"rank": rank, **facts}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
