"""Shared limits for compiling kernel specializations concurrently."""

from __future__ import annotations

import os

from hydroforge.data.distributed import get_local_process_rank


def precompile_jobs(count: int, default_jobs: int) -> int:
    configured_jobs = os.environ.get("HYDROFORGE_PRECOMPILE_JOBS")
    if configured_jobs is None:
        jobs = 1 if get_local_process_rank() != 0 else default_jobs
    else:
        try:
            jobs = int(configured_jobs)
        except ValueError as error:
            raise ValueError(
                "HYDROFORGE_PRECOMPILE_JOBS must be a positive integer, "
                f"got {configured_jobs!r}"
            ) from error
    if type(jobs) is not int or jobs < 1:
        name = (
            "default_jobs" if configured_jobs is None else "HYDROFORGE_PRECOMPILE_JOBS"
        )
        raise ValueError(f"{name} must be a positive integer, got {jobs!r}")
    return min(jobs, count)
