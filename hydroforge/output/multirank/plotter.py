"""Visualization helpers for multi-rank statistics output."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import netCDF4 as nc
import numpy as np
from pydantic import BeforeValidator, Field, field_validator, model_validator

from hydroforge.contracts.errors import cleanup_on_exit
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.numeric import (
    finite_float64,
    positive_finite_float64,
)
from hydroforge.serialization.files import atomic_output_path

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from hydroforge.output.multirank.reader import MultiRankStatsReader


_FigureExtent = Annotated[
    float, BeforeValidator(partial(positive_finite_float64, label="figsize extent"))
]
_FigureSize = tuple[_FigureExtent, _FigureExtent]
_ColorLimit = Annotated[
    float, BeforeValidator(partial(finite_float64, label="color limit"))
]


class _PlotStyle(HydroForgeModel):
    vmin: _ColorLimit | None = None
    vmax: _ColorLimit | None = None
    cmap: str = Field(default="viridis", min_length=1)
    figsize: _FigureSize = (8.0, 6.0)
    auto_crop: bool = True
    crop_pad: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _validate_color_range(self):
        if self.vmin is not None and self.vmax is not None and self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        return self


class _SinglePlotRequest(_PlotStyle):
    t_index: int = 0
    level: int | None = None
    member: int = 0
    as_scatter_if_no_map: bool = True
    s: Annotated[
        float, BeforeValidator(partial(positive_finite_float64, label="s"))
    ] = 1.0


class _AnimationRequest(_PlotStyle):
    out_path: str | Path
    level: int | None = None
    member: int = 0
    fps: int = Field(default=10, gt=0)
    x_range: tuple[int, int] | None = None
    y_range: tuple[int, int] | None = None
    t_range: tuple[int, int] | None = None


class _SeriesPlotRequest(HydroForgeModel):
    member: int | Annotated[list[int], Field(min_length=1)] = 0
    figsize: _FigureSize = (12.0, 6.0)
    title: str | None = None
    labels: list[Annotated[str, Field(min_length=1)]] | None = None

    @field_validator("member")
    @classmethod
    def _validate_ensemble(cls, value):
        if isinstance(value, list) and len(set(value)) != len(value):
            raise ValueError("member list must not contain duplicates")
        return value

    @property
    def members(self) -> tuple[int, ...]:
        return (self.member,) if isinstance(self.member, int) else tuple(self.member)


@contextmanager
def _plot_axes(ax, figsize):
    """Keep successful figures; close only the figure owned by a failed plot."""
    import matplotlib.pyplot as plt

    if ax is not None:
        yield ax.figure, ax, False
        return
    figure, axes = plt.subplots(figsize=figsize)
    try:
        yield figure, axes, True
    except BaseException:
        with cleanup_on_exit("plot figure", (partial(plt.close, figure),)):
            raise


class MultiRankPlotter:
    """Explicit plotting service for one multi-rank reader."""

    def __init__(self, owner: MultiRankStatsReader) -> None:
        self.owner = owner

    @property
    def map_shape(self):
        return self.owner.map_shape

    @property
    def time_len(self):
        return self.owner._time_len

    @property
    def times(self):
        return self.owner._time_datetimes

    @property
    def var_name(self):
        return self.owner.var_name

    def get_grid(
        self,
        t_index: int,
        level: int | None = None,
        member: int = 0,
        fill_value: float = np.nan,
        dtype: Any = None,
    ) -> np.ndarray:
        return self.owner._get_grid(t_index, level, member, fill_value, dtype)

    def get_series(
        self,
        points: np.ndarray | Sequence[np.ndarray] | list[int],
        level: int | None = None,
        member: int = 0,
        fill_value: float = np.nan,
        dtype: Any = None,
        *,
        time_slice: slice | None = None,
    ) -> np.ndarray:
        del fill_value
        return self.owner.get_series(
            points,
            level,
            member,
            dtype,
            time_slice=time_slice,
        )

    @staticmethod
    def _strict_half_open_range(
        value: tuple[int, int] | None,
        *,
        length: int,
        label: str,
    ) -> tuple[int, int]:
        if length <= 0:
            raise ValueError(f"{label} cannot select from an empty timeline")
        if value is None:
            return 0, length
        start, end = value
        if start < 0 or end > length or start >= end:
            raise ValueError(
                f"{label} must satisfy 0 <= start < end <= {length}; got {value}"
            )
        return start, end

    @staticmethod
    def _strict_inclusive_range(
        value: tuple[int, int] | None,
        *,
        length: int,
        label: str,
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        start, end = value
        if start < 0 or end >= length or start > end:
            raise ValueError(
                f"{label} must satisfy 0 <= start <= end < {length}; got {value}"
            )
        return start, end

    def plot_single_time(
        self,
        t_index: int = 0,
        level: int | None = None,
        member: int = 0,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (8, 6),
        as_scatter_if_no_map: bool = True,
        s: float = 1.0,
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        request = _SinglePlotRequest(
            t_index=t_index,
            level=level,
            member=member,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            as_scatter_if_no_map=as_scatter_if_no_map,
            s=s,
            auto_crop=auto_crop,
            crop_pad=crop_pad,
        )
        t_index, level, member = request.t_index, request.level, request.member
        vmin, vmax, cmap = request.vmin, request.vmax, request.cmap
        figsize, s = request.figsize, request.s
        auto_crop, crop_pad = request.auto_crop, request.crop_pad
        as_scatter_if_no_map = request.as_scatter_if_no_map
        self.owner._data_access._make_row_request(
            time_index=t_index, level=level, member=member
        )

        t_str = f"t={t_index}"
        if len(self.times) > 0:
            t_str = self.owner._safe_time_str(self.times[t_index])

        # Check if we have members to display in title
        has_ensemble = self.owner._rank_files[0]["has_ensemble"]

        title_str = f"{self.var_name} @ {t_str}"
        if has_ensemble:
            title_str += f" (Member {member})"

        with _plot_axes(None, figsize) as (fig, ax, _created):
            if self.map_shape is not None:
                grid = self.get_grid(t_index, level=level, member=member)
                im = ax.imshow(grid.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title_str)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

                if auto_crop:
                    valid_mask = np.isfinite(grid)
                    if np.any(valid_mask):
                        xs, ys = np.where(valid_mask)
                        if len(xs) > 0:
                            xmin, xmax = xs.min(), xs.max()
                            ymin, ymax = ys.min(), ys.max()

                            # Apply padding
                            xmin = max(0, xmin - crop_pad)
                            xmax = min(grid.shape[0] - 1, xmax + crop_pad)
                            ymin = max(0, ymin - crop_pad)
                            ymax = min(grid.shape[1] - 1, ymax + crop_pad)

                            ax.set_xlim(xmin - 0.5, xmax + 0.5)
                            ax.set_ylim(ymax + 0.5, ymin - 0.5)

            elif as_scatter_if_no_map:
                x_all, y_all = self.owner.get_all_coords_xy()
                if x_all is None or y_all is None:
                    raise RuntimeError(
                        "map_shape not set and no converter-provided (x,y)."
                    )
                v_all = self.owner._get_vector(
                    t_index,
                    level=level,
                    member=member,
                )
                if not isinstance(v_all, np.ndarray) or v_all.ndim != 1:
                    raise ValueError(
                        "get_vector() must return a one-dimensional ndarray"
                    )
                if v_all.shape[0] != x_all.shape[0]:
                    raise ValueError("scatter coordinate and value counts do not match")
                sc = ax.scatter(
                    x_all, y_all, c=v_all, s=s, cmap=cmap, vmin=vmin, vmax=vmax
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{title_str} (scatter)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

                if auto_crop and len(x_all) > 0:
                    xmin, xmax = x_all.min(), x_all.max()
                    ymin, ymax = y_all.min(), y_all.max()

                    ax.set_xlim(xmin - crop_pad, xmax + crop_pad)
                    ax.set_ylim(ymax + crop_pad, ymin - crop_pad)

            else:
                raise RuntimeError(
                    "Cannot plot without map_shape when scatter plotting is disabled"
                )
            fig.tight_layout()

    def animate(
        self,
        out_path: str | Path,
        level: int | None = None,
        member: int = 0,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
        t_range: tuple[int, int] | None = None,
        fps: int = 10,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (8, 6),
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        request = _AnimationRequest(
            out_path=out_path,
            level=level,
            member=member,
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
            fps=fps,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            auto_crop=auto_crop,
            crop_pad=crop_pad,
        )
        if self.map_shape is None:
            raise RuntimeError("Animation requires map_shape.")
        out_path, level, member = Path(request.out_path), request.level, request.member
        x_range, y_range, t_range = request.x_range, request.y_range, request.t_range
        fps, vmin, vmax = request.fps, request.vmin, request.vmax
        cmap, figsize = request.cmap, request.figsize
        auto_crop, crop_pad = request.auto_crop, request.crop_pad

        t_start, t_end = self._strict_half_open_range(
            t_range,
            length=self.owner._time_len,
            label="t_range",
        )

        nx_, ny_ = self.map_shape
        strict_x = self._strict_inclusive_range(
            x_range,
            length=nx_,
            label="x_range",
        )
        strict_y = self._strict_inclusive_range(
            y_range,
            length=ny_,
            label="y_range",
        )

        xmin = 0
        xmax = nx_ - 1
        ymin = 0
        ymax = ny_ - 1

        grid_0 = self.get_grid(t_start, level=level, member=member)
        if auto_crop:
            crop_xmin, crop_xmax = nx_, -1
            crop_ymin, crop_ymax = ny_, -1
            for ti in range(t_start, t_end):
                grid = (
                    grid_0
                    if ti == t_start
                    else self.get_grid(
                        ti,
                        level=level,
                        member=member,
                    )
                )
                xs, ys = np.where(np.isfinite(grid))
                if xs.size:
                    crop_xmin = min(crop_xmin, int(xs.min()))
                    crop_xmax = max(crop_xmax, int(xs.max()))
                    crop_ymin = min(crop_ymin, int(ys.min()))
                    crop_ymax = max(crop_ymax, int(ys.max()))
            if crop_xmax >= crop_xmin:
                xmin = max(0, crop_xmin - crop_pad)
                xmax = min(nx_ - 1, crop_xmax + crop_pad)
                ymin = max(0, crop_ymin - crop_pad)
                ymax = min(ny_ - 1, crop_ymax + crop_pad)

        if strict_x is not None:
            xmin, xmax = strict_x
        if strict_y is not None:
            ymin, ymax = strict_y

        window = grid_0[xmin : xmax + 1, ymin : ymax + 1]
        if vmin is None or vmax is None:
            observed_min = np.inf
            observed_max = -np.inf
            for ti in range(t_start, t_end):
                grid = (
                    grid_0
                    if ti == t_start
                    else self.get_grid(
                        ti,
                        level=level,
                        member=member,
                    )
                )
                current = grid[xmin : xmax + 1, ymin : ymax + 1]
                finite = current[np.isfinite(current)]
                if finite.size:
                    observed_min = min(observed_min, float(finite.min()))
                    observed_max = max(observed_max, float(finite.max()))
            if vmin is None:
                vmin = 0.0 if observed_min == np.inf else observed_min
            if vmax is None:
                vmax = 1.0 if observed_max == -np.inf else observed_max
        if not (vmax > vmin):
            scale = max(abs(vmin), 1.0)
            expanded_max = vmin + scale * 1e-6
            if np.isfinite(expanded_max) and expanded_max > vmin:
                vmax = expanded_max
            else:
                expanded_min = vmin - scale * 1e-6
                if not np.isfinite(expanded_min) or expanded_min >= vmin:
                    raise ValueError(
                        "automatic animation color limits cannot be expanded"
                    )
                vmax = vmin
                vmin = expanded_min

        extent = (xmin - 0.5, xmax + 0.5, ymax + 0.5, ymin - 0.5)

        if out_path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "ffmpeg writer not found. Install ffmpeg or use .gif."
                )
            writer_type = animation.writers["ffmpeg"]
            writer = writer_type(
                fps=fps,
                metadata={"artist": "MultiRankStatsReader"},
            )

        fig, ax = plt.subplots(figsize=figsize)
        with cleanup_on_exit("animation figure", (partial(plt.close, fig),)):
            im = ax.imshow(
                window.T,
                origin="upper",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            t_label = f"t={t_start}"
            if len(self.times) > 0:
                t_label = self.owner._safe_time_str(self.times[t_start])

            ttl = ax.set_title(f"{self.var_name} @ {t_label}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.tight_layout()

            def _update(frame_idx: int):
                ti = t_start + frame_idx
                grid = self.get_grid(ti, level=level, member=member)
                win = grid[xmin : xmax + 1, ymin : ymax + 1]
                im.set_data(win.T)

                t_lbl = f"t={ti}"
                if len(self.times) > 0:
                    t_lbl = self.owner._safe_time_str(self.times[ti])

                ttl.set_text(f"{self.var_name} @ {t_lbl}")
                return [im, ttl]

            ani = animation.FuncAnimation(
                fig,
                _update,
                frames=t_end - t_start,
                interval=1000 / fps,
                blit=False,
            )
            with atomic_output_path(
                out_path,
                preserve_suffix=True,
            ) as temporary:
                ani.save(temporary, writer=writer)

    def plot_series(
        self,
        points: np.ndarray | Sequence[np.ndarray] | list[int],
        level: int | None = None,
        member: int | list[int] = 0,
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
        ax: plt.Axes | None = None,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Plot time series for specified points (IDs or XY coordinates).

        Args:
            points: One or more points. Can be a list of IDs/catchment_ids, or a list of (x,y) tuples.
            level: Level index if variable has levels.
            member: Single member index (int) or list of member indices.
            figsize: Figure size tuple (width, height) if creating new figure.
            title: Title of the plot.
            ax: Existing matplotlib axis to plot on.
            labels: Optional list of labels for the points (length must match number of points).
            **kwargs: Additional keyword arguments passed to ax.plot

        Returns:
            The matplotlib Axes object.
        """
        from matplotlib.ticker import FuncFormatter

        request = _SeriesPlotRequest(
            member=member, figsize=figsize, title=title, labels=labels
        )
        members, figsize = request.members, request.figsize
        title, labels = request.title, request.labels

        use_numeric_time = False
        if (
            self.owner._time_values_num is not None
            and self.owner._time_units is not None
            and self.owner._time_calendar is not None
        ):
            times_to_plot = self.owner._time_values_num
            use_numeric_time = True
        elif len(self.times) > 0:
            times_to_plot = self.times
        else:
            times_to_plot = np.arange(self.time_len)

        datasets = []
        expected_points = None
        for t in members:
            data = self.get_series(points, level=level, member=t)
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError("get_series() must return a two-dimensional ndarray")
            if data.shape[0] != len(times_to_plot):
                raise ValueError(
                    "series time dimension does not match the reader timeline"
                )
            num_points = data.shape[1]
            if expected_points is None:
                expected_points = num_points
            elif num_points != expected_points:
                raise ValueError("series point count differs between requested members")
            datasets.append((t, data))

        if labels is not None and len(labels) != expected_points:
            raise ValueError(
                f"labels length {len(labels)} does not match point count "
                f"{expected_points}"
            )
        if expected_points == 0:
            raise ValueError("points must select at least one series")

        with _plot_axes(ax, figsize) as (fig, ax, created_fig):
            for t, data in datasets:
                num_points = data.shape[1]
                for i in range(num_points):
                    # Construct label
                    # If multiple members, include member info. If multiple points, include point info.
                    lbl_parts = []

                    # Point Label
                    if labels is not None:
                        lbl_parts.append(labels[i])
                    else:
                        # Try to give a sensible default label from points
                        if isinstance(points, (list, tuple, np.ndarray)):
                            # If points passed as [1, 2], points[i] is 1
                            # If points passed as [[1,2], [3,4]], points[i] is [1,2]
                            if i < len(points):
                                pt_val = points[i]
                                lbl_parts.append(f"Pt {pt_val}")
                            else:
                                lbl_parts.append(f"Pt {i}")
                        else:
                            lbl_parts.append(f"Pt {i}")

                    # Member Label (only if ambiguous or multiple members)
                    if len(members) > 1:
                        lbl_parts.append(f"(Member {t})")
                    elif labels is None and num_points == 1:
                        lbl_parts.append(f"(Member {t})")

                    label_str = " ".join(lbl_parts)

                    ax.plot(times_to_plot, data[:, i], label=label_str, **kwargs)

            # Setup Axis Formatting
            if use_numeric_time:

                def time_tick_formatter(x, pos):
                    try:
                        # Use netcdf4 num2date to convert scalar to cftime/datetime object
                        # This works for ALL calendars (360_day, noleap, etc)
                        d = nc.num2date(
                            x,
                            units=self.owner._time_units,
                            calendar=self.owner._time_calendar,
                        )
                        return d.strftime("%Y-%m-%d")
                    except (TypeError, ValueError, OverflowError):
                        return f"{x:.1f}"

                ax.xaxis.set_major_formatter(FuncFormatter(time_tick_formatter))
                ax.set_xlabel(f"Time ({self.owner._time_calendar})")
            else:
                ax.set_xlabel("Time")

            ax.set_ylabel(self.var_name)

            if title is not None:
                ax.set_title(title)
            elif not ax.get_title():
                # Default title
                t_str = ""
                if len(times_to_plot) > 0:
                    if use_numeric_time:
                        try:
                            start_d = nc.num2date(
                                times_to_plot[0],
                                units=self.owner._time_units,
                                calendar=self.owner._time_calendar,
                            )
                            end_d = nc.num2date(
                                times_to_plot[-1],
                                units=self.owner._time_units,
                                calendar=self.owner._time_calendar,
                            )
                            t_str = f"{start_d.strftime('%Y-%m-%d')} - {end_d.strftime('%Y-%m-%d')}"
                        except (TypeError, ValueError, OverflowError):
                            pass
                    elif hasattr(times_to_plot[0], "date"):
                        t_str = (
                            f"{times_to_plot[0].date()} - {times_to_plot[-1].date()}"
                        )
                ax.set_title(f"{self.var_name} Time Series {t_str}")

            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3)

            # If we created the figure, layout tight
            if created_fig:
                fig.tight_layout()

            return ax
