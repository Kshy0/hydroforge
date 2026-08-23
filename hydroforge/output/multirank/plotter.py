"""Visualization helpers for multi-rank statistics output."""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import netCDF4 as nc
import numpy as np

from hydroforge.data.numeric import (
    finite_float64,
    positive_finite_float64,
)
from hydroforge.serialization.files import atomic_output_path

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from hydroforge.output.multirank.reader import MultiRankStatsReader


class MultiRankPlotter:
    """Explicit plotting service for one multi-rank reader."""

    def __init__(self, owner: MultiRankStatsReader) -> None:
        self.owner = owner

    @property
    def _map_shape(self):
        return self.owner.map_shape

    @property
    def _rank_files(self):
        return self.owner._rank_files

    @property
    def _t_indices(self):
        return self.owner._t_indices

    @property
    def _time_calendar(self):
        return self.owner._time_calendar

    @property
    def _time_len(self):
        return self.owner._time_len

    @property
    def _time_units(self):
        return self.owner._time_units

    @property
    def _time_values_num(self):
        return self.owner._time_values_num

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

    def _safe_time_str(self, value):
        return self.owner._safe_time_str(value)

    def get_grid(
        self, t_index: int, level: int | None = None, trial: int = 0,
        fill_value: float = np.nan, dtype: Any = None,
    ) -> np.ndarray:
        return self.owner._get_grid(t_index, level, trial, fill_value, dtype)

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray], List[int]],
        level: int | None = None,
        trial: int = 0,
        fill_value: float = np.nan,
        dtype: Any = None,
        *, time_slice: slice | None = None,
    ) -> np.ndarray:
        del fill_value
        return self.owner.get_series(
            points, level, trial, dtype, time_slice=time_slice,
        )

    @staticmethod
    def _strict_half_open_range(
        value: tuple[int, int] | None, *, length: int, label: str,
    ) -> tuple[int, int]:
        if length <= 0:
            raise ValueError(f"{label} cannot select from an empty timeline")
        if value is None:
            return 0, length
        if type(value) is not tuple or len(value) != 2:
            raise TypeError(f"{label} must be an exact (start, end) tuple")
        start, end = value
        if type(start) is not int or type(end) is not int:
            raise TypeError(f"{label} bounds must be exact ints")
        if start < 0 or end > length or start >= end:
            raise ValueError(
                f"{label} must satisfy 0 <= start < end <= {length}; "
                f"got {value}"
            )
        return start, end

    @staticmethod
    def _strict_inclusive_range(
        value: tuple[int, int] | None, *, length: int, label: str,
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        if type(value) is not tuple or len(value) != 2:
            raise TypeError(f"{label} must be an exact (start, end) tuple")
        start, end = value
        if type(start) is not int or type(end) is not int:
            raise TypeError(f"{label} bounds must be exact ints")
        if start < 0 or end >= length or start > end:
            raise ValueError(
                f"{label} must satisfy 0 <= start <= end < {length}; "
                f"got {value}"
            )
        return start, end

    @staticmethod
    def _validate_figsize(value: tuple[Real, Real]) -> tuple[float, float]:
        if type(value) is not tuple or len(value) != 2:
            raise TypeError("figsize must be an exact two-element tuple")
        width = positive_finite_float64(
            value[0], label="figsize width",
        )
        height = positive_finite_float64(
            value[1], label="figsize height",
        )
        return width, height

    @staticmethod
    def _validate_color_limit(value: Real | None, *, label: str) -> float | None:
        if value is None:
            return None
        return finite_float64(value, label=label)

    @staticmethod
    def _validate_crop(auto_crop: bool, crop_pad: int) -> None:
        if type(auto_crop) is not bool:
            raise TypeError("auto_crop must be an exact bool")
        if type(crop_pad) is not int:
            raise TypeError("crop_pad must be an exact int")
        if crop_pad < 0:
            raise ValueError("crop_pad must be non-negative")

    @staticmethod
    def _validate_cmap(cmap: str) -> None:
        if type(cmap) is not str or not cmap:
            raise TypeError("cmap must be a non-empty exact str")

    def plot_single_time(
        self,
        t_index: int = 0,
        level: Optional[int] = None,
        trial: int = 0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        as_scatter_if_no_map: bool = True,
        s: float = 1.0,
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        import matplotlib.pyplot as plt

        if type(t_index) is not int:
            raise TypeError("t_index must be an exact int")
        if not 0 <= t_index < self._time_len:
            raise IndexError(
                f"t_index out of range [0, {self._time_len - 1}]"
            )
        if type(as_scatter_if_no_map) is not bool:
            raise TypeError("as_scatter_if_no_map must be an exact bool")
        s = positive_finite_float64(s, label="s")
        self._validate_crop(auto_crop, crop_pad)
        self._validate_cmap(cmap)
        figsize = self._validate_figsize(figsize)
        vmin = self._validate_color_limit(vmin, label="vmin")
        vmax = self._validate_color_limit(vmax, label="vmax")
        if vmin is not None and vmax is not None and vmax <= vmin:
            raise ValueError("vmax must be greater than vmin")

        t_str = f"t={t_index}"
        if len(self.times) > 0:
            t_str = self._safe_time_str(self.times[t_index])

        # Check if we have trials to display in title
        has_trials = False
        if self._rank_files and self._rank_files[0]["has_trials"]:
            has_trials = True

        title_str = f"{self.var_name} @ {t_str}"
        if has_trials:
            title_str += f" (Trial {trial})"

        fig, ax = plt.subplots(figsize=figsize)
        if self.map_shape is not None:
            grid = self.get_grid(t_index, level=level, trial=trial)
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
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for info in self._rank_files:
                if info["saved_points"] == 0:
                    continue
                if info["x"] is None or info["y"] is None:
                    raise RuntimeError("map_shape not set and no converter-provided (x,y).")
                xs.append(info["x"])
                ys.append(info["y"])
            x_all = np.concatenate(xs) if xs else np.array([])
            y_all = np.concatenate(ys) if ys else np.array([])
            v_all = self.owner._get_vector(
                t_index, level=level, trial=trial,
            )
            if not isinstance(v_all, np.ndarray) or v_all.ndim != 1:
                raise ValueError("get_vector() must return a one-dimensional ndarray")
            if v_all.shape[0] != x_all.shape[0]:
                raise ValueError(
                    "scatter coordinate and value counts do not match"
                )
            sc = ax.scatter(x_all, y_all, c=v_all, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
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
        out_path: Union[str, Path],
        level: Optional[int] = None,
        trial: int = 0,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if self._map_shape is None:
            raise RuntimeError("Animation requires map_shape.")
        if type(fps) is not int:
            raise TypeError("fps must be an exact int")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._validate_crop(auto_crop, crop_pad)
        self._validate_cmap(cmap)
        figsize = self._validate_figsize(figsize)
        vmin = self._validate_color_limit(vmin, label="vmin")
        vmax = self._validate_color_limit(vmax, label="vmax")
        if vmin is not None and vmax is not None and vmax <= vmin:
            raise ValueError("vmax must be greater than vmin")

        out_path = Path(out_path)

        t_start, t_end = self._strict_half_open_range(
            t_range, length=self._time_len, label="t_range",
        )

        nx_, ny_ = self._map_shape
        strict_x = self._strict_inclusive_range(
            x_range, length=nx_, label="x_range",
        )
        strict_y = self._strict_inclusive_range(
            y_range, length=ny_, label="y_range",
        )

        xmin = 0
        xmax = nx_ - 1
        ymin = 0
        ymax = ny_ - 1

        grid_0 = self.get_grid(t_start, level=level, trial=trial)
        if auto_crop:
            crop_xmin, crop_xmax = nx_, -1
            crop_ymin, crop_ymax = ny_, -1
            for ti in range(t_start, t_end):
                grid = grid_0 if ti == t_start else self.get_grid(
                    ti, level=level, trial=trial,
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

        window = grid_0[xmin:xmax + 1, ymin:ymax + 1]
        if vmin is None or vmax is None:
            observed_min = np.inf
            observed_max = -np.inf
            for ti in range(t_start, t_end):
                grid = grid_0 if ti == t_start else self.get_grid(
                    ti, level=level, trial=trial,
                )
                current = grid[xmin:xmax + 1, ymin:ymax + 1]
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
                fps=fps, metadata={"artist": "MultiRankStatsReader"},
            )

        fig, ax = plt.subplots(figsize=figsize)
        try:
            im = ax.imshow(
                window.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=extent,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            t_label = f"t={t_start}"
            if len(self.times) > 0:
                t_label = self._safe_time_str(self.times[t_start])

            ttl = ax.set_title(f"{self.var_name} @ {t_label}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.tight_layout()

            def _update(frame_idx: int):
                ti = t_start + frame_idx
                grid = self.get_grid(ti, level=level, trial=trial)
                win = grid[xmin:xmax + 1, ymin:ymax + 1]
                im.set_data(win.T)

                t_lbl = f"t={ti}"
                if len(self.times) > 0:
                    t_lbl = self._safe_time_str(self.times[ti])

                ttl.set_text(f"{self.var_name} @ {t_lbl}")
                return [im, ttl]

            ani = animation.FuncAnimation(
                fig, _update, frames=t_end - t_start,
                interval=1000 / fps, blit=False,
            )
            with atomic_output_path(
                out_path, preserve_suffix=True,
            ) as temporary:
                ani.save(temporary, writer=writer)
        finally:
            plt.close(fig)

    def plot_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray], List[int]],
        level: Optional[int] = None,
        trial: Union[int, List[int]] = 0,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Plot time series for specified points (IDs or XY coordinates).

        Args:
            points: One or more points. Can be a list of IDs/catchment_ids, or a list of (x,y) tuples.
            level: Level index if variable has levels.
            trial: Single trial index (int) or list of trial indices.
            figsize: Figure size tuple (width, height) if creating new figure.
            title: Title of the plot.
            ax: Existing matplotlib axis to plot on.
            labels: Optional list of labels for the points (length must match number of points).
            **kwargs: Additional keyword arguments passed to ax.plot

        Returns:
            The matplotlib Axes object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        if type(trial) is int:
            trials = (trial,)
        else:
            if type(trial) is not list or not trial:
                raise TypeError(
                    "trial must be an exact int or a non-empty list of ints"
                )
            if any(type(value) is not int for value in trial):
                raise TypeError("trial list entries must be exact ints")
            if len(set(trial)) != len(trial):
                raise ValueError("trial list must not contain duplicates")
            trials = tuple(trial)
        figsize = self._validate_figsize(figsize)
        if title is not None and type(title) is not str:
            raise TypeError("title must be an exact str or None")
        if labels is not None:
            if type(labels) is not list:
                raise TypeError("labels must be a list of strings or None")
            if any(type(label) is not str or not label for label in labels):
                raise ValueError("labels must contain non-empty exact strings")

        use_numeric_time = False
        if (
            self._time_values_num is not None
            and self._time_units is not None
            and self._time_calendar is not None
        ):
            times_to_plot = self._time_values_num
            use_numeric_time = True
        elif len(self.times) > 0:
            times_to_plot = self.times
        else:
            times_to_plot = np.arange(self.time_len)

        datasets = []
        expected_points = None
        for t in trials:
            data = self.get_series(points, level=level, trial=t)
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError(
                    "get_series() must return a two-dimensional ndarray"
                )
            if data.shape[0] != len(times_to_plot):
                raise ValueError(
                    "series time dimension does not match the reader timeline"
                )
            num_points = data.shape[1]
            if expected_points is None:
                expected_points = num_points
            elif num_points != expected_points:
                raise ValueError(
                    "series point count differs between requested trials"
                )
            datasets.append((t, data))

        if labels is not None and len(labels) != expected_points:
            raise ValueError(
                f"labels length {len(labels)} does not match point count "
                f"{expected_points}"
            )
        if expected_points == 0:
            raise ValueError("points must select at least one series")

        created_fig = False
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        for t, data in datasets:
            num_points = data.shape[1]
            for i in range(num_points):
                # Construct label
                # If multiple trials, include trial info. If multiple points, include point info.
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

                # Trial Label (only if ambiguous or multiple trials)
                if len(trials) > 1:
                    lbl_parts.append(f"(Trial {t})")
                elif labels is None and num_points == 1:
                    lbl_parts.append(f"(Trial {t})")

                label_str = " ".join(lbl_parts)

                ax.plot(times_to_plot, data[:, i], label=label_str, **kwargs)

        # Setup Axis Formatting
        if use_numeric_time:
            def time_tick_formatter(x, pos):
                try:
                    # Use netcdf4 num2date to convert scalar to cftime/datetime object
                    # This works for ALL calendars (360_day, noleap, etc)
                    d = nc.num2date(x, units=self._time_units, calendar=self._time_calendar)
                    return d.strftime('%Y-%m-%d')
                except (TypeError, ValueError, OverflowError):
                    return f"{x:.1f}"

            ax.xaxis.set_major_formatter(FuncFormatter(time_tick_formatter))
            ax.set_xlabel(f"Time ({self._time_calendar})")
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
                        start_d = nc.num2date(times_to_plot[0], units=self._time_units, calendar=self._time_calendar)
                        end_d = nc.num2date(times_to_plot[-1], units=self._time_units, calendar=self._time_calendar)
                        t_str = f"{start_d.strftime('%Y-%m-%d')} - {end_d.strftime('%Y-%m-%d')}"
                     except (TypeError, ValueError, OverflowError):
                        pass
                elif hasattr(times_to_plot[0], 'date'):
                    t_str = f"{times_to_plot[0].date()} - {times_to_plot[-1].date()}"
            ax.set_title(f"{self.var_name} Time Series {t_str}")

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        # If we created the figure, layout tight
        if created_fig:
            plt.tight_layout()

        return ax

    # ----------------------------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------------------------
