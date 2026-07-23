"""Visualization helpers for multi-rank statistics output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import netCDF4 as nc
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class MultiRankPlotter:
    """Explicit plotting service for one multi-rank reader."""

    def __init__(self, owner) -> None:
        self.owner = owner

    @property
    def _map_shape(self):
        return self.owner._map_shape

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
        return self.owner.time_len

    @property
    def times(self):
        return self.owner.times

    @property
    def var_name(self):
        return self.owner.var_name

    def _safe_time_str(self, value):
        return self.owner._safe_time_str(value)

    def get_grid(
        self, t_index: int, level=None, trial: int = 0,
        fill_value: float = np.nan, dtype=None,
    ):
        return self.owner.get_grid(t_index, level, trial, fill_value, dtype)

    def get_series(
        self, points, level=None, trial: int = 0,
        fill_value: float = np.nan, dtype=None,
    ):
        return self.owner.get_series(points, level, trial, fill_value, dtype)

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

        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")

        t_str = f"t={t_index}"
        if self.times:
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
            vals: List[np.ndarray] = []
            for info in self._rank_files:
                if info["saved_points"] == 0:
                    continue
                if info["x"] is None or info["y"] is None:
                    raise RuntimeError("map_shape not set and no converter-provided (x,y).")
                xs.append(info["x"])
                ys.append(info["y"])
                cache_arr = info.get("cache")
                if cache_arr is not None:
                    # cache_arr shape: (time, [trial], saved_points, [levels])
                    indices = [t_index]
                    if info["has_trials"]:
                        indices.append(trial)
                    indices.append(slice(None))
                    if info["has_levels"]:
                        indices.append(level if level is not None else 0)
                    vv = cache_arr[tuple(indices)]
                else:
                    orig_t = int(self._t_indices[t_index])
                    with nc.Dataset(info["path"], "r") as ds:
                        var = ds.variables[self.var_name]

                        # Build slicing tuple
                        # 1. time
                        indices = [orig_t]

                        # 2. trial
                        if info["has_trials"]:
                            indices.append(trial)

                        # 3. saved_points (all)
                        indices.append(slice(None))

                        # 4. levels
                        if info["has_levels"]:
                            indices.append(level if level is not None else 0)

                        vv = var[tuple(indices)]
                vals.append(np.array(vv))
            x_all = np.concatenate(xs) if xs else np.array([])
            y_all = np.concatenate(ys) if ys else np.array([])
            v_all = np.concatenate(vals) if vals else np.array([])
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
                ax.invert_yaxis()

        else:
            raise RuntimeError("Cannot plot without map_shape and scatter fallback disabled.")
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
        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        nx_, ny_ = self._map_shape

        xmin = 0
        xmax = nx_ - 1
        ymin = 0
        ymax = ny_ - 1

        if auto_crop and (x_range is None and y_range is None):
            # Fetch first frame
            grid_0 = self.get_grid(t_start, level=level, trial=trial)
            valid_mask = np.isfinite(grid_0)
            if np.any(valid_mask):
                xs, ys = np.where(valid_mask)
                xmin_c, xmax_c = xs.min(), xs.max()
                ymin_c, ymax_c = ys.min(), ys.max()

                xmin = max(0, xmin_c - crop_pad)
                xmax = min(nx_ - 1, xmax_c + crop_pad)
                ymin = max(0, ymin_c - crop_pad)
                ymax = min(ny_ - 1, ymax_c + crop_pad)

        # Override with manual ranges if provided
        if x_range is not None:
            xmin = max(0, int(x_range[0]))
            xmax = min(nx_ - 1, int(x_range[1]))
        if y_range is not None:
            ymin = max(0, int(y_range[0]))
            ymax = min(ny_ - 1, int(y_range[1]))

        if xmin > xmax or ymin > ymax:
            raise ValueError("Invalid x_range or y_range")

        first_grid = self.get_grid(t_start, level=level, trial=trial)
        window = first_grid[xmin:xmax + 1, ymin:ymax + 1]
        if vmin is None:
            vmin = np.nanmin(window) if np.isfinite(window).any() else 0.0
        if vmax is None:
            vmax = np.nanmax(window) if np.isfinite(window).any() else 1.0
        if not (vmax > vmin):
            vmax = vmin + 1.0

        extent = (xmin - 0.5, xmax + 0.5, ymax + 0.5, ymin - 0.5)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(window.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Use our robust time logic if available
        t_label = f"t={t_start}"
        if self.times:
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
            if self.times:
                t_lbl = self._safe_time_str(self.times[ti])

            ttl.set_text(f"{self.var_name} @ {t_lbl}")
            return [im, ttl]

        frames = t_end - t_start
        ani = animation.FuncAnimation(fig, _update, frames=frames, interval=1000 / fps, blit=False)

        out_path = Path(out_path)
        if out_path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=fps)
            ani.save(out_path, writer=writer)
        else:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("ffmpeg writer not found. Install ffmpeg or use .gif.")
            writer_type = animation.writers["ffmpeg"]
            writer = writer_type(
                fps=fps, metadata={"artist": "MultiRankStatsReader"},
            )
            ani.save(out_path, writer=writer)
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
        **kwargs
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

        if isinstance(trial, int):
            trials = [trial]
        else:
            trials = trial

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Select Time Axis Strategy
        # Prefer raw numeric values + FuncFormatter for perfect calendar support
        use_numeric_time = False
        if (
            self._time_values_num is not None
            and self._time_units is not None
            and self._time_calendar is not None
        ):
            times_to_plot = self._time_values_num
            use_numeric_time = True
        elif self.times:
            # Fallback to datetime list (cached property)
            times_to_plot = self.times
        else:
            # Fallback to simple indices
            times_to_plot = np.arange(self.time_len)

        # Ensure points is in a format suitable for get_series

        for t in trials:
            # Fetch data: shape (time_len, num_points)
            data = self.get_series(points, level=level, trial=t)
            num_points = data.shape[1]

            for i in range(num_points):
                # Construct label
                # If multiple trials, include trial info. If multiple points, include point info.
                lbl_parts = []

                # Point Label
                if labels and i < len(labels):
                    lbl_parts.append(str(labels[i]))
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
                elif not labels and num_points == 1:
                     # Single point, single trial, explicit label is nice
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

        if title:
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
