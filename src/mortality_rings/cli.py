from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, TwoSlopeNorm, to_rgb

try:
    from mortality_rings.v2 import render_v2_views, simulate_v2_cells
    from mortality_rings.v3 import render_v3_views, simulate_v3_cells
except ModuleNotFoundError:  # Allows direct execution as python src\mortality_rings\cli.py
    from v2 import render_v2_views, simulate_v2_cells
    from v3 import render_v3_views, simulate_v3_cells


STATBEL_URL = "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking/TF_DEATHS.zip"
WEEKS_PER_YEAR = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dendrochronology-inspired mortality cell charts from daily death counts."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--statbel", action="store_true", help="Download Statbel's Belgian daily deaths file.")
    source.add_argument("--input", type=Path, help="CSV/TXT file, or ZIP containing one CSV/TXT file.")
    parser.add_argument("--date-column", default="DATE_DEATH", help="Input date column.")
    parser.add_argument("--count-column", default="CNT", help="Input count column.")
    parser.add_argument("--sep", default="|", help="Input delimiter for CSV/TXT files.")
    parser.add_argument("--dayfirst", action="store_true", default=True, help="Parse dates as day-first.")
    parser.add_argument("--monthfirst", dest="dayfirst", action="store_false", help="Parse dates as month-first.")
    parser.add_argument("--first-year", type=int, default=None, help="First visible year. Defaults to data minimum.")
    parser.add_argument("--last-year", type=int, default=None, help="Last visible year. Defaults to data maximum.")
    parser.add_argument("--baseline-start", type=int, default=1992, help="First baseline year.")
    parser.add_argument("--baseline-end", type=int, default=2019, help="Last baseline year.")
    parser.add_argument("--people-per-cell", type=float, default=25, help="Deaths represented by one deposited cell.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic cell deposition.")
    parser.add_argument("--cell-growth", type=float, default=0.0018, help="How strongly each cell grows the bark.")
    parser.add_argument("--ring-rest", type=float, default=0.012, help="Quiet growth added after each year's cells.")
    parser.add_argument("--five-year-gap", type=float, default=0.055, help="Extra pale resting band after five-year intervals.")
    parser.add_argument("--cell-min-length", type=float, default=0.013, help="Minimum rendered cell length.")
    parser.add_argument("--cell-max-length", type=float, default=0.034, help="Maximum rendered cell length.")
    parser.add_argument("--clip-low", type=float, default=-0.18, help="Low color clipping bound, e.g. -0.18.")
    parser.add_argument("--clip-high", type=float, default=0.40, help="High color clipping bound, e.g. 0.40.")
    parser.add_argument("--colormap", default="wood-blood", help="V1 Matplotlib colormap for seasonal excess.")
    parser.add_argument("--renderer", choices=["v1", "v2", "v3", "both", "all"], default="both", help="Renderer to run.")
    parser.add_argument("--v2-views", choices=["art", "analytical", "both"], default="both", help="V2 views to render.")
    parser.add_argument("--v2-cell-radius", type=float, default=0.0038, help="V2 deposited-cell collision radius.")
    parser.add_argument("--v2-relax-iterations", type=int, default=72, help="V2 relaxation iterations per year.")
    parser.add_argument("--v2-outward-strength", type=float, default=0.016, help="V2 weak outward growth force.")
    parser.add_argument("--v3-views", choices=["art", "analytical", "both"], default="both", help="V3 views to render.")
    parser.add_argument("--v3-people-per-cell", type=float, default=100, help="Deaths represented by one V3 cambium cell.")
    parser.add_argument("--v3-cell-radius", type=float, default=0.0062, help="V3 deposited-cell collision radius.")
    parser.add_argument("--v3-relax-iterations", type=int, default=60, help="V3 relaxation iterations per year.")
    parser.add_argument("--title", default="Belgium mortality tree", help="Chart title.")
    parser.add_argument("--subtitle", default=None, help="Chart subtitle.")
    parser.add_argument("--source-note", default=None, help="Footer source note.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--name", default=None, help="Output filename stem.")
    parser.add_argument("--png-dpi", type=int, default=420, help="Static PNG export resolution.")
    parser.add_argument("--fps", type=float, default=1.35, help="Animation frames per second.")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF export.")
    parser.add_argument("--no-mp4", action="store_true", help="Skip MP4 export.")
    return parser.parse_args()


def download_statbel() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="mortality_rings_"))
    target = tmp_dir / "TF_DEATHS.zip"
    urllib.request.urlretrieve(STATBEL_URL, target)
    return target


def read_table(path: Path, sep: str) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in {".csv", ".txt"}
            ]
            if not candidates:
                raise ValueError(f"No CSV/TXT file found inside {path}")
            with archive.open(candidates[0]) as handle:
                return pd.read_csv(handle, sep=sep)
    return pd.read_csv(path, sep=sep)


def load_daily_deaths(args: argparse.Namespace) -> pd.DataFrame:
    input_path = download_statbel() if args.statbel or args.input is None else args.input
    raw = read_table(input_path, args.sep)
    if args.date_column not in raw.columns or args.count_column not in raw.columns:
        raise ValueError(
            f"Input must contain columns {args.date_column!r} and {args.count_column!r}. "
            f"Available columns: {', '.join(raw.columns)}"
        )

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[args.date_column], dayfirst=args.dayfirst, errors="coerce"),
            "deaths": pd.to_numeric(raw[args.count_column], errors="coerce"),
        }
    ).dropna()
    if daily.empty:
        raise ValueError("No valid daily rows found after parsing dates and counts.")
    return daily.sort_values("date")


def add_calendar_fields(daily: pd.DataFrame) -> pd.DataFrame:
    days_in_year = np.where(daily["date"].dt.is_leap_year, 366, 365)
    out = daily.copy()
    out["year"] = out["date"].dt.year.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear.astype(int)
    out["days_in_year"] = days_in_year
    out["week"] = (((out["day_of_year"] - 1) * WEEKS_PER_YEAR) // days_in_year + 1).astype(int)
    out["theta_center"] = 2 * np.pi * (out["day_of_year"] - 0.5) / days_in_year
    return out


def build_weekly_data(
    daily: pd.DataFrame,
    first_year: int,
    last_year: int,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame:
    weekly = (
        daily.query("@first_year <= year <= @last_year")
        .groupby(["year", "week"], as_index=False)["deaths"]
        .sum()
    )
    full_index = pd.MultiIndex.from_product(
        [range(first_year, last_year + 1), range(1, WEEKS_PER_YEAR + 1)],
        names=["year", "week"],
    )
    weekly = weekly.set_index(["year", "week"]).reindex(full_index).reset_index()
    baseline = weekly.query("@baseline_start <= year <= @baseline_end")
    if baseline["deaths"].notna().sum() == 0:
        raise ValueError("Baseline has no data. Adjust --baseline-start/--baseline-end.")

    week_reference = baseline.groupby("week")["deaths"].median().rename("baseline_median").reset_index()
    week_reference["baseline_share"] = week_reference["baseline_median"] / week_reference["baseline_median"].sum()
    weekly = weekly.merge(week_reference, on="week", how="left")
    annual_totals = weekly.groupby("year")["deaths"].transform("sum")
    weekly["expected_share_deaths"] = annual_totals * weekly["baseline_share"]
    weekly["seasonal_concentration_pct"] = (
        weekly["deaths"] - weekly["expected_share_deaths"]
    ) / weekly["expected_share_deaths"]

    baseline_annual = (
        weekly.query("@baseline_start <= year <= @baseline_end")
        .groupby("year", as_index=False)["deaths"]
        .sum()
        .dropna()
    )
    years = baseline_annual["year"].to_numpy(dtype=float)
    totals = baseline_annual["deaths"].to_numpy(dtype=float)
    if len(baseline_annual) >= 2:
        deltas_x = years[:, None] - years[None, :]
        deltas_y = totals[:, None] - totals[None, :]
        upper = deltas_x > 0
        slope = float(np.median(deltas_y[upper] / deltas_x[upper]))
        intercept = float(np.median(totals - slope * years))
    else:
        slope = 0.0
        intercept = float(totals[0])

    annual_expected = weekly[["year"]].drop_duplicates().copy()
    annual_expected["expected_annual_deaths"] = intercept + slope * annual_expected["year"]
    floor = max(float(np.nanmedian(totals)) * 0.35, 1.0)
    annual_expected["expected_annual_deaths"] = annual_expected["expected_annual_deaths"].clip(lower=floor)
    weekly = weekly.merge(annual_expected, on="year", how="left")
    weekly["expected_abs_deaths"] = weekly["expected_annual_deaths"] * weekly["baseline_share"]
    weekly["absolute_excess_deaths"] = weekly["deaths"] - weekly["expected_abs_deaths"]
    weekly["positive_excess_deaths"] = weekly["absolute_excess_deaths"].clip(lower=0)
    weekly["deficit_deaths"] = (-weekly["absolute_excess_deaths"]).clip(lower=0)
    weekly["absolute_excess_pct"] = weekly["absolute_excess_deaths"] / weekly["expected_abs_deaths"]

    weekly["expected_deaths"] = weekly["expected_share_deaths"]
    weekly["seasonal_excess_pct"] = weekly["seasonal_concentration_pct"]
    weekly["deaths"] = weekly["deaths"].fillna(0)
    for column in ["seasonal_concentration_pct", "seasonal_excess_pct", "absolute_excess_pct"]:
        weekly[column] = weekly[column].replace([np.inf, -np.inf], np.nan)
    return weekly


def attach_weekly_context(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    return daily.merge(weekly[["year", "week", "seasonal_excess_pct"]], on=["year", "week"], how="left")


def polar_to_xy(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return radius * np.sin(theta), radius * np.cos(theta)


def angular_distance(grid: np.ndarray, theta: float) -> np.ndarray:
    return np.angle(np.exp(1j * (grid - theta)))


def smooth_noise(rng: np.random.Generator, size: int, passes: int = 20) -> np.ndarray:
    values = rng.normal(0, 1, size)
    for _ in range(passes):
        values = (np.roll(values, 1) + values + np.roll(values, -1)) / 3
    values -= values.mean()
    scale = np.max(np.abs(values))
    return values / scale if scale else values


def smooth_boundary(boundary: np.ndarray, passes: int = 1) -> np.ndarray:
    out = boundary.copy()
    for _ in range(passes):
        out = np.roll(out, 1) * 0.24 + out * 0.52 + np.roll(out, -1) * 0.24
    return out


def season_tint(theta: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [0.00, *to_rgb("#6f9fa9")],
            [0.17, *to_rgb("#aebd9d")],
            [0.33, *to_rgb("#d6a15f")],
            [0.50, *to_rgb("#d49b55")],
            [0.67, *to_rgb("#bc755d")],
            [0.83, *to_rgb("#8da0a1")],
            [1.00, *to_rgb("#6f9fa9")],
        ]
    )
    position = (theta % (2 * np.pi)) / (2 * np.pi)
    return np.column_stack(
        [
            np.interp(position, anchors[:, 0], anchors[:, 1]),
            np.interp(position, anchors[:, 0], anchors[:, 2]),
            np.interp(position, anchors[:, 0], anchors[:, 3]),
        ]
    )


def cell_colors(cells: pd.DataFrame, cmap: Colormap, norm: TwoSlopeNorm) -> np.ndarray:
    excess = cells["seasonal_excess_pct"].clip(norm.vmin, norm.vmax).fillna(0)
    excess_rgb = cmap(norm(excess))[:, :3]
    tint_rgb = season_tint(cells["theta"].to_numpy())
    rgb = 0.92 * excess_rgb + 0.08 * tint_rgb
    alpha = cells["alpha"].to_numpy() if "alpha" in cells else np.full(len(cells), 0.9)
    return np.column_stack([np.clip(rgb, 0, 1), alpha])


def simulate_cells(daily: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    theta_grid = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    boundary = 0.035 + 0.012 * smooth_noise(rng, len(theta_grid))
    cambium_bias = smooth_noise(rng, len(theta_grid), passes=38)
    rows: list[dict[str, float]] = []

    for year, year_days in daily.groupby("year", sort=True):
        typical_deaths = max(float(year_days["deaths"].median()), 1.0)
        events = year_days.sample(frac=1, random_state=int(rng.integers(0, 2**32 - 1)))
        for record in events.itertuples(index=False):
            if not np.isfinite(record.deaths) or record.deaths <= 0:
                continue
            count = max(1, int(np.floor(record.deaths / args.people_per_cell + rng.random())))
            day_width = 2 * np.pi / record.days_in_year

            for _ in range(count):
                theta = (record.theta_center + rng.normal(0, day_width * 2.1)) % (2 * np.pi)
                index = int(theta / (2 * np.pi) * len(theta_grid)) % len(theta_grid)
                radius = boundary[index] + rng.normal(0, 0.014) - rng.uniform(0.0, 0.04)
                radius = max(0.002, radius)
                x, y = polar_to_xy(np.array([theta]), np.array([radius]))
                ink_strength = float(np.clip((radius / 0.92) ** 0.9, 0.035, 1.0))
                radial_scale = 0.18 + 0.82 * (ink_strength**0.65)

                tangent = np.array([np.cos(theta), -np.sin(theta)])
                normal = np.array([np.sin(theta), np.cos(theta)])
                direction = tangent * rng.normal(1.0, 0.08) + normal * rng.normal(0.0, 0.22)
                direction = direction / np.linalg.norm(direction)
                length = rng.uniform(args.cell_min_length, args.cell_max_length) * radial_scale

                rows.append(
                    {
                        "year": year,
                        "x1": x[0] - direction[0] * length / 2,
                        "y1": y[0] - direction[1] * length / 2,
                        "x2": x[0] + direction[0] * length / 2,
                        "y2": y[0] + direction[1] * length / 2,
                        "theta": theta,
                        "seasonal_excess_pct": record.seasonal_excess_pct,
                        "line_width": rng.uniform(0.32, 0.78) * radial_scale,
                        "alpha": 0.035 + 0.42 * ink_strength,
                    }
                )

                spread = 0.024 + 0.026 * rng.random()
                local_bias = 0.72 + 0.68 * (cambium_bias[index] + 1) / 2
                death_pressure = np.clip((record.deaths / typical_deaths) ** 0.55, 0.72, 1.75)
                growth = args.cell_growth * local_bias * death_pressure * rng.uniform(0.55, 1.35)
                distance = angular_distance(theta_grid, theta)
                boundary += growth * np.exp(-0.5 * (distance / spread) ** 2)

        boundary = smooth_boundary(boundary, passes=3)
        cambium_bias = smooth_boundary(0.72 * cambium_bias + 0.28 * smooth_noise(rng, len(theta_grid), passes=34), passes=2)
        rest_profile = 0.2 + 1.2 * (cambium_bias + 1) / 2
        boundary += args.ring_rest * rest_profile + 0.012 * smooth_noise(rng, len(theta_grid), passes=24)
        if year % 5 == 0:
            gap_profile = 0.82 + 0.36 * (cambium_bias + 1) / 2
            boundary += args.five_year_gap * gap_profile

    return pd.DataFrame(rows)


def make_palette(args: argparse.Namespace) -> tuple[Colormap, TwoSlopeNorm]:
    if args.colormap == "wood-blood":
        cmap = LinearSegmentedColormap.from_list(
            "wood_blood_v1",
            ["#376b72", "#b8afa0", "#d8c5a0", "#c58b33", "#a94f2d", "#421827"],
        )
    else:
        cmap = plt.get_cmap(args.colormap)
    return cmap, TwoSlopeNorm(vmin=args.clip_low, vcenter=0, vmax=args.clip_high)


def add_figure_text(fig: plt.Figure, args: argparse.Namespace, first_year: int, last_year: int) -> None:
    subtitle = args.subtitle or (
        f"Belgian deaths, {first_year}-{last_year}: one deposited cell is about {args.people_per_cell:g} deaths"
    )
    note = args.source_note or f"Source: Statbel open data, Number of deaths per day, {first_year}-{last_year}."
    method = (
        f"Cells grow near the bark; angle follows day of year; color compares weeks with a "
        f"{args.baseline_start}-{args.baseline_end} seasonal profile."
    )
    fig.text(0.5, 0.962, args.title, ha="center", va="center", fontsize=30, fontweight="bold", color="#201f1b")
    fig.text(0.5, 0.928, subtitle, ha="center", va="center", fontsize=12.5, color="#635f56")
    fig.text(0.5, 0.047, note, ha="center", va="center", fontsize=9.3, color="#635f56")
    fig.text(0.5, 0.029, method, ha="center", va="center", fontsize=8.7, color="#756f64")


def add_colorbar(fig: plt.Figure, cmap: Colormap, norm: TwoSlopeNorm) -> None:
    legend_ax = fig.add_axes([0.31, 0.083, 0.38, 0.018])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    legend_ax.imshow(gradient, aspect="auto", cmap=cmap)
    legend_ax.set_yticks([])
    legend_ax.set_xticks([0, 64, 128, 192, 255])
    legend_ax.set_xticklabels(
        [f"{norm.vmin:.0%}", "lower", "expected", "higher", f"{norm.vmax:.0%}+"],
        fontsize=8,
        color="#635f56",
    )
    for spine in legend_ax.spines.values():
        spine.set_visible(False)


def draw_frame(
    ax: plt.Axes,
    cells: pd.DataFrame,
    max_extent: float,
    year: int,
    cmap: Colormap,
    norm: TwoSlopeNorm,
    annual_totals: pd.Series,
) -> None:
    ax.clear()
    visible = cells[cells["year"] <= year]
    if not visible.empty:
        colors = cell_colors(visible, cmap, norm)
        segments = visible[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 2, 2)
        ax.add_collection(
            LineCollection(
                segments,
                colors=colors,
                linewidths=visible["line_width"].to_numpy(),
                alpha=0.88,
                capstyle="round",
            )
        )

    latest_total = int(annual_totals.loc[year])
    ax.text(
        -max_extent * 0.92,
        -max_extent * 0.94,
        f"{year}  |  {latest_total:,} deaths",
        ha="left",
        va="center",
        fontsize=11,
        color="#5b554b",
    )

    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#fbf7ef")


def prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, int, int]:
    all_daily = add_calendar_fields(load_daily_deaths(args))
    first_year = args.first_year or int(all_daily["year"].min())
    last_year = args.last_year or int(all_daily["year"].max())
    context_start = min(first_year, args.baseline_start)
    context_end = max(last_year, args.baseline_end)
    weekly_source = all_daily.query("@context_start <= year <= @context_end").copy()
    weekly = build_weekly_data(weekly_source, context_start, context_end, args.baseline_start, args.baseline_end)
    daily = all_daily.query("@first_year <= year <= @last_year").copy()
    daily = attach_weekly_context(daily, weekly)
    annual_totals = daily.groupby("year")["deaths"].sum().astype(int).loc[first_year:last_year]
    return daily, weekly, annual_totals, first_year, last_year


def render_v1(
    daily: pd.DataFrame,
    annual_totals: pd.Series,
    first_year: int,
    last_year: int,
    args: argparse.Namespace,
    stem: str,
) -> list[Path]:
    cells = simulate_cells(daily, args)
    if cells.empty:
        raise ValueError("No cells were generated. Check the date range and people-per-cell value.")

    years = sorted(annual_totals.index)
    max_extent = float(np.nanmax(np.abs(cells[["x1", "y1", "x2", "y2"]].to_numpy()))) * 1.11

    cmap, norm = make_palette(args)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fbf7ef", "savefig.facecolor": "#fbf7ef"})
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.945, top=0.875, bottom=0.155)
    add_figure_text(fig, args, first_year, last_year)
    add_colorbar(fig, cmap, norm)

    def update(frame_index: int) -> None:
        draw_frame(ax, cells, max_extent, years[frame_index], cmap, norm, annual_totals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    update(len(years) - 1)
    png_path = args.output_dir / f"{stem}_v1.png"
    fig.savefig(png_path, dpi=args.png_dpi)
    outputs.append(png_path)

    if not args.no_gif:
        ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
        gif_path = args.output_dir / f"{stem}_v1.gif"
        ani.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
        outputs.append(gif_path)
    if not args.no_mp4:
        ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
        mp4_path = args.output_dir / f"{stem}_v1.mp4"
        try:
            ani.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
            outputs.append(mp4_path)
        except Exception as exc:
            print(f"Skipping MP4 export because ffmpeg failed: {exc}")

    plt.close(fig)
    return outputs


def selected_v2_views(args: argparse.Namespace) -> list[str]:
    if args.v2_views == "both":
        return ["art", "analytical"]
    return [args.v2_views]


def selected_v3_views(args: argparse.Namespace) -> list[str]:
    if args.v3_views == "both":
        return ["art", "analytical"]
    return [args.v3_views]


def render(args: argparse.Namespace) -> list[Path]:
    daily, weekly, annual_totals, first_year, last_year = prepare_data(args)
    stem = args.name or f"mortality_tree_{first_year}_{last_year}"
    outputs: list[Path] = []

    if args.renderer in {"v1", "both", "all"}:
        outputs.extend(render_v1(daily, annual_totals, first_year, last_year, args, stem))

    if args.renderer in {"v2", "both", "all"}:
        simulation = simulate_v2_cells(weekly, first_year, last_year, args)
        if simulation.cells.empty:
            raise ValueError("No V2 cells were generated. Check the date range and people-per-cell value.")
        outputs.extend(
            render_v2_views(
                simulation=simulation,
                annual_totals=annual_totals,
                first_year=first_year,
                last_year=last_year,
                args=args,
                stem=stem,
                views=selected_v2_views(args),
            )
        )

    if args.renderer in {"v3", "all"}:
        simulation = simulate_v3_cells(weekly, first_year, last_year, args)
        if simulation.cells.empty:
            raise ValueError("No V3 cells were generated. Check the date range and v3-people-per-cell value.")
        outputs.extend(
            render_v3_views(
                simulation=simulation,
                annual_totals=annual_totals,
                first_year=first_year,
                last_year=last_year,
                args=args,
                stem=stem,
                views=selected_v3_views(args),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "weekly_mortality_summary.csv"
    weekly.to_csv(summary_path, index=False)
    outputs.append(summary_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = render(args)
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
