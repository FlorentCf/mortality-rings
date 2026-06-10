from __future__ import annotations

import argparse
import math
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


STATBEL_URL = "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking/TF_DEATHS.zip"
WEEKS_PER_YEAR = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create radial mortality ring charts and animations from daily death counts."
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
    parser.add_argument("--clip-low", type=float, default=-0.25, help="Low color clipping bound, e.g. -0.25.")
    parser.add_argument("--clip-high", type=float, default=0.75, help="High color clipping bound, e.g. 0.75.")
    parser.add_argument("--title", default="Belgium mortality rings", help="Chart title.")
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Chart subtitle. Defaults to a generated baseline description.",
    )
    parser.add_argument("--source-note", default=None, help="Footer source note.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--name", default=None, help="Output filename stem.")
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


def build_weekly_data(
    daily: pd.DataFrame,
    first_year: int,
    last_year: int,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame:
    days_in_year = np.where(daily["date"].dt.is_leap_year, 366, 365)
    weekly = (
        daily.assign(
            year=daily["date"].dt.year.astype(int),
            week=(((daily["date"].dt.dayofyear - 1) * WEEKS_PER_YEAR) // days_in_year + 1).astype(int),
        )
        .query("@first_year <= year <= @last_year")
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
    weekly = weekly.merge(week_reference, on="week", how="left")
    weekly["excess_pct"] = (weekly["deaths"] - weekly["baseline_median"]) / weekly["baseline_median"]
    weekly["deaths"] = weekly["deaths"].fillna(0)
    weekly["excess_pct"] = weekly["excess_pct"].replace([np.inf, -np.inf], np.nan)
    return weekly


def prepare_plot_data(weekly: pd.DataFrame) -> pd.DataFrame:
    years = sorted(weekly["year"].unique())
    year_to_ring = {year: i for i, year in enumerate(years)}
    data = weekly.copy()
    data["ring"] = data["year"].map(year_to_ring)
    data["theta"] = 2 * np.pi * (data["week"] - 0.5) / WEEKS_PER_YEAR
    data["width"] = 2 * np.pi / WEEKS_PER_YEAR * 0.94
    return data


def make_palette() -> tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    cmap = LinearSegmentedColormap.from_list(
        "mortality_balance",
        ["#24476f", "#70a7c8", "#f7f0df", "#e39a5d", "#8f1d2c", "#361020"],
    )
    return cmap, TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.75)


def add_figure_text(fig: plt.Figure, args: argparse.Namespace, first_year: int, last_year: int) -> None:
    subtitle = args.subtitle or (
        f"Weekly deaths by calendar year, colored versus the "
        f"{args.baseline_start}-{args.baseline_end} median for the same week"
    )
    note = args.source_note or (
        f"Source: Statbel open data, Number of deaths per day, {first_year}-{last_year}. "
        f"Color scale is clipped at {args.clip_low:.0%} and {args.clip_high:.0%} "
        "so ordinary seasonal variation remains visible."
    )
    fig.text(0.5, 0.962, args.title, ha="center", va="center", fontsize=30, fontweight="bold", color="#1f2933")
    fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=13, color="#59616e")
    fig.text(0.5, 0.045, note, ha="center", va="center", fontsize=9.5, color="#59616e")


def add_legend(fig: plt.Figure, cmap: LinearSegmentedColormap, norm: TwoSlopeNorm) -> None:
    legend_ax = fig.add_axes([0.31, 0.077, 0.38, 0.018])
    gradient = np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1)
    legend_ax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm)
    legend_ax.set_yticks([])
    legend_ax.set_xticks([0, 64, 128, 192, 255])
    legend_ax.set_xticklabels([f"{norm.vmin:.0%}", "lower", "baseline", "higher", f"{norm.vmax:.0%}+"], fontsize=8, color="#59616e")
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    fig.legend(
        handles=[Patch(facecolor="#8f1d2c", label="Extreme high weeks are intentionally saturated")],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.103),
        frameon=False,
        fontsize=9,
        labelcolor="#59616e",
    )


def draw_frame(
    ax: plt.Axes,
    data: pd.DataFrame,
    year: int,
    cmap: LinearSegmentedColormap,
    norm: TwoSlopeNorm,
    annual_totals: pd.Series,
) -> None:
    ax.clear()
    visible = data[data["year"] <= year].copy()
    colors = cmap(norm(visible["excess_pct"].clip(norm.vmin, norm.vmax).fillna(0)))
    colors[visible["deaths"].eq(0).to_numpy()] = (0.88, 0.88, 0.86, 0.18)

    ax.bar(
        visible["theta"],
        np.full(len(visible), 0.86),
        width=visible["width"],
        bottom=visible["ring"] + 1.3,
        color=colors,
        edgecolor="#f7f2e8",
        linewidth=0.24,
        align="center",
    )

    years = sorted(data["year"].unique())
    outer_radius = len(years) + 3.1
    for theta, label in zip(
        np.linspace(0, 2 * np.pi, 12, endpoint=False),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ):
        ax.text(theta, outer_radius, label, ha="center", va="center", fontsize=11, color="#3f4652")

    for label_year in [years[0], 2000, 2010, 2020, years[-1]]:
        if label_year <= year and label_year in years:
            ring = data.loc[data["year"].eq(label_year), "ring"].iloc[0] + 1.73
            ax.text(math.radians(356), ring, str(label_year), ha="right", va="center", fontsize=8, color="#374151")

    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0, len(years) + 4.4)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#fbfaf6")

    latest_total = int(annual_totals.loc[year])
    ax.text(
        0.5,
        0.5,
        f"{year}\n{latest_total:,}\ndeaths",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color="#222831",
        fontweight="bold",
        linespacing=1.18,
    )


def render(args: argparse.Namespace) -> list[Path]:
    daily = load_daily_deaths(args)
    first_year = args.first_year or int(daily["date"].dt.year.min())
    last_year = args.last_year or int(daily["date"].dt.year.max())
    weekly = build_weekly_data(daily, first_year, last_year, args.baseline_start, args.baseline_end)
    data = prepare_plot_data(weekly)
    annual_totals = daily.groupby(daily["date"].dt.year)["deaths"].sum().astype(int)
    annual_totals = annual_totals.loc[first_year:last_year]
    years = sorted(data["year"].unique())

    cmap, norm = make_palette()
    norm.vmin = args.clip_low
    norm.vmax = args.clip_high

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fbfaf6", "savefig.facecolor": "#fbfaf6"})
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=150, subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.065, right=0.935, top=0.845, bottom=0.18)
    add_figure_text(fig, args, first_year, last_year)
    add_legend(fig, cmap, norm)

    def update(frame_index: int) -> None:
        draw_frame(ax, data, years[frame_index], cmap, norm, annual_totals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"mortality_rings_{first_year}_{last_year}"
    outputs = []

    update(len(years) - 1)
    png_path = args.output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=180)
    outputs.append(png_path)

    if not args.no_gif:
        ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
        gif_path = args.output_dir / f"{stem}.gif"
        ani.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
        outputs.append(gif_path)
    if not args.no_mp4:
        ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
        mp4_path = args.output_dir / f"{stem}.mp4"
        try:
            ani.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
            outputs.append(mp4_path)
        except Exception as exc:
            print(f"Skipping MP4 export because ffmpeg failed: {exc}")

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
