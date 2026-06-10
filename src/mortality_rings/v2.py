from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree


BACKGROUND = "#fbf7ef"
WEEKS_PER_YEAR = 52

SEASON_PALETTE = np.array(
    [
        [0.00, *to_rgb("#2f7fb8")],
        [0.15, *to_rgb("#3197bf")],
        [0.32, *to_rgb("#17a96c")],
        [0.48, *to_rgb("#69b947")],
        [0.64, *to_rgb("#c4a03a")],
        [0.78, *to_rgb("#c06b3b")],
        [0.90, *to_rgb("#4f8fa9")],
        [1.00, *to_rgb("#2f7fb8")],
    ]
)
OCHRE = np.array(to_rgb("#c58b33"))
RUST = np.array(to_rgb("#a94f2d"))
DRIED_RED = np.array(to_rgb("#7f2630"))
BURGUNDY = np.array(to_rgb("#421827"))
COOL_DEFICIT = np.array(to_rgb("#376b72"))
CONTOUR = "#fffdf8"
CONTOUR_ANALYTICAL = "#fff8ec"


@dataclass(frozen=True)
class V2Simulation:
    cells: pd.DataFrame
    outlines: dict[int, np.ndarray]
    theta_grid: np.ndarray
    diagnostics: dict[str, float]


def _polar_to_xy(theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])


def _theta_from_xy(positions: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(positions[:, 0], positions[:, 1]), 2 * np.pi)


def _smooth_circular(values: np.ndarray, passes: int = 2) -> np.ndarray:
    out = values.astype(float).copy()
    for _ in range(passes):
        out = np.roll(out, 1) * 0.24 + out * 0.52 + np.roll(out, -1) * 0.24
    return out


def _smooth_noise(rng: np.random.Generator, size: int, passes: int = 24) -> np.ndarray:
    values = rng.normal(0, 1, size)
    for _ in range(passes):
        values = (np.roll(values, 1) + values + np.roll(values, -1)) / 3
    values -= values.mean()
    scale = np.max(np.abs(values))
    return values / scale if scale else values


def _stochastic_count(value: float, rng: np.random.Generator) -> int:
    if not np.isfinite(value) or value <= 0:
        return 0
    base = int(np.floor(value))
    return base + int(rng.random() < value - base)


def _fill_profile(profile: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    finite = np.isfinite(profile)
    if finite.all():
        return profile
    if not finite.any():
        return fallback.copy()

    index = np.arange(len(profile))
    known = index[finite]
    values = profile[finite]
    padded_index = np.concatenate([known - len(profile), known, known + len(profile)])
    padded_values = np.concatenate([values, values, values])
    return np.interp(index, padded_index, padded_values)


def _bark_profile(positions: np.ndarray, theta_grid: np.ndarray, fallback: np.ndarray, cell_radius: float) -> np.ndarray:
    if len(positions) == 0:
        return fallback.copy()

    theta = _theta_from_xy(positions)
    radius = np.linalg.norm(positions, axis=1)
    bins = np.floor(theta / (2 * np.pi) * len(theta_grid)).astype(int) % len(theta_grid)
    profile = np.full(len(theta_grid), np.nan)
    maxima = np.full(len(theta_grid), -np.inf)
    np.maximum.at(maxima, bins, radius)
    profile[maxima > -np.inf] = maxima[maxima > -np.inf] + cell_radius * 2.1
    profile = _fill_profile(profile, fallback)
    return _smooth_circular(profile, passes=5)


def _season_mu(week: int) -> float:
    position = (week - 0.5) / WEEKS_PER_YEAR
    flow = 0.30 * np.sin(2 * np.pi * position * 2.6 + 0.4)
    flow += 0.16 * np.sin(2 * np.pi * position * 5.2 + 1.7)
    return float((2 * np.pi * position + flow) % (2 * np.pi))


def _sample_theta(
    rng: np.random.Generator,
    week: int,
    role: str,
    anomaly: float,
) -> float:
    mu = _season_mu(week)
    if role == "excess":
        kappa = 4.2 + 16 * np.clip(anomaly, 0, 0.9)
    elif role == "deficit":
        kappa = 1.25
    else:
        kappa = 0.85
        if rng.random() < 0.24:
            return float(rng.uniform(0, 2 * np.pi))
    return float(rng.vonmises(mu, kappa) % (2 * np.pi))


def _orientation_profile(radius_profile: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    smooth_radius = _smooth_circular(radius_profile, passes=26)
    step = 2 * np.pi / len(radius_profile)
    dr = (np.roll(smooth_radius, -1) - np.roll(smooth_radius, 1)) / (2 * step)
    dx = dr * np.cos(theta_grid) - smooth_radius * np.sin(theta_grid)
    dy = dr * np.sin(theta_grid) + smooth_radius * np.cos(theta_grid)
    return np.arctan2(dy, dx)


def _relax_positions(
    positions: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    outward_strength: float,
    mobility: np.ndarray | None = None,
) -> float:
    if len(positions) < 2:
        return 0.0

    if mobility is None:
        mobility = np.ones(len(positions), dtype=float)

    max_pair_distance = float(radii.max() * 2.08)
    for _ in range(iterations):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(max_pair_distance, output_type="ndarray")
        if len(pairs) == 0:
            break

        left = pairs[:, 0]
        right = pairs[:, 1]
        delta = positions[left] - positions[right]
        distance = np.linalg.norm(delta, axis=1)
        zero = distance < 1e-9
        if np.any(zero):
            delta[zero] = rng.normal(0, 1, (int(zero.sum()), 2))
            distance[zero] = np.linalg.norm(delta[zero], axis=1)

        wanted = radii[left] + radii[right]
        overlap = wanted - distance
        active = overlap > 0
        if not np.any(active):
            break

        left = left[active]
        right = right[active]
        delta = delta[active]
        distance = distance[active]
        overlap = overlap[active]

        pair_mobility = np.maximum(mobility[left] + mobility[right], 1e-9)
        direction = delta / distance[:, None]
        left_step = overlap * (mobility[left] / pair_mobility) * 1.02
        right_step = overlap * (mobility[right] / pair_mobility) * 1.02
        np.add.at(positions, left, direction * left_step[:, None])
        np.add.at(positions, right, -direction * right_step[:, None])

        radial = np.linalg.norm(positions, axis=1)
        unit = positions / np.maximum(radial[:, None], 1e-9)
        pressure = np.zeros(len(positions), dtype=float)
        np.add.at(pressure, left, overlap)
        np.add.at(pressure, right, overlap)
        crowded = pressure > 0
        if np.any(crowded):
            jitter = rng.normal(0, 1, (int(crowded.sum()), 2))
            jitter /= np.maximum(np.linalg.norm(jitter, axis=1)[:, None], 1e-9)
            positions[crowded] += jitter * pressure[crowded, None] * mobility[crowded, None] * 0.28
        positions += unit * (pressure[:, None] * mobility[:, None] * 0.055)
        positions += unit * (radii[:, None] * mobility[:, None] * outward_strength)

    return _measure_max_overlap(positions, radii)


def _measure_max_overlap(positions: np.ndarray, radii: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    max_pair_distance = float(radii.max() * 2.08)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(max_pair_distance, output_type="ndarray")
    if len(pairs) == 0:
        return 0.0
    left = pairs[:, 0]
    right = pairs[:, 1]
    distance = np.linalg.norm(positions[left] - positions[right], axis=1)
    overlap = radii[left] + radii[right] - distance
    overlap = overlap[overlap > 0]
    return float(overlap.max()) if len(overlap) else 0.0


def _cell_roles(
    deaths: float,
    expected_deaths: float,
    anomaly: float,
    people_per_cell: float,
    rng: np.random.Generator,
) -> list[str]:
    total_count = _stochastic_count(deaths / people_per_cell, rng)
    if total_count == 0:
        return []

    excess_people = max(deaths - expected_deaths, 0)
    excess_count = min(total_count, _stochastic_count(excess_people / people_per_cell, rng))
    remaining = total_count - excess_count

    deficit_count = 0
    if anomaly < -0.10 and remaining:
        deficit_share = min(abs(anomaly) * 0.75, 0.45)
        deficit_count = min(remaining, _stochastic_count(remaining * deficit_share, rng))
        remaining -= deficit_count

    roles = ["excess"] * excess_count + ["deficit"] * deficit_count + ["tissue"] * remaining
    rng.shuffle(roles)
    return roles


def _season_color(weeks: np.ndarray) -> np.ndarray:
    position = ((weeks - 0.5) % WEEKS_PER_YEAR) / WEEKS_PER_YEAR
    return np.column_stack(
        [
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 1]),
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 2]),
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 3]),
        ]
    )


def _pack_year_cells(
    candidates: list[dict[str, float | int | str]],
    boundary: np.ndarray,
    theta_grid: np.ndarray,
    cell_radius: float,
    rng: np.random.Generator,
    existing_positions: np.ndarray,
    existing_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]], np.ndarray]:
    if not candidates:
        return np.empty((0, 2)), np.empty(0), [], boundary.copy()

    bin_count = 16384
    lane_gap = cell_radius * 1.72
    year_gap = cell_radius * 1.88
    min_arc_spacing = cell_radius * 2.28
    lanes: list[set[int]] = []
    positions: list[np.ndarray] = []
    radii: list[float] = []
    rows: list[dict[str, float | int | str]] = []
    outer = np.full(len(theta_grid), np.nan)
    orientation_by_theta = _orientation_profile(boundary, theta_grid)
    placed_radius = cell_radius * 0.68
    existing_tree = cKDTree(existing_positions) if len(existing_positions) else None
    existing_query_radius = placed_radius + float(existing_radii.max()) if len(existing_radii) else 0.0
    current_bin_size = placed_radius * 2.05
    current_bins: dict[tuple[int, int], list[int]] = {}

    def clears_existing(candidate_xy: np.ndarray) -> bool:
        if existing_tree is None:
            return True
        neighbors = existing_tree.query_ball_point(candidate_xy, existing_query_radius)
        if not neighbors:
            return True
        neighbor_positions = existing_positions[neighbors]
        distance = np.linalg.norm(neighbor_positions - candidate_xy, axis=1)
        wanted = placed_radius + existing_radii[neighbors]
        return bool(np.all(distance >= wanted * 1.015))

    def current_key(candidate_xy: np.ndarray) -> tuple[int, int]:
        return int(np.floor(candidate_xy[0] / current_bin_size)), int(np.floor(candidate_xy[1] / current_bin_size))

    def clears_current(candidate_xy: np.ndarray) -> bool:
        if not positions:
            return True
        key_x, key_y = current_key(candidate_xy)
        neighbor_indexes: list[int] = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                neighbor_indexes.extend(current_bins.get((key_x + dx, key_y + dy), []))
        if not neighbor_indexes:
            return True
        neighbor_positions = np.vstack([positions[index] for index in neighbor_indexes])
        distance = np.linalg.norm(neighbor_positions - candidate_xy, axis=1)
        return bool(np.all(distance >= placed_radius * 2.03))

    def register_current(candidate_xy: np.ndarray) -> None:
        current_bins.setdefault(current_key(candidate_xy), []).append(len(positions) - 1)

    ordered = sorted(candidates, key=lambda item: (float(item["theta_seed"]) + rng.normal(0, 0.014)) % (2 * np.pi))

    for item in ordered:
        theta = float(item["theta_seed"])
        placed = False
        for attempt in range(18):
            candidate_theta = (theta + rng.normal(0, 0.012 + 0.002 * attempt)) % (2 * np.pi)
            grid_index = int(candidate_theta / (2 * np.pi) * len(theta_grid)) % len(theta_grid)
            bin_id = int(candidate_theta / (2 * np.pi) * bin_count) % bin_count
            lane_order = range(len(lanes) + 1)
            for lane_index in lane_order:
                if lane_index == len(lanes):
                    lanes.append(set())
                radius = boundary[grid_index] + year_gap + lane_index * lane_gap + rng.normal(0, cell_radius * 0.10)
                radius = max(cell_radius * 2.5, radius)
                bin_gap = max(2, int(np.ceil(min_arc_spacing / radius / (2 * np.pi) * bin_count)))
                occupied = lanes[lane_index]
                xy = _polar_to_xy(np.array([candidate_theta]), np.array([radius]))[0]
                if (
                    all(((bin_id + offset) % bin_count) not in occupied for offset in range(-bin_gap, bin_gap + 1))
                    and clears_existing(xy)
                    and clears_current(xy)
                ):
                    for offset in range(-bin_gap, bin_gap + 1):
                        occupied.add((bin_id + offset) % bin_count)
                    positions.append(xy)
                    register_current(xy)
                    radii.append(placed_radius)
                    row = dict(item)
                    row["theta_seed"] = candidate_theta
                    row["lane"] = lane_index
                    row["orientation"] = float(orientation_by_theta[grid_index])
                    row["dash_length"] = float(cell_radius * rng.uniform(1.34, 2.04))
                    row["line_width"] = float(rng.uniform(0.30, 0.54))
                    row["orientation_jitter"] = float(rng.normal(0, 0.045))
                    rows.append(row)
                    outer[grid_index] = max(float(outer[grid_index]) if np.isfinite(outer[grid_index]) else 0, radius + cell_radius * 1.08)
                    placed = True
                    break
            if placed:
                break

        if not placed:
            grid_index = int(theta / (2 * np.pi) * len(theta_grid)) % len(theta_grid)
            lane_index = len(lanes)
            lanes.append(set())
            radius = boundary[grid_index] + year_gap + lane_index * lane_gap
            xy = _polar_to_xy(np.array([theta]), np.array([radius]))[0]
            while not (clears_existing(xy) and clears_current(xy)):
                lane_index += 1
                radius = boundary[grid_index] + year_gap + lane_index * lane_gap
                xy = _polar_to_xy(np.array([theta]), np.array([radius]))[0]
            positions.append(xy)
            register_current(xy)
            radii.append(placed_radius)
            row = dict(item)
            row["lane"] = lane_index
            row["orientation"] = float(orientation_by_theta[grid_index])
            row["dash_length"] = float(cell_radius * 1.55)
            row["line_width"] = float(rng.uniform(0.30, 0.54))
            row["orientation_jitter"] = float(rng.normal(0, 0.045))
            rows.append(row)
            outer[grid_index] = max(float(outer[grid_index]) if np.isfinite(outer[grid_index]) else 0, radius + cell_radius * 1.08)

    next_boundary = _fill_profile(outer, boundary + year_gap)
    next_boundary = _smooth_circular(next_boundary, passes=4)
    next_boundary = np.maximum(next_boundary, boundary + year_gap)
    return np.vstack(positions), np.asarray(radii), rows, next_boundary


def simulate_v2_cells(
    weekly: pd.DataFrame,
    first_year: int,
    last_year: int,
    args,
) -> V2Simulation:
    rng = np.random.default_rng(args.seed)
    theta_grid = np.linspace(0, 2 * np.pi, 1080, endpoint=False)
    cell_radius = float(args.v2_cell_radius)
    boundary = cell_radius * 2.5 + cell_radius * 0.55 * _smooth_noise(rng, len(theta_grid), passes=34)
    boundary = np.maximum(boundary, cell_radius * 1.8)
    positions = np.empty((0, 2), dtype=float)
    radii = np.empty(0, dtype=float)
    rows: list[dict[str, float | int | str]] = []
    outlines: dict[int, np.ndarray] = {}
    target_cells = 0

    visible_weekly = weekly.query("@first_year <= year <= @last_year").copy()
    visible_weekly["deaths"] = visible_weekly["deaths"].fillna(0)
    visible_weekly["expected_deaths"] = visible_weekly["expected_deaths"].fillna(0)
    visible_weekly["seasonal_excess_pct"] = visible_weekly["seasonal_excess_pct"].fillna(0)

    for year, year_weeks in visible_weekly.groupby("year", sort=True):
        candidates: list[dict[str, float | int | str]] = []

        for record in year_weeks.itertuples(index=False):
            deaths = float(record.deaths)
            expected = float(record.expected_deaths)
            anomaly = float(record.seasonal_excess_pct)
            roles = _cell_roles(deaths, expected, anomaly, args.people_per_cell, rng)
            target_cells += deaths / args.people_per_cell

            for role in roles:
                theta = _sample_theta(rng, int(record.week), role, anomaly)
                candidates.append(
                    {
                        "year": int(record.year),
                        "week": int(record.week),
                        "theta_seed": theta,
                        "deaths": deaths,
                        "expected_deaths": expected,
                        "seasonal_excess_pct": anomaly,
                        "role": role,
                        "tone": float(rng.random()),
                    }
                )

        if candidates:
            new_positions, new_radii, new_rows, boundary = _pack_year_cells(
                candidates,
                boundary,
                theta_grid,
                cell_radius,
                rng,
                positions,
                radii,
            )
            positions = np.vstack([positions, new_positions])
            radii = np.concatenate([radii, new_radii])
            rows.extend(new_rows)

        boundary = _bark_profile(positions, theta_grid, boundary, cell_radius)
        boundary += cell_radius * 0.46 + cell_radius * 0.10 * _smooth_noise(rng, len(theta_grid), passes=28)
        outlines[int(year)] = boundary.copy()

    cells = pd.DataFrame(rows)
    if not cells.empty:
        cells[["x", "y"]] = positions[: len(cells)]
        cells["radius"] = radii[: len(cells)]

    diagnostics = {
        "cell_count": float(len(cells)),
        "target_cell_count": float(target_cells),
        "max_overlap": float(_measure_max_overlap(positions, radii)),
    }
    return V2Simulation(cells=cells, outlines=outlines, theta_grid=theta_grid, diagnostics=diagnostics)


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return a * (1 - np.asarray(t)[..., None]) + b * np.asarray(t)[..., None]


def _v2_colors(cells: pd.DataFrame, view: str) -> np.ndarray:
    if cells.empty:
        return np.empty((0, 4))

    role = cells["role"].to_numpy()
    anomaly = cells["seasonal_excess_pct"].fillna(0).to_numpy()
    tone = cells["tone"].to_numpy()
    rgb = _season_color(cells["week"].to_numpy())
    paper = np.array(to_rgb(BACKGROUND))
    rgb = _mix(rgb, paper, 0.06 + 0.08 * tone)
    alpha = np.full(len(cells), 0.93 if view == "art" else 0.84)

    excess_mask = role == "excess"
    if np.any(excess_mask):
        strength = np.clip(anomaly[excess_mask] / 0.55, 0, 1)
        mid = _mix(OCHRE, RUST, np.clip(strength * 1.4, 0, 1))
        high = _mix(DRIED_RED, BURGUNDY, np.clip((strength - 0.55) / 0.45, 0, 1))
        event_rgb = np.where((strength < 0.55)[:, None], mid, high)
        rgb[excess_mask] = _mix(rgb[excess_mask], event_rgb, 0.72 if view == "art" else 0.82)
        alpha[excess_mask] = 0.95 if view == "art" else 0.96

    deficit_mask = role == "deficit"
    if np.any(deficit_mask):
        rgb[deficit_mask] = _mix(rgb[deficit_mask], COOL_DEFICIT, 0.48 if view == "art" else 0.68)
        alpha[deficit_mask] = 0.84 if view == "art" else 0.88

    return np.column_stack([np.clip(rgb, 0, 1), alpha])


def _cell_segments(cells: pd.DataFrame) -> np.ndarray:
    positions = cells[["x", "y"]].to_numpy()
    if "orientation" in cells:
        angle = cells["orientation"].to_numpy() + cells["orientation_jitter"].to_numpy()
    else:
        theta = _theta_from_xy(positions) + cells["orientation_jitter"].to_numpy()
        angle = theta + np.pi / 2
    tangent = np.column_stack([np.cos(angle), np.sin(angle)])
    length = cells["dash_length"].to_numpy()
    return np.stack([positions - tangent * length[:, None] / 2, positions + tangent * length[:, None] / 2], axis=1)


def _outline_xy(theta_grid: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = _smooth_circular(radius, passes=18)
    closed_theta = np.concatenate([theta_grid, theta_grid[:1]])
    closed_radius = np.concatenate([radius, radius[:1]])
    xy = _polar_to_xy(closed_theta, closed_radius)
    return xy[:, 0], xy[:, 1]


def _max_extent(cells: pd.DataFrame, outlines: dict[int, np.ndarray]) -> float:
    cell_extent = 1.0
    if not cells.empty:
        cell_extent = float(np.nanmax(np.abs(cells[["x", "y"]].to_numpy())))
    outline_extent = max((float(np.nanmax(radius)) for radius in outlines.values()), default=cell_extent)
    return max(cell_extent, outline_extent) * 1.13


def _draw_v2_frame(
    ax: plt.Axes,
    simulation: V2Simulation,
    annual_totals: pd.Series,
    year: int,
    max_extent: float,
    view: str,
) -> None:
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)

    visible = simulation.cells[simulation.cells["year"] <= year]
    if not visible.empty:
        ax.add_collection(
            LineCollection(
                _cell_segments(visible),
                colors=_v2_colors(visible, view),
                linewidths=visible["line_width"].to_numpy(),
                capstyle="round",
            )
        )

    contour_color = CONTOUR if view == "art" else CONTOUR_ANALYTICAL
    for outline_year, radius in simulation.outlines.items():
        if outline_year <= year:
            x, y = _outline_xy(simulation.theta_grid, radius)
            five_year = outline_year % 5 == 0
            ax.plot(
                x,
                y,
                color=contour_color,
                linewidth=(1.02 if five_year else 0.68) if view == "art" else (0.82 if five_year else 0.54),
                alpha=0.97 if five_year else 0.92,
                zorder=4,
            )

    if view == "analytical":
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


def _add_v2_analytical_text(
    fig: plt.Figure,
    args,
    first_year: int,
    last_year: int,
    diagnostics: dict[str, float],
) -> None:
    subtitle = args.subtitle or (
        f"Belgian deaths, {first_year}-{last_year}: one cambium cell is about {args.people_per_cell:g} deaths"
    )
    source = args.source_note or f"Source: Statbel open data, Number of deaths per day, {first_year}-{last_year}."
    method = (
        "All cells carry seasonal color; excess mortality warms toward ochre, rust, and burgundy; "
        "thin white annual channels trace bark growth, with subtle five-year accents."
    )
    fig.text(0.5, 0.962, args.title, ha="center", va="center", fontsize=30, fontweight="bold", color="#201f1b")
    fig.text(0.5, 0.928, subtitle, ha="center", va="center", fontsize=12.5, color="#635f56")
    fig.text(0.5, 0.060, source, ha="center", va="center", fontsize=9.3, color="#635f56")
    fig.text(0.5, 0.041, method, ha="center", va="center", fontsize=8.7, color="#756f64")
    fig.text(
        0.5,
        0.022,
        f"V2 packed cells: {diagnostics['cell_count']:.0f}; measured max collision overlap: {diagnostics['max_overlap']:.4f}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#8a8275",
    )


def _add_v2_legend(fig: plt.Figure) -> None:
    legend_ax = fig.add_axes([0.34, 0.086, 0.32, 0.020])
    gradient = np.linspace(0, 1, 256)
    left = _mix(OCHRE, RUST, np.clip(gradient * 1.4, 0, 1))
    right = _mix(DRIED_RED, BURGUNDY, np.clip((gradient - 0.55) / 0.45, 0, 1))
    rgb = np.where((gradient < 0.55)[:, None], left, right).reshape(1, 256, 3)
    legend_ax.imshow(rgb, aspect="auto")
    legend_ax.set_yticks([])
    legend_ax.set_xticks([0, 128, 255])
    legend_ax.set_xticklabels(["mild excess", "high", "extreme"], fontsize=8, color="#635f56")
    for spine in legend_ax.spines.values():
        spine.set_visible(False)


def render_v2_views(
    simulation: V2Simulation,
    annual_totals: pd.Series,
    first_year: int,
    last_year: int,
    args,
    stem: str,
    views: list[str],
) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(annual_totals.index)
    max_extent = _max_extent(simulation.cells, simulation.outlines)
    outputs: list[Path] = []

    for view in views:
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND})
        fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=150)
        if view == "art":
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            fig.subplots_adjust(left=0.055, right=0.945, top=0.875, bottom=0.155)
            _add_v2_analytical_text(fig, args, first_year, last_year, simulation.diagnostics)
            _add_v2_legend(fig)

        def update(frame_index: int) -> None:
            _draw_v2_frame(ax, simulation, annual_totals, years[frame_index], max_extent, view)

        update(len(years) - 1)
        png_path = args.output_dir / f"{stem}_v2_{view}.png"
        fig.savefig(png_path, dpi=args.png_dpi)
        outputs.append(png_path)

        if not args.no_gif:
            ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
            gif_path = args.output_dir / f"{stem}_v2_{view}.gif"
            ani.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
            outputs.append(gif_path)

        if not args.no_mp4:
            ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
            mp4_path = args.output_dir / f"{stem}_v2_{view}.mp4"
            try:
                ani.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
                outputs.append(mp4_path)
            except Exception as exc:
                print(f"Skipping MP4 export for {view} because ffmpeg failed: {exc}")

        plt.close(fig)

    return outputs
