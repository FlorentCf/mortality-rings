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
PACKING_FRACTION = 0.26

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
WOOD_BASE = np.array(to_rgb("#b9ad9a"))
WOOD_DARK = np.array(to_rgb("#8d8172"))
OCHRE = np.array(to_rgb("#c58b33"))
RUST = np.array(to_rgb("#a94f2d"))
DRIED_RED = np.array(to_rgb("#7f2630"))
BURGUNDY = np.array(to_rgb("#421827"))
GUIDE = "#fffdf8"
INK = "#201f1b"


@dataclass(frozen=True)
class V3Simulation:
    cells: pd.DataFrame
    outlines: dict[int, np.ndarray]
    inners: dict[int, np.ndarray]
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


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return a * (1 - np.asarray(t)[..., None]) + b * np.asarray(t)[..., None]


def _angular_distance(grid: np.ndarray, theta: float) -> np.ndarray:
    return np.angle(np.exp(1j * (grid - theta)))


def _season_mu(week: int) -> float:
    position = (week - 0.5) / WEEKS_PER_YEAR
    flow = 0.30 * np.sin(2 * np.pi * position * 2.6 + 0.4)
    flow += 0.16 * np.sin(2 * np.pi * position * 5.2 + 1.7)
    return float((2 * np.pi * position + flow) % (2 * np.pi))


def _season_color(weeks: np.ndarray) -> np.ndarray:
    position = ((weeks - 0.5) % WEEKS_PER_YEAR) / WEEKS_PER_YEAR
    return np.column_stack(
        [
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 1]),
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 2]),
            np.interp(position, SEASON_PALETTE[:, 0], SEASON_PALETTE[:, 3]),
        ]
    )


def _interp_profile(profile: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    theta_values = np.asarray(theta)
    position = (theta_values % (2 * np.pi)) / (2 * np.pi) * len(profile)
    left = np.floor(position).astype(int) % len(profile)
    right = (left + 1) % len(profile)
    frac = position - np.floor(position)
    return profile[left] * (1 - frac) + profile[right] * frac


def _profile_tangent_angles(profile: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    smooth_profile = _smooth_circular(profile, passes=24)
    step = 2 * np.pi / len(profile)
    dr = (np.roll(smooth_profile, -1) - np.roll(smooth_profile, 1)) / (2 * step)
    dx = dr * np.sin(theta_grid) + smooth_profile * np.cos(theta_grid)
    dy = dr * np.cos(theta_grid) - smooth_profile * np.sin(theta_grid)
    return np.arctan2(dy, dx)


def _field_from_weeks(
    year_weeks: pd.DataFrame,
    value_column: str,
    theta_grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    field = np.zeros(len(theta_grid), dtype=float)
    for record in year_weeks.itertuples(index=False):
        value = float(getattr(record, value_column))
        if not np.isfinite(value) or value <= 0:
            continue
        mu = _season_mu(int(record.week))
        distance = _angular_distance(theta_grid, mu)
        field += value * np.exp(-0.5 * (distance / sigma) ** 2)
    return _smooth_circular(field, passes=3)


def _normalize_field(field: np.ndarray) -> np.ndarray:
    scale = float(np.nanpercentile(field, 94))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(field)
    return np.clip(field / scale, 0, 1.8)


def _annual_gap(year: int, first_year: int, cell_radius: float) -> float:
    if year == first_year:
        return cell_radius * 0.35
    if year % 5 == 0:
        return cell_radius * 3.7
    return cell_radius * 2.15


def _solve_band_scale(inner: np.ndarray, profile: np.ndarray, target_area: float, cell_radius: float) -> float:
    dtheta = 2 * np.pi / len(inner)
    quadratic = 0.5 * float(np.sum(profile**2) * dtheta)
    linear = float(np.sum(inner * profile) * dtheta)
    if quadratic <= 1e-12:
        scale = target_area / max(linear, 1e-12)
    else:
        scale = (-linear + np.sqrt(linear**2 + 4 * quadratic * target_area)) / (2 * quadratic)
    return max(float(scale), cell_radius * 3.1)


def _build_growth_profile(
    year_weeks: pd.DataFrame,
    theta_grid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mass_field = _field_from_weeks(year_weeks, "deaths", theta_grid, sigma=0.25)
    excess_field = _field_from_weeks(year_weeks, "positive_excess_deaths", theta_grid, sigma=0.16)
    deficit_field = _field_from_weeks(year_weeks, "deficit_deaths", theta_grid, sigma=0.18)

    event = _normalize_field(excess_field)
    deficit = _normalize_field(deficit_field)
    roughness = 0.20 * _smooth_noise(rng, len(theta_grid), passes=36)
    profile = 1.0 + 1.35 * event - 0.48 * deficit + roughness
    profile = _smooth_circular(profile, passes=10)
    profile = np.clip(profile, 0.34, 2.85)
    return profile, mass_field, excess_field, deficit_field


def _make_candidates(year_weeks: pd.DataFrame, people_per_cell: float, rng: np.random.Generator) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for record in year_weeks.itertuples(index=False):
        deaths = float(record.deaths)
        expected = float(record.expected_abs_deaths)
        tissue_people = min(deaths, expected)
        excess_people = max(deaths - expected, 0)
        tissue_count = _stochastic_count(tissue_people / people_per_cell, rng)
        excess_count = _stochastic_count(excess_people / people_per_cell, rng)
        common = {
            "year": int(record.year),
            "week": int(record.week),
            "deaths": deaths,
            "expected_abs_deaths": expected,
            "absolute_excess_deaths": float(record.absolute_excess_deaths),
            "absolute_excess_pct": float(record.absolute_excess_pct) if np.isfinite(record.absolute_excess_pct) else 0.0,
            "seasonal_concentration_pct": float(record.seasonal_concentration_pct)
            if np.isfinite(record.seasonal_concentration_pct)
            else 0.0,
        }
        for _ in range(tissue_count):
            item = dict(common)
            item["role"] = "tissue"
            item["strength"] = 0.0
            item["tone"] = float(rng.random())
            candidates.append(item)
        for _ in range(excess_count):
            item = dict(common)
            item["role"] = "excess"
            item["strength"] = float(np.clip(item["absolute_excess_pct"] / 0.65, 0, 1.8))
            item["tone"] = float(rng.random())
            candidates.append(item)

    rng.shuffle(candidates)
    candidates.sort(key=lambda item: 0 if item["role"] == "excess" else 1)
    return candidates


def _sample_theta_for_candidate(candidate: dict[str, float | int | str], rng: np.random.Generator) -> float:
    role = str(candidate["role"])
    week = int(candidate["week"])
    mu = _season_mu(week)
    if role == "excess":
        kappa = 8.0 + 12.0 * float(candidate["strength"])
        return float(rng.vonmises(mu, kappa) % (2 * np.pi))
    if rng.random() < 0.18:
        return float(rng.uniform(0, 2 * np.pi))
    return float(rng.vonmises(mu, 1.25) % (2 * np.pi))


def _sample_radius(inner: np.ndarray, outer: np.ndarray, theta: float, role: str, cell_radius: float, rng: np.random.Generator) -> float:
    inner_at_theta = float(_interp_profile(inner, theta))
    outer_at_theta = float(_interp_profile(outer, theta))
    width = max(outer_at_theta - inner_at_theta, cell_radius * 2.6)
    margin = min(cell_radius * 1.25, width * 0.32)
    low = inner_at_theta + margin
    high = outer_at_theta - margin
    if high <= low:
        return inner_at_theta + width * 0.5
    if role == "excess":
        u = float(rng.beta(2.1, 1.35))
    else:
        u = float(rng.random())
    return float(np.sqrt(low**2 + u * (high**2 - low**2)))


def _poisson_place_candidates(
    candidates: list[dict[str, float | int | str]],
    inner: np.ndarray,
    outer: np.ndarray,
    theta_grid: np.ndarray,
    cell_radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]], int]:
    min_distance = cell_radius * 2.03
    relaxed_distance = cell_radius * 1.62
    cell_size = min_distance
    bins: dict[tuple[int, int], list[int]] = {}
    positions: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []
    skipped = 0

    def key(xy: np.ndarray) -> tuple[int, int]:
        return int(np.floor(xy[0] / cell_size)), int(np.floor(xy[1] / cell_size))

    def clears(xy: np.ndarray, required_distance: float) -> bool:
        if not positions:
            return True
        key_x, key_y = key(xy)
        neighbor_indexes: list[int] = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                neighbor_indexes.extend(bins.get((key_x + dx, key_y + dy), []))
        if not neighbor_indexes:
            return True
        neighbor_positions = np.vstack([positions[index] for index in neighbor_indexes])
        distance = np.linalg.norm(neighbor_positions - xy, axis=1)
        return bool(np.all(distance >= required_distance))

    def register(xy: np.ndarray) -> None:
        bins.setdefault(key(xy), []).append(len(positions) - 1)

    for candidate in candidates:
        accepted = False
        for attempt in range(220):
            theta = _sample_theta_for_candidate(candidate, rng)
            radius = _sample_radius(inner, outer, theta, str(candidate["role"]), cell_radius, rng)
            xy = _polar_to_xy(np.array([theta]), np.array([radius]))[0]
            required = min_distance if attempt < 150 else relaxed_distance
            if clears(xy, required):
                row = dict(candidate)
                row["theta"] = theta
                row["radial"] = radius
                row["inner_radius"] = float(_interp_profile(inner, theta))
                row["outer_radius"] = float(_interp_profile(outer, theta))
                row["radius"] = cell_radius
                positions.append(xy)
                rows.append(row)
                register(xy)
                accepted = True
                break
        if not accepted:
            skipped += 1

    radii = np.full(len(positions), cell_radius, dtype=float)
    return np.asarray(positions), radii, rows, skipped


def _project_into_band(positions: np.ndarray, inner: np.ndarray, outer: np.ndarray, cell_radius: float) -> None:
    theta = _theta_from_xy(positions)
    radial = np.linalg.norm(positions, axis=1)
    low = _interp_profile(inner, theta) + cell_radius * 1.04
    high = _interp_profile(outer, theta) - cell_radius * 1.04
    center = (low + high) / 2
    low = np.minimum(low, center)
    high = np.maximum(high, center)
    radial = np.clip(radial, low, high)
    positions[:, :] = _polar_to_xy(theta, radial)


def _relax_year_cells(
    positions: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    cell_radius: float,
    rng: np.random.Generator,
    iterations: int,
) -> None:
    if len(positions) < 2:
        _project_into_band(positions, inner, outer, cell_radius)
        return

    min_distance = cell_radius * 2.0
    for _ in range(iterations):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(min_distance * 1.02, output_type="ndarray")
        if len(pairs):
            left = pairs[:, 0]
            right = pairs[:, 1]
            delta = positions[left] - positions[right]
            distance = np.linalg.norm(delta, axis=1)
            zero = distance < 1e-9
            if np.any(zero):
                delta[zero] = rng.normal(0, 1, (int(zero.sum()), 2))
                distance[zero] = np.linalg.norm(delta[zero], axis=1)
            overlap = min_distance - distance
            active = overlap > 0
            if np.any(active):
                left = left[active]
                right = right[active]
                direction = delta[active] / distance[active, None]
                step = overlap[active] * 0.52
                np.add.at(positions, left, direction * step[:, None])
                np.add.at(positions, right, -direction * step[:, None])

        theta = _theta_from_xy(positions)
        tangent = np.column_stack([np.cos(theta), -np.sin(theta)])
        positions += tangent * rng.normal(0, cell_radius * 0.018, positions.shape[0])[:, None]
        _project_into_band(positions, inner, outer, cell_radius)


def _assign_orientations(
    positions: np.ndarray,
    rows: list[dict[str, float | int | str]],
    inner: np.ndarray,
    outer: np.ndarray,
    theta_grid: np.ndarray,
    cell_radius: float,
    rng: np.random.Generator,
    early_year: bool,
) -> None:
    if len(positions) == 0:
        return
    theta = _theta_from_xy(positions)
    radial = np.linalg.norm(positions, axis=1)
    midline = (inner + outer) / 2
    band_angles = _profile_tangent_angles(midline, theta_grid)
    tree = cKDTree(positions)

    for index, row in enumerate(rows):
        grid_index = int(theta[index] / (2 * np.pi) * len(theta_grid)) % len(theta_grid)
        band_vec = np.array([np.cos(band_angles[grid_index]), np.sin(band_angles[grid_index])])
        pca_vec = band_vec.copy()
        neighbors = tree.query_ball_point(positions[index], cell_radius * 6.0)
        if len(neighbors) >= 4:
            local = positions[neighbors] - positions[neighbors].mean(axis=0)
            covariance = local.T @ local
            values, vectors = np.linalg.eigh(covariance)
            pca_vec = vectors[:, int(np.argmax(values))]
            if float(np.dot(pca_vec, band_vec)) < 0:
                pca_vec *= -1
        noise_angle = float(rng.uniform(0, 2 * np.pi))
        noise_vec = np.array([np.cos(noise_angle), np.sin(noise_angle)])
        if early_year:
            final = 0.45 * band_vec + 0.35 * pca_vec + 0.20 * noise_vec
        else:
            final = 0.66 * band_vec + 0.24 * pca_vec + 0.10 * noise_vec
        final /= max(float(np.linalg.norm(final)), 1e-9)
        strength = float(row["strength"])
        row["x"] = float(positions[index, 0])
        row["y"] = float(positions[index, 1])
        row["theta"] = float(theta[index])
        row["radial"] = float(radial[index])
        row["inner_radius"] = float(_interp_profile(inner, theta[index]))
        row["outer_radius"] = float(_interp_profile(outer, theta[index]))
        row["orientation"] = float(np.arctan2(final[1], final[0]))
        row["orientation_jitter"] = float(rng.normal(0, 0.052 if early_year else 0.038))
        row["dash_length"] = float(cell_radius * rng.uniform(1.32, 2.05) * (1 + 0.25 * strength))
        row["line_width"] = float(rng.uniform(0.35, 0.62) * (1 + 0.15 * strength))


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


def simulate_v3_cells(
    weekly: pd.DataFrame,
    first_year: int,
    last_year: int,
    args,
) -> V3Simulation:
    rng = np.random.default_rng(args.seed)
    theta_grid = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    cell_radius = float(args.v3_cell_radius)
    people_per_cell = float(args.v3_people_per_cell)

    previous_outer = cell_radius * (1.05 + 0.28 * _smooth_noise(rng, len(theta_grid), passes=34))
    previous_outer = np.maximum(previous_outer, cell_radius * 0.65)
    all_positions: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_rows: list[dict[str, float | int | str]] = []
    outlines: dict[int, np.ndarray] = {}
    inners: dict[int, np.ndarray] = {}
    skipped_total = 0
    target_cells = 0.0

    visible_weekly = weekly.query("@first_year <= year <= @last_year").copy()
    fill_columns = [
        "deaths",
        "expected_abs_deaths",
        "absolute_excess_deaths",
        "positive_excess_deaths",
        "deficit_deaths",
        "absolute_excess_pct",
        "seasonal_concentration_pct",
    ]
    for column in fill_columns:
        visible_weekly[column] = visible_weekly[column].fillna(0)

    for year, year_weeks in visible_weekly.groupby("year", sort=True):
        year = int(year)
        profile, _mass_field, _excess_field, _deficit_field = _build_growth_profile(year_weeks, theta_grid, rng)
        candidates = _make_candidates(year_weeks, people_per_cell, rng)
        target_cells += float(year_weeks["deaths"].sum()) / people_per_cell
        gap = _annual_gap(year, first_year, cell_radius)
        inner = previous_outer + gap
        target_area = max(len(candidates), 1) * np.pi * cell_radius**2 / PACKING_FRACTION
        scale = _solve_band_scale(inner, profile, target_area, cell_radius)
        outer = inner + scale * profile

        positions, radii, rows, skipped = _poisson_place_candidates(
            candidates,
            inner,
            outer,
            theta_grid,
            cell_radius,
            rng,
        )
        skipped_total += skipped
        if len(positions):
            _relax_year_cells(positions, inner, outer, cell_radius, rng, int(args.v3_relax_iterations))
            _assign_orientations(
                positions,
                rows,
                inner,
                outer,
                theta_grid,
                cell_radius,
                rng,
                early_year=year <= first_year + 2,
            )
            all_positions.append(positions)
            all_radii.append(radii)
            all_rows.extend(rows)

        inners[year] = inner.copy()
        outlines[year] = outer.copy()
        previous_outer = _smooth_circular(outer, passes=3)

    cells = pd.DataFrame(all_rows)
    positions_all = np.vstack(all_positions) if all_positions else np.empty((0, 2))
    radii_all = np.concatenate(all_radii) if all_radii else np.empty(0)
    diagnostics = {
        "cell_count": float(len(cells)),
        "target_cell_count": float(target_cells),
        "skipped_cells": float(skipped_total),
        "max_overlap": float(_measure_max_overlap(positions_all, radii_all)),
    }
    return V3Simulation(cells=cells, outlines=outlines, inners=inners, theta_grid=theta_grid, diagnostics=diagnostics)


def _v3_colors(cells: pd.DataFrame, view: str) -> np.ndarray:
    if cells.empty:
        return np.empty((0, 4))
    role = cells["role"].to_numpy()
    strength = np.clip(cells["strength"].fillna(0).to_numpy(), 0, 1.8)
    tone = cells["tone"].to_numpy()
    season_rgb = _season_color(cells["week"].to_numpy())
    wood = _mix(WOOD_BASE, WOOD_DARK, 0.18 * tone)
    rgb = _mix(wood, season_rgb, 0.20 if view == "art" else 0.26)
    alpha = np.full(len(cells), 0.86 if view == "art" else 0.78)

    excess_mask = role == "excess"
    if np.any(excess_mask):
        local_strength = np.clip(strength[excess_mask], 0, 1)
        mid = _mix(OCHRE, RUST, np.clip(local_strength * 1.25, 0, 1))
        high = _mix(DRIED_RED, BURGUNDY, np.clip((local_strength - 0.45) / 0.55, 0, 1))
        event_rgb = np.where((local_strength < 0.48)[:, None], mid, high)
        rgb[excess_mask] = _mix(rgb[excess_mask], event_rgb, 0.76 if view == "art" else 0.84)
        alpha[excess_mask] = 0.95 if view == "art" else 0.97

    return np.column_stack([np.clip(rgb, 0, 1), alpha])


def _cell_segments(cells: pd.DataFrame) -> np.ndarray:
    positions = cells[["x", "y"]].to_numpy()
    angle = cells["orientation"].to_numpy() + cells["orientation_jitter"].to_numpy()
    tangent = np.column_stack([np.cos(angle), np.sin(angle)])
    length = cells["dash_length"].to_numpy()
    return np.stack([positions - tangent * length[:, None] / 2, positions + tangent * length[:, None] / 2], axis=1)


def _outline_xy(theta_grid: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = _smooth_circular(radius, passes=16)
    closed_theta = np.concatenate([theta_grid, theta_grid[:1]])
    closed_radius = np.concatenate([radius, radius[:1]])
    xy = _polar_to_xy(closed_theta, closed_radius)
    return xy[:, 0], xy[:, 1]


def _max_extent(cells: pd.DataFrame, outlines: dict[int, np.ndarray]) -> float:
    cell_extent = 1.0
    if not cells.empty:
        cell_extent = float(np.nanmax(np.abs(cells[["x", "y"]].to_numpy())))
    outline_extent = max((float(np.nanmax(radius)) for radius in outlines.values()), default=cell_extent)
    return max(cell_extent, outline_extent) * 1.12


def _draw_v3_frame(
    ax: plt.Axes,
    simulation: V3Simulation,
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
                colors=_v3_colors(visible, view),
                linewidths=visible["line_width"].to_numpy(),
                capstyle="round",
            )
        )

    if view == "analytical":
        for outline_year, radius in simulation.outlines.items():
            if outline_year <= year and outline_year % 5 == 0:
                x, y = _outline_xy(simulation.theta_grid, radius)
                ax.plot(x, y, color=GUIDE, linewidth=0.72, alpha=0.72, zorder=4)
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


def _add_v3_analytical_text(
    fig: plt.Figure,
    args,
    first_year: int,
    last_year: int,
    diagnostics: dict[str, float],
) -> None:
    subtitle = args.subtitle or (
        f"Belgian deaths, {first_year}-{last_year}: one V3 cambium cell is about {args.v3_people_per_cell:g} deaths"
    )
    source = args.source_note or f"Source: Statbel open data, Number of deaths per day, {first_year}-{last_year}."
    method = (
        "V3 uses absolute expected mortality, empty annual bands, and excess-driven local growth scars; "
        "deficits reduce band density rather than becoming cells."
    )
    fig.text(0.5, 0.962, args.title, ha="center", va="center", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.5, 0.928, subtitle, ha="center", va="center", fontsize=12.5, color="#635f56")
    fig.text(0.5, 0.060, source, ha="center", va="center", fontsize=9.3, color="#635f56")
    fig.text(0.5, 0.041, method, ha="center", va="center", fontsize=8.7, color="#756f64")
    fig.text(
        0.5,
        0.022,
        (
            f"V3 cells: {diagnostics['cell_count']:.0f}; skipped candidates: {diagnostics['skipped_cells']:.0f}; "
            f"max overlap diagnostic: {diagnostics['max_overlap']:.4f}"
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#8a8275",
    )


def _add_v3_legend(fig: plt.Figure) -> None:
    legend_ax = fig.add_axes([0.34, 0.086, 0.32, 0.020])
    gradient = np.linspace(0, 1, 256)
    left = _mix(OCHRE, RUST, np.clip(gradient * 1.25, 0, 1))
    right = _mix(DRIED_RED, BURGUNDY, np.clip((gradient - 0.45) / 0.55, 0, 1))
    rgb = np.where((gradient < 0.48)[:, None], left, right).reshape(1, 256, 3)
    legend_ax.imshow(rgb, aspect="auto")
    legend_ax.set_yticks([])
    legend_ax.set_xticks([0, 128, 255])
    legend_ax.set_xticklabels(["mild excess", "scar", "extreme"], fontsize=8, color="#635f56")
    for spine in legend_ax.spines.values():
        spine.set_visible(False)


def render_v3_views(
    simulation: V3Simulation,
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
            _add_v3_analytical_text(fig, args, first_year, last_year, simulation.diagnostics)
            _add_v3_legend(fig)

        def update(frame_index: int) -> None:
            _draw_v3_frame(ax, simulation, annual_totals, years[frame_index], max_extent, view)

        update(len(years) - 1)
        png_path = args.output_dir / f"{stem}_v3_{view}.png"
        fig.savefig(png_path, dpi=args.png_dpi)
        outputs.append(png_path)

        if not args.no_gif:
            ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
            gif_path = args.output_dir / f"{stem}_v3_{view}.gif"
            ani.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
            outputs.append(gif_path)

        if not args.no_mp4:
            ani = animation.FuncAnimation(fig, update, frames=len(years), interval=720, repeat=True)
            mp4_path = args.output_dir / f"{stem}_v3_{view}.mp4"
            try:
                ani.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
                outputs.append(mp4_path)
            except Exception as exc:
                print(f"Skipping MP4 export for {view} because ffmpeg failed: {exc}")

        plt.close(fig)

    return outputs
