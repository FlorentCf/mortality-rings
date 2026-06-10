from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


BACKGROUND = "#f7efe3"
WEEKS_PER_YEAR = 52
THETA_POINTS = 1440
PACKING_FRACTION = 0.27

SEASON_PALETTE = np.array(
    [
        [0.00, *to_rgb("#597f8a")],
        [0.16, *to_rgb("#7d9b8e")],
        [0.32, *to_rgb("#9ca56a")],
        [0.49, *to_rgb("#b8934e")],
        [0.66, *to_rgb("#b77a4d")],
        [0.82, *to_rgb("#8b6c67")],
        [1.00, *to_rgb("#597f8a")],
    ]
)
WOOD_BASE = np.array(to_rgb("#9f9a88"))
WOOD_DARK = np.array(to_rgb("#6f7568"))
OCHRE = np.array(to_rgb("#b6782e"))
RUST = np.array(to_rgb("#94402b"))
DRIED_RED = np.array(to_rgb("#69202a"))
BURGUNDY = np.array(to_rgb("#2d111b"))
BLUE_GREY = np.array(to_rgb("#5d7b86"))
GUIDE = "#fff9ed"
INK = "#201f1b"


@dataclass(frozen=True)
class V3Config:
    people_per_cell: float
    cell_radius: float
    relax_iterations: int
    pith_radius: float = 0.11
    annual_gap_factor: float = 0.40
    five_year_gap_factor: float = 1.80
    persistent_noise_strength: float = 0.04
    year_noise_strength: float = 0.03
    scar_geometry_strength: float = 0.85
    annual_stress_strength: float = 0.22
    deficit_geometry_strength: float = 0.08


@dataclass(frozen=True)
class V3Simulation:
    cells: pd.DataFrame
    outlines: dict[int, np.ndarray]
    inners: dict[int, np.ndarray]
    theta_grid: np.ndarray
    diagnostics: dict[str, float]
    fields: pd.DataFrame
    weekly: pd.DataFrame
    pith_radius: float


def _config_from_args(args) -> V3Config:
    return V3Config(
        people_per_cell=float(getattr(args, "v3_people_per_cell", 220)),
        cell_radius=float(getattr(args, "v3_cell_radius", 0.0072)),
        relax_iterations=int(getattr(args, "v3_relax_iterations", 60)),
        pith_radius=float(getattr(args, "v3_pith_radius", 0.11)),
        annual_gap_factor=float(getattr(args, "v3_annual_gap_factor", 0.40)),
        five_year_gap_factor=float(getattr(args, "v3_five_year_gap_factor", 1.80)),
        scar_geometry_strength=float(getattr(args, "v3_scar_geometry_strength", 0.85)),
    )


def _polar_to_xy(theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])


def _theta_from_xy(positions: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(positions[:, 0], positions[:, 1]), 2 * np.pi)


def _gaussian_smooth_circular(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(float).copy()
    return gaussian_filter1d(values.astype(float), sigma=sigma, mode="wrap")


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
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        weight = float(t_arr)
        return a * (1 - weight) + b * weight
    return a * (1 - t_arr[..., None]) + b * t_arr[..., None]


def _angular_distance(grid: np.ndarray, theta: float) -> np.ndarray:
    return np.angle(np.exp(1j * (grid - theta)))


def _season_mu(week: int) -> float:
    position = (week - 0.5) / WEEKS_PER_YEAR
    flow = 0.22 * np.sin(2 * np.pi * position * 2.6 + 0.4)
    flow += 0.12 * np.sin(2 * np.pi * position * 5.2 + 1.7)
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
    smooth_profile = _gaussian_smooth_circular(profile, sigma=len(profile) / WEEKS_PER_YEAR * 0.75)
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
    return _smooth_circular(field, passes=2)


def _normalize_field(field: np.ndarray, percentile: float = 94, clip: float = 1.8) -> np.ndarray:
    finite = field[np.isfinite(field)]
    if len(finite) == 0:
        return np.zeros_like(field)
    scale = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(field)
    return np.clip(field / scale, 0, clip)


def _annual_gap(year: int, first_year: int, config: V3Config) -> float:
    if year == first_year:
        return config.cell_radius * config.annual_gap_factor
    if year % 5 == 0:
        return config.cell_radius * config.five_year_gap_factor
    return config.cell_radius * config.annual_gap_factor


def _solve_band_scale(inner: np.ndarray, profile: np.ndarray, target_area: float, cell_radius: float) -> float:
    dtheta = 2 * np.pi / len(inner)
    quadratic = 0.5 * float(np.sum(profile**2) * dtheta)
    linear = float(np.sum(inner * profile) * dtheta)
    if quadratic <= 1e-12:
        scale = target_area / max(linear, 1e-12)
    else:
        scale = (-linear + np.sqrt(linear**2 + 4 * quadratic * target_area)) / (2 * quadratic)
    return max(float(scale), cell_radius * 3.45)


def _prepare_v3_weekly(weekly: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    out = weekly.query("@first_year <= year <= @last_year").copy()
    for column in [
        "deaths",
        "expected_abs_deaths",
        "absolute_excess_deaths",
        "positive_excess_deaths",
        "deficit_deaths",
        "absolute_excess_pct",
        "seasonal_concentration_pct",
    ]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = out[column].fillna(0)

    if "acute_threshold_deaths" not in out.columns:
        out["acute_threshold_deaths"] = 0.10 * out["expected_abs_deaths"]
    if "acute_excess_deaths" not in out.columns:
        out["acute_excess_deaths"] = (
            out["absolute_excess_deaths"] - out["acute_threshold_deaths"]
        ).clip(lower=0)
    if "acute_excess_pct" not in out.columns:
        out["acute_excess_pct"] = out["acute_excess_deaths"] / out["expected_abs_deaths"].replace(0, np.nan)

    for column in ["acute_threshold_deaths", "acute_excess_deaths", "acute_excess_pct"]:
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def _build_growth_profile(
    year_weeks: pd.DataFrame,
    theta_grid: np.ndarray,
    rng: np.random.Generator,
    persistent_noise: np.ndarray,
    config: V3Config,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    points_per_week = len(theta_grid) / WEEKS_PER_YEAR
    mass_field = _field_from_weeks(year_weeks, "deaths", theta_grid, sigma=0.24)
    acute_field = _field_from_weeks(year_weeks, "acute_excess_deaths", theta_grid, sigma=0.12)
    deficit_field = _field_from_weeks(year_weeks, "deficit_deaths", theta_grid, sigma=0.19)

    acute = _normalize_field(acute_field, percentile=91, clip=1.6)
    deficit = _normalize_field(deficit_field, percentile=94, clip=1.2)
    year_noise = _smooth_noise(rng, len(theta_grid), passes=62)

    expected = max(float(year_weeks["expected_abs_deaths"].sum()), 1.0)
    absolute_excess = float(year_weeks["absolute_excess_deaths"].sum())
    annual_stress = float(np.clip(max(absolute_excess, 0.0) / expected / 0.20, 0, 1.2))

    profile = np.ones(len(theta_grid), dtype=float)
    profile += config.persistent_noise_strength * persistent_noise
    profile += config.year_noise_strength * year_noise
    profile += config.scar_geometry_strength * acute
    profile += config.annual_stress_strength * annual_stress
    profile -= config.deficit_geometry_strength * deficit
    profile = _gaussian_smooth_circular(profile, sigma=points_per_week * 1.8)
    profile = np.clip(profile, 0.70, 1.85)

    fields: dict[str, np.ndarray | float] = {
        "mass_field": mass_field,
        "acute_field": acute_field,
        "deficit_field": deficit_field,
        "acute_field_norm": acute,
        "deficit_field_norm": deficit,
        "annual_stress": annual_stress,
    }
    return profile, fields


def _make_candidates(year_weeks: pd.DataFrame, people_per_cell: float, rng: np.random.Generator) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for record in year_weeks.itertuples(index=False):
        deaths = float(record.deaths)
        acute = min(max(float(record.acute_excess_deaths), 0.0), max(deaths, 0.0))
        tissue_people = max(deaths - acute, 0.0)
        tissue_count = _stochastic_count(tissue_people / people_per_cell, rng)
        excess_count = _stochastic_count(acute / people_per_cell, rng)
        expected = float(record.expected_abs_deaths)
        deficit_pct = float(record.deficit_deaths / expected) if expected > 0 else 0.0
        common = {
            "year": int(record.year),
            "week": int(record.week),
            "deaths": deaths,
            "expected_abs_deaths": expected,
            "absolute_excess_deaths": float(record.absolute_excess_deaths),
            "absolute_excess_pct": float(record.absolute_excess_pct) if np.isfinite(record.absolute_excess_pct) else 0.0,
            "acute_excess_deaths": acute,
            "acute_excess_pct": float(record.acute_excess_pct) if np.isfinite(record.acute_excess_pct) else 0.0,
            "acute_threshold_deaths": float(record.acute_threshold_deaths),
            "deficit_deaths": float(record.deficit_deaths),
            "deficit_pct": deficit_pct if np.isfinite(deficit_pct) else 0.0,
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
            item["strength"] = float(np.clip(item["acute_excess_pct"] / 0.32, 0, 1.8))
            item["tone"] = float(rng.random())
            candidates.append(item)

    rng.shuffle(candidates)
    candidates.sort(key=lambda item: (0 if item["role"] == "excess" else 1, -float(item["strength"])))
    return candidates


def _sample_theta_for_candidate(candidate: dict[str, float | int | str], rng: np.random.Generator) -> float:
    role = str(candidate["role"])
    week = int(candidate["week"])
    mu = _season_mu(week)
    if role == "excess":
        kappa = 10.0 + 20.0 * float(candidate["strength"])
        return float(rng.vonmises(mu, kappa) % (2 * np.pi))
    if rng.random() < 0.26:
        return float(rng.uniform(0, 2 * np.pi))
    return float(rng.vonmises(mu, 0.95) % (2 * np.pi))


def _sample_radius(inner: np.ndarray, outer: np.ndarray, theta: float, role: str, cell_radius: float, rng: np.random.Generator) -> float:
    inner_at_theta = float(_interp_profile(inner, theta))
    outer_at_theta = float(_interp_profile(outer, theta))
    width = max(outer_at_theta - inner_at_theta, cell_radius * 2.8)
    margin = min(cell_radius * 0.95, width * 0.34)
    low = inner_at_theta + margin
    high = outer_at_theta - margin
    if high <= low:
        return inner_at_theta + width * 0.5
    if role == "excess":
        u = float(rng.beta(2.4, 1.20))
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
    del theta_grid
    min_distance = cell_radius * 2.05
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
        for attempt in range(260):
            theta = _sample_theta_for_candidate(candidate, rng)
            radius = _sample_radius(inner, outer, theta, str(candidate["role"]), cell_radius, rng)
            xy = _polar_to_xy(np.array([theta]), np.array([radius]))[0]
            required = min_distance if attempt < 175 else relaxed_distance
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
    low = _interp_profile(inner, theta) + cell_radius * 0.95
    high = _interp_profile(outer, theta) - cell_radius * 0.95
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
                step = overlap[active] * 0.55
                np.add.at(positions, left, direction * step[:, None])
                np.add.at(positions, right, -direction * step[:, None])

        theta = _theta_from_xy(positions)
        tangent = np.column_stack([np.cos(theta), -np.sin(theta)])
        positions += tangent * rng.normal(0, cell_radius * 0.022, positions.shape[0])[:, None]
        _project_into_band(positions, inner, outer, cell_radius)


def _assign_orientations(
    positions: np.ndarray,
    rows: list[dict[str, float | int | str]],
    inner: np.ndarray,
    outer: np.ndarray,
    theta_grid: np.ndarray,
    cell_radius: float,
    rng: np.random.Generator,
) -> None:
    if len(positions) == 0:
        return
    theta = _theta_from_xy(positions)
    radial = np.linalg.norm(positions, axis=1)
    midline = (inner + outer) / 2
    band_angles = _profile_tangent_angles(midline, theta_grid)
    curl_field = _smooth_noise(rng, len(theta_grid), passes=28)
    tree = cKDTree(positions)

    for index, row in enumerate(rows):
        grid_index = int(theta[index] / (2 * np.pi) * len(theta_grid)) % len(theta_grid)
        band_angle = float(band_angles[grid_index])
        band_vec = np.array([np.cos(band_angle), np.sin(band_angle)])
        pca_vec = band_vec.copy()
        neighbors = tree.query_ball_point(positions[index], cell_radius * 6.3)
        if len(neighbors) >= 4:
            local = positions[neighbors] - positions[neighbors].mean(axis=0)
            covariance = local.T @ local
            values, vectors = np.linalg.eigh(covariance)
            pca_vec = vectors[:, int(np.argmax(values))]
            if float(np.dot(pca_vec, band_vec)) < 0:
                pca_vec *= -1
        curl_angle = band_angle + 1.7 * float(curl_field[grid_index]) + 0.65 * np.sin(2.0 * theta[index] + 4.0 * radial[index])
        noise_vec = np.array([np.cos(curl_angle), np.sin(curl_angle)])
        final = 0.55 * band_vec + 0.25 * pca_vec + 0.20 * noise_vec
        final /= max(float(np.linalg.norm(final)), 1e-9)

        strength = float(row["strength"])
        is_excess = row["role"] == "excess"
        base_length = cell_radius * rng.uniform(1.42, 2.12)
        if is_excess:
            length = base_length * (1.35 + 0.75 * np.clip(strength, 0, 1))
            line_width = rng.uniform(0.55, 0.82) * (1.10 + 0.25 * np.clip(strength, 0, 1))
            jitter = rng.normal(0, 0.48 + 0.06 * np.clip(strength, 0, 1))
        else:
            length = base_length
            line_width = rng.uniform(0.48, 0.74)
            jitter = rng.normal(0, 0.32)

        row["x"] = float(positions[index, 0])
        row["y"] = float(positions[index, 1])
        row["theta"] = float(theta[index])
        row["radial"] = float(radial[index])
        row["inner_radius"] = float(_interp_profile(inner, theta[index]))
        row["outer_radius"] = float(_interp_profile(outer, theta[index]))
        row["orientation"] = float(np.arctan2(final[1], final[0]))
        row["orientation_jitter"] = float(jitter)
        row["dash_length"] = float(length)
        row["line_width"] = float(line_width)


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
    theta_grid = np.linspace(0, 2 * np.pi, THETA_POINTS, endpoint=False)
    config = _config_from_args(args)
    points_per_week = len(theta_grid) / WEEKS_PER_YEAR

    persistent_noise = _smooth_noise(rng, len(theta_grid), passes=120)
    previous_outer = np.full(len(theta_grid), config.pith_radius)
    previous_outer += config.pith_radius * 0.010 * _smooth_noise(rng, len(theta_grid), passes=80)
    previous_outer = np.maximum(previous_outer, config.pith_radius * 0.94)

    all_positions: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_rows: list[dict[str, float | int | str]] = []
    field_rows: list[pd.DataFrame] = []
    outlines: dict[int, np.ndarray] = {}
    inners: dict[int, np.ndarray] = {}
    skipped_total = 0
    target_cells = 0.0

    visible_weekly = _prepare_v3_weekly(weekly, first_year, last_year)

    for year, year_weeks in visible_weekly.groupby("year", sort=True):
        year = int(year)
        profile, fields = _build_growth_profile(year_weeks, theta_grid, rng, persistent_noise, config)
        candidates = _make_candidates(year_weeks, config.people_per_cell, rng)
        target_cells += float(year_weeks["deaths"].sum()) / config.people_per_cell
        gap = _annual_gap(year, first_year, config)
        inner = previous_outer + gap
        annual_stress = float(fields["annual_stress"])
        target_area = max(len(candidates), 1) * np.pi * config.cell_radius**2 / PACKING_FRACTION
        target_area *= 1.0 + 0.18 * annual_stress
        scale = _solve_band_scale(inner, profile, target_area, config.cell_radius)
        outer = inner + scale * profile

        positions, radii, rows, skipped = _poisson_place_candidates(
            candidates,
            inner,
            outer,
            theta_grid,
            config.cell_radius,
            rng,
        )
        skipped_total += skipped
        if len(positions):
            _relax_year_cells(positions, inner, outer, config.cell_radius, rng, config.relax_iterations)
            _assign_orientations(positions, rows, inner, outer, theta_grid, config.cell_radius, rng)
            all_positions.append(positions)
            all_radii.append(radii)
            all_rows.extend(rows)

        field_rows.append(
            pd.DataFrame(
                {
                    "year": year,
                    "theta_index": np.arange(len(theta_grid)),
                    "theta": theta_grid,
                    "growth_profile": profile,
                    "acute_field": fields["acute_field"],
                    "acute_field_norm": fields["acute_field_norm"],
                    "deficit_field_norm": fields["deficit_field_norm"],
                    "mass_field": fields["mass_field"],
                    "annual_stress": float(fields["annual_stress"]),
                    "inner_radius": inner,
                    "outer_radius": outer,
                }
            )
        )
        inners[year] = inner.copy()
        outlines[year] = outer.copy()
        previous_outer = _gaussian_smooth_circular(outer, sigma=points_per_week * 0.35)

    cells = pd.DataFrame(all_rows)
    positions_all = np.vstack(all_positions) if all_positions else np.empty((0, 2))
    radii_all = np.concatenate(all_radii) if all_radii else np.empty(0)
    fields = pd.concat(field_rows, ignore_index=True) if field_rows else pd.DataFrame()
    diagnostics = {
        "cell_count": float(len(cells)),
        "target_cell_count": float(target_cells),
        "skipped_cells": float(skipped_total),
        "max_overlap": float(_measure_max_overlap(positions_all, radii_all)),
        "pith_radius": float(config.pith_radius),
        "acute_excess_cells": float((cells["role"] == "excess").sum()) if not cells.empty else 0.0,
    }
    return V3Simulation(
        cells=cells,
        outlines=outlines,
        inners=inners,
        theta_grid=theta_grid,
        diagnostics=diagnostics,
        fields=fields,
        weekly=visible_weekly,
        pith_radius=config.pith_radius,
    )


def _v3_colors(cells: pd.DataFrame, view: str) -> np.ndarray:
    if cells.empty:
        return np.empty((0, 4))
    role = cells["role"].to_numpy()
    strength = np.clip(cells["strength"].fillna(0).to_numpy(), 0, 1.8)
    tone = cells["tone"].to_numpy()
    deficit = np.clip(cells.get("deficit_pct", pd.Series(0, index=cells.index)).fillna(0).to_numpy() / 0.22, 0, 1)
    season_rgb = _season_color(cells["week"].to_numpy())
    wood = _mix(WOOD_BASE, WOOD_DARK, 0.22 * tone)
    rgb = _mix(wood, season_rgb, 0.16 if view == "art" else 0.22)
    alpha = np.full(len(cells), 0.42 if view == "art" else 0.50)
    alpha -= 0.10 * deficit

    excess_mask = role == "excess"
    if np.any(excess_mask):
        local_strength = np.clip(strength[excess_mask], 0, 1)
        mid = _mix(OCHRE, RUST, np.clip(local_strength * 1.30, 0, 1))
        high = _mix(DRIED_RED, BURGUNDY, np.clip((local_strength - 0.35) / 0.65, 0, 1))
        event_rgb = np.where((local_strength < 0.42)[:, None], mid, high)
        rgb[excess_mask] = _mix(rgb[excess_mask], event_rgb, 0.86 if view == "art" else 0.90)
        alpha[excess_mask] = 0.88 + 0.09 * local_strength

    return np.column_stack([np.clip(rgb, 0, 1), np.clip(alpha, 0.12, 0.98)])


def _cell_segments(cells: pd.DataFrame) -> np.ndarray:
    positions = cells[["x", "y"]].to_numpy()
    angle = cells["orientation"].to_numpy() + cells["orientation_jitter"].to_numpy()
    tangent = np.column_stack([np.cos(angle), np.sin(angle)])
    length = cells["dash_length"].to_numpy()
    return np.stack([positions - tangent * length[:, None] / 2, positions + tangent * length[:, None] / 2], axis=1)


def _outline_xy(theta_grid: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = _gaussian_smooth_circular(radius, sigma=len(radius) / WEEKS_PER_YEAR * 0.75)
    closed_theta = np.concatenate([theta_grid, theta_grid[:1]])
    closed_radius = np.concatenate([radius, radius[:1]])
    xy = _polar_to_xy(closed_theta, closed_radius)
    return xy[:, 0], xy[:, 1]


def _max_extent(cells: pd.DataFrame, outlines: dict[int, np.ndarray], pith_radius: float) -> float:
    cell_extent = pith_radius
    if not cells.empty:
        cell_extent = float(np.nanmax(np.abs(cells[["x", "y"]].to_numpy())))
    outline_extent = max((float(np.nanmax(radius)) for radius in outlines.values()), default=cell_extent)
    return max(cell_extent, outline_extent) * 1.13


def _draw_tree_frame(
    ax: plt.Axes,
    simulation: V3Simulation,
    annual_totals: pd.Series,
    year: int,
    max_extent: float,
    view: str,
    *,
    color_mode: str = "default",
    show_guides: bool = False,
    show_year_label: bool = False,
) -> None:
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)

    visible = simulation.cells[simulation.cells["year"] <= year]
    if not visible.empty:
        if color_mode == "year":
            values = (visible["year"].to_numpy() - visible["year"].min()) / max(1, visible["year"].max() - visible["year"].min())
            colors = plt.get_cmap("cividis")(values)
            colors[:, 3] = 0.72
        elif color_mode == "role":
            colors = np.tile(np.array([*to_rgb("#8d9786"), 0.44]), (len(visible), 1))
            excess = visible["role"].to_numpy() == "excess"
            colors[excess] = np.array([*BURGUNDY, 0.96])
        else:
            colors = _v3_colors(visible, view)
        ax.add_collection(
            LineCollection(
                _cell_segments(visible),
                colors=colors,
                linewidths=visible["line_width"].to_numpy(),
                capstyle="round",
            )
        )

    if show_guides:
        for outline_year, radius in simulation.outlines.items():
            if outline_year <= year and outline_year % 5 == 0:
                x, y = _outline_xy(simulation.theta_grid, radius)
                ax.plot(x, y, color=GUIDE, linewidth=0.66, alpha=0.62, zorder=4)

    if show_year_label:
        latest_total = int(annual_totals.loc[year])
        ax.text(
            -max_extent * 0.93,
            -max_extent * 0.96,
            f"{year}  |  {latest_total:,} deaths",
            ha="left",
            va="center",
            fontsize=9.5,
            color="#5b554b",
        )
        if simulation.inners:
            first_year = min(simulation.inners)
            first_inner = simulation.inners[first_year]
            theta = np.deg2rad(245)
            radius = float(_interp_profile(first_inner, theta)) + simulation.pith_radius * 0.28
            xy = _polar_to_xy(np.array([theta]), np.array([radius]))[0]
            ax.text(xy[0], xy[1], str(first_year), ha="center", va="center", fontsize=7.5, color="#80786a")

    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)


def _annual_summary(simulation: V3Simulation, annual_totals: pd.Series) -> pd.DataFrame:
    grouped = (
        simulation.weekly.groupby("year", as_index=False)
        .agg(
            expected_abs_deaths=("expected_abs_deaths", "sum"),
            absolute_excess_deaths=("absolute_excess_deaths", "sum"),
            acute_excess_deaths=("acute_excess_deaths", "sum"),
        )
        .rename(columns={"year": "year"})
    )
    grouped["observed_deaths"] = grouped["year"].map(annual_totals.to_dict())
    return grouped


def _weekly_matrix(simulation: V3Simulation, column: str, current_year: int | None = None) -> tuple[np.ndarray, list[int]]:
    weekly = simulation.weekly.copy()
    if current_year is not None:
        weekly.loc[weekly["year"] > current_year, column] = np.nan
    years = sorted(weekly["year"].unique())
    matrix = (
        weekly.pivot_table(index="year", columns="week", values=column, aggfunc="sum")
        .reindex(index=years, columns=range(1, WEEKS_PER_YEAR + 1))
        .to_numpy(dtype=float)
    )
    return matrix, years


def _draw_analytical_panel(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    simulation: V3Simulation,
    annual_totals: pd.Series,
    args,
    first_year: int,
    last_year: int,
    year: int,
    max_extent: float,
) -> None:
    for axis in axes.values():
        axis.clear()
    _draw_tree_frame(
        axes["tree"],
        simulation,
        annual_totals,
        year,
        max_extent,
        "analytical",
        show_guides=True,
        show_year_label=True,
    )

    annual = _annual_summary(simulation, annual_totals)
    past = annual[annual["year"] <= year]
    ax = axes["annual"]
    ax.plot(past["year"], past["observed_deaths"], color="#4f5852", linewidth=2.0, label="observed")
    ax.plot(past["year"], past["expected_abs_deaths"], color="#b48349", linewidth=1.8, label="expected")
    ax.fill_between(
        past["year"],
        past["expected_abs_deaths"],
        past["observed_deaths"],
        where=past["observed_deaths"] >= past["expected_abs_deaths"],
        color="#8f2d2c",
        alpha=0.22,
    )
    ax.scatter([year], [float(annual_totals.loc[year])], s=32, color="#2d111b", zorder=3)
    ax.set_title("Observed vs expected annual deaths", loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlim(first_year, last_year)
    ax.grid(axis="y", color="#ded3c4", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=8, colors="#5d564c")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    matrix, years = _weekly_matrix(simulation, "acute_excess_pct", current_year=year)
    heat_ax = axes["heatmap"]
    cmap = LinearSegmentedColormap.from_list("acute_wood_blood", ["#f8efe2", "#d7a856", "#93402b", "#2d111b"])
    cmap.set_bad("#fbf6ee")
    heat_ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=max(0.45, float(np.nanpercentile(matrix, 98)) if np.isfinite(matrix).any() else 0.45),
        extent=[1, WEEKS_PER_YEAR, years[0] - 0.5, years[-1] + 0.5],
    )
    heat_ax.axhline(year + 0.5, color="#2d111b", linewidth=0.8, alpha=0.45)
    heat_ax.set_title("Acute weekly excess driving scars", loc="left", fontsize=11, color=INK, pad=8)
    heat_ax.set_xlabel("week of year", fontsize=8, color="#5d564c")
    heat_ax.set_ylabel("year", fontsize=8, color="#5d564c")
    heat_ax.set_xticks([1, 13, 26, 39, 52])
    heat_ax.tick_params(labelsize=8, colors="#5d564c")
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    legend_ax = axes["legend"]
    legend_ax.axis("off")
    legend_ax.set_facecolor(BACKGROUND)
    legend_ax.text(0.0, 0.88, args.title, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    subtitle = args.subtitle or (
        f"Belgian deaths, {first_year}-{last_year}: one V3.1 cambium cell is about {args.v3_people_per_cell:g} deaths"
    )
    legend_ax.text(0.0, 0.62, subtitle, ha="left", va="top", fontsize=8.8, color="#5d564c", wrap=True)
    method = (
        "Observed deaths set cell count and annual band area. Expected deaths form quiet tissue. "
        "Only acute weekly excess above a robust threshold becomes scar tissue; deficits stay mostly as absence."
    )
    legend_ax.text(0.0, 0.36, method, ha="left", va="top", fontsize=7.8, color="#6f665a", wrap=True)
    source = args.source_note or f"Source: Statbel open data, Number of deaths per day, {first_year}-{last_year}."
    legend_ax.text(0.0, 0.06, source, ha="left", va="bottom", fontsize=7.3, color="#80776a", wrap=True)

    fig.suptitle("", y=0.995)


def _save_tree_png(
    path: Path,
    simulation: V3Simulation,
    annual_totals: pd.Series,
    year: int,
    max_extent: float,
    *,
    color_mode: str = "default",
    show_guides: bool = False,
    dpi: int = 180,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor(BACKGROUND)
    _draw_tree_frame(
        ax,
        simulation,
        annual_totals,
        year,
        max_extent,
        "art",
        color_mode=color_mode,
        show_guides=show_guides,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


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
    max_extent = _max_extent(simulation.cells, simulation.outlines, simulation.pith_radius)
    outputs: list[Path] = []

    for view in views:
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND})
        if view == "art":
            fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=150)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            def update(frame_index: int) -> None:
                _draw_tree_frame(ax, simulation, annual_totals, years[frame_index], max_extent, view)

        else:
            fig = plt.figure(figsize=(14.8, 9.2), dpi=145)
            fig.patch.set_facecolor(BACKGROUND)
            gs = fig.add_gridspec(
                3,
                2,
                width_ratios=[1.13, 0.87],
                height_ratios=[0.42, 0.38, 0.20],
                left=0.035,
                right=0.975,
                top=0.965,
                bottom=0.060,
                wspace=0.12,
                hspace=0.32,
            )
            axes = {
                "tree": fig.add_subplot(gs[:, 0]),
                "annual": fig.add_subplot(gs[0, 1]),
                "heatmap": fig.add_subplot(gs[1, 1]),
                "legend": fig.add_subplot(gs[2, 1]),
            }

            def update(frame_index: int) -> None:
                _draw_analytical_panel(
                    fig,
                    axes,
                    simulation,
                    annual_totals,
                    args,
                    first_year,
                    last_year,
                    years[frame_index],
                    max_extent,
                )

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


def _save_profile_heatmap(
    path: Path,
    matrix: np.ndarray,
    years: list[int],
    title: str,
    cmap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    dpi: int = 180,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=130)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=[1, matrix.shape[1], years[0] - 0.5, years[-1] + 0.5],
    )
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_xlabel("angle / week-like seasonal position", color="#5d564c")
    ax.set_ylabel("year", color="#5d564c")
    ax.tick_params(colors="#5d564c")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def render_v3_debug_fields(
    simulation: V3Simulation,
    annual_totals: pd.Series,
    first_year: int,
    last_year: int,
    args,
    stem: str,
) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    max_extent = _max_extent(simulation.cells, simulation.outlines, simulation.pith_radius)
    final_year = last_year

    outlines_path = args.output_dir / f"{stem}_v3_debug_outlines.png"
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)
    circle = plt.Circle((0, 0), simulation.pith_radius, edgecolor="#d3c6b5", facecolor="none", linewidth=1.0, alpha=0.75)
    ax.add_patch(circle)
    for year, radius in simulation.outlines.items():
        if year > final_year:
            continue
        x, y = _outline_xy(simulation.theta_grid, radius)
        ax.plot(x, y, color="#7d756b" if year % 5 == 0 else "#c8bba9", linewidth=0.95 if year % 5 == 0 else 0.42, alpha=0.80)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    fig.savefig(outlines_path, dpi=args.png_dpi)
    plt.close(fig)
    outputs.append(outlines_path)

    year_path = args.output_dir / f"{stem}_v3_debug_cells_by_year.png"
    _save_tree_png(year_path, simulation, annual_totals, final_year, max_extent, color_mode="year", dpi=args.png_dpi)
    outputs.append(year_path)

    role_path = args.output_dir / f"{stem}_v3_debug_cells_by_role.png"
    _save_tree_png(role_path, simulation, annual_totals, final_year, max_extent, color_mode="role", dpi=args.png_dpi)
    outputs.append(role_path)

    years = sorted(simulation.fields["year"].unique())
    profile_matrix = (
        simulation.fields.pivot_table(index="year", columns="theta_index", values="growth_profile", aggfunc="mean")
        .reindex(index=years)
        .to_numpy(dtype=float)
    )
    profile_path = args.output_dir / f"{stem}_v3_debug_growth_profile.png"
    _save_profile_heatmap(profile_path, profile_matrix, years, "V3.1 growth profile by year and angle", "magma", vmin=0.70, vmax=1.85)
    outputs.append(profile_path)

    absolute_matrix, heat_years = _weekly_matrix(simulation, "absolute_excess_pct")
    abs_limit = max(0.35, float(np.nanpercentile(np.abs(absolute_matrix), 98)) if np.isfinite(absolute_matrix).any() else 0.35)
    absolute_path = args.output_dir / f"{stem}_v3_debug_absolute_excess_heatmap.png"
    diverging = LinearSegmentedColormap.from_list("deficit_to_excess", ["#46616b", "#f7efe3", "#a43c2c", "#2d111b"])
    _save_profile_heatmap(
        absolute_path,
        absolute_matrix,
        heat_years,
        "Absolute excess mortality by year and week",
        diverging,
        norm=TwoSlopeNorm(vmin=-abs_limit, vcenter=0, vmax=abs_limit),
    )
    outputs.append(absolute_path)

    acute_matrix, acute_years = _weekly_matrix(simulation, "acute_excess_pct")
    acute_path = args.output_dir / f"{stem}_v3_debug_acute_excess_heatmap.png"
    acute_cmap = LinearSegmentedColormap.from_list("acute_debug", ["#f7efe3", "#d3a753", "#8f2d2c", "#2d111b"])
    _save_profile_heatmap(
        acute_path,
        acute_matrix,
        acute_years,
        "Acute weekly excess used for scar cells",
        acute_cmap,
        vmin=0,
        vmax=max(0.45, float(np.nanpercentile(acute_matrix, 98)) if np.isfinite(acute_matrix).any() else 0.45),
    )
    outputs.append(acute_path)

    return outputs


def _render_sweep_thumbnail(
    path: Path,
    simulation: V3Simulation,
    annual_totals: pd.Series,
    final_year: int,
    label: str,
) -> None:
    max_extent = _max_extent(simulation.cells, simulation.outlines, simulation.pith_radius)
    fig, ax = plt.subplots(figsize=(3.1, 3.1), dpi=120)
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
    fig.patch.set_facecolor(BACKGROUND)
    _draw_tree_frame(ax, simulation, annual_totals, final_year, max_extent, "art")
    fig.text(0.5, 0.975, label, ha="center", va="top", fontsize=6.5, color="#5d564c")
    fig.savefig(path, dpi=125)
    plt.close(fig)


def render_v3_preset_sweep(
    weekly: pd.DataFrame,
    annual_totals: pd.Series,
    first_year: int,
    last_year: int,
    args,
    stem: str,
) -> list[Path]:
    if args.v3_preset_sweep != "scar-gap-texture":
        return []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, float]] = []
    for people in [180, 220, 280, 300]:
        for annual_gap, five_gap in [(0.25, 1.4), (0.40, 1.8), (0.60, 2.2)]:
            for scar_strength in [0.55, 0.85]:
                variants.append(
                    {
                        "people_per_cell": people,
                        "annual_gap_factor": annual_gap,
                        "five_year_gap_factor": five_gap,
                        "scar_geometry_strength": scar_strength,
                    }
                )

    outputs: list[Path] = []
    manifest_rows: list[dict[str, float | str | int]] = []
    thumbnails: list[Path] = []
    for index, variant in enumerate(variants, start=1):
        sweep_args = SimpleNamespace(**vars(args))
        sweep_args.v3_people_per_cell = variant["people_per_cell"]
        sweep_args.v3_annual_gap_factor = variant["annual_gap_factor"]
        sweep_args.v3_five_year_gap_factor = variant["five_year_gap_factor"]
        sweep_args.v3_scar_geometry_strength = variant["scar_geometry_strength"]
        sweep_args.v3_relax_iterations = min(int(args.v3_relax_iterations), 34)
        simulation = simulate_v3_cells(weekly, first_year, last_year, sweep_args)
        label = (
            f"{index:02d} p{variant['people_per_cell']:.0f} "
            f"g{variant['annual_gap_factor']:.2f}/{variant['five_year_gap_factor']:.1f} "
            f"s{variant['scar_geometry_strength']:.2f}"
        )
        thumb_path = args.output_dir / f"{stem}_v3_sweep_{index:02d}.png"
        _render_sweep_thumbnail(thumb_path, simulation, annual_totals, last_year, label)
        thumbnails.append(thumb_path)
        outputs.append(thumb_path)
        manifest_rows.append(
            {
                "variant": index,
                "thumbnail": thumb_path.name,
                **variant,
                "cell_count": int(simulation.diagnostics["cell_count"]),
                "acute_excess_cells": int(simulation.diagnostics["acute_excess_cells"]),
                "max_overlap": simulation.diagnostics["max_overlap"],
            }
        )

    images = [Image.open(path).convert("RGB") for path in thumbnails]
    width, height = images[0].size
    columns = 6
    rows = int(np.ceil(len(images) / columns))
    background_rgb = tuple(int(round(channel * 255)) for channel in to_rgb(BACKGROUND))
    sheet = Image.new("RGB", (columns * width, rows * height), background_rgb)
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(image, (x, y))
    sheet_path = args.output_dir / f"{stem}_v3_sweep_contact_sheet.png"
    sheet.save(sheet_path)
    outputs.append(sheet_path)
    for image in images:
        image.close()

    manifest_path = args.output_dir / f"{stem}_v3_sweep_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    outputs.append(manifest_path)
    return outputs
