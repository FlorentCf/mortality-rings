from __future__ import annotations

import hashlib
import unittest
from argparse import Namespace

import numpy as np
import pandas as pd

from mortality_rings.v3 import _season_mu, simulate_v3_cells


YEARS = list(range(2000, 2006))
BASE_WEEKLY_DEATHS = 1000.0


def _weekly_synthetic(
    *,
    spike: tuple[int, int, float] | None = None,
    wave: tuple[int, range, float] | None = None,
    uniform_high_year: int | None = None,
) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for week in range(1, 53):
            expected = BASE_WEEKLY_DEATHS
            deaths = expected
            if uniform_high_year == year:
                deaths *= 1.10
            if spike and spike[0] == year and spike[1] == week:
                deaths += spike[2]
            if wave and wave[0] == year and week in wave[1]:
                deaths += wave[2]
            absolute_excess = deaths - expected
            threshold = max(0.0, 0.10 * expected)
            acute = max(absolute_excess - threshold, 0.0)
            rows.append(
                {
                    "year": year,
                    "week": week,
                    "deaths": deaths,
                    "baseline_median": expected,
                    "baseline_share": 1.0 / 52,
                    "expected_share_deaths": deaths,
                    "seasonal_concentration_pct": 0.0,
                    "expected_annual_deaths": expected * 52,
                    "expected_abs_deaths": expected,
                    "absolute_excess_deaths": absolute_excess,
                    "positive_excess_deaths": max(absolute_excess, 0.0),
                    "deficit_deaths": max(-absolute_excess, 0.0),
                    "absolute_excess_pct": absolute_excess / expected,
                    "weekly_mad_sigma": 0.0,
                    "acute_threshold_deaths": threshold,
                    "acute_excess_deaths": acute,
                    "acute_excess_pct": acute / expected,
                    "expected_deaths": deaths,
                    "seasonal_excess_pct": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _args(seed: int = 7) -> Namespace:
    return Namespace(
        seed=seed,
        v3_people_per_cell=180,
        v3_cell_radius=0.0062,
        v3_relax_iterations=22,
        v3_pith_radius=0.11,
    )


def _cell_hash(cells: pd.DataFrame) -> str:
    cols = ["year", "week", "role", "x", "y", "radial", "theta", "orientation", "dash_length", "line_width"]
    payload = cells[cols].sort_values(["year", "week", "role", "x", "y"]).round(8).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


class V3SyntheticTests(unittest.TestCase):
    def test_flat_mortality_has_no_scars_and_keeps_empty_pith(self) -> None:
        simulation = simulate_v3_cells(_weekly_synthetic(), YEARS[0], YEARS[-1], _args())

        self.assertNotIn("excess", set(simulation.cells["role"]))
        self.assertGreater(float(simulation.cells["radial"].min()), simulation.pith_radius)
        final_outline = simulation.outlines[YEARS[-1]]
        self.assertLess(float(np.ptp(final_outline) / np.mean(final_outline)), 0.35)

    def test_one_week_spike_creates_local_scar_at_seasonal_angle(self) -> None:
        spike_year = 2003
        spike_week = 34
        simulation = simulate_v3_cells(
            _weekly_synthetic(spike=(spike_year, spike_week, 2200.0)),
            YEARS[0],
            YEARS[-1],
            _args(seed=10),
        )
        excess = simulation.cells[(simulation.cells["role"] == "excess") & (simulation.cells["year"] == spike_year)]

        self.assertGreater(len(excess), 5)
        self.assertEqual(set(excess["week"]), {spike_week})
        distance = _angular_distance(excess["theta"].to_numpy(), _season_mu(spike_week))
        self.assertLess(float(np.median(distance)), 0.32)

    def test_long_wave_creates_broad_scar_sector(self) -> None:
        wave_year = 2003
        high_weeks = range(18, 28)
        simulation = simulate_v3_cells(
            _weekly_synthetic(wave=(wave_year, high_weeks, 1500.0)),
            YEARS[0],
            YEARS[-1],
            _args(seed=11),
        )
        excess = simulation.cells[(simulation.cells["role"] == "excess") & (simulation.cells["year"] == wave_year)]
        field = simulation.fields[simulation.fields["year"] == wave_year]["acute_field_norm"].to_numpy()

        self.assertGreaterEqual(excess["week"].nunique(), 8)
        self.assertGreater(float((field > 0.35).mean()), 0.12)

    def test_uniform_high_year_thickens_ring_without_acute_scar_cells(self) -> None:
        high_year = 2004
        simulation = simulate_v3_cells(
            _weekly_synthetic(uniform_high_year=high_year),
            YEARS[0],
            YEARS[-1],
            _args(seed=12),
        )
        baseline = simulate_v3_cells(_weekly_synthetic(), YEARS[0], YEARS[-1], _args(seed=12))
        high_cells = simulation.cells[simulation.cells["year"] == high_year]
        widths = {year: float(np.mean(simulation.outlines[year] - simulation.inners[year])) for year in YEARS}
        baseline_width = float(np.mean(baseline.outlines[high_year] - baseline.inners[high_year]))

        self.assertNotIn("excess", set(high_cells["role"]))
        self.assertGreater(widths[high_year], baseline_width * 1.08)

    def test_deterministic_inside_bands_and_no_deficit_cells(self) -> None:
        weekly = _weekly_synthetic(spike=(2003, 2, 1800.0), wave=(2004, range(24, 32), 1100.0))
        simulation_a = simulate_v3_cells(weekly, YEARS[0], YEARS[-1], _args(seed=21))
        simulation_b = simulate_v3_cells(weekly, YEARS[0], YEARS[-1], _args(seed=21))
        inside = (
            (simulation_a.cells["radial"] >= simulation_a.cells["inner_radius"] - 1e-8)
            & (simulation_a.cells["radial"] <= simulation_a.cells["outer_radius"] + 1e-8)
        )

        self.assertEqual(_cell_hash(simulation_a.cells), _cell_hash(simulation_b.cells))
        self.assertTrue(bool(inside.all()))
        self.assertNotIn("deficit", set(simulation_a.cells["role"]))


if __name__ == "__main__":
    unittest.main()
