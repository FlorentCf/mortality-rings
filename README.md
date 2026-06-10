# Mortality Rings

Create dendrochronology-inspired mortality charts and animations from daily death counts.

The Belgian example downloads Statbel's open-data file and turns daily deaths into simulated tree sections. The default renderer is now `v3`, the artistic cambium renderer:

- `v1`: the older polar/radial cell renderer, kept as a comparison baseline.
- `v2`: a Cruz-inspired cambium simulation where weekly mortality deposits colored cells below the bark, packs them without visible overlap, and renders both an art view and an analytical view.
- `v3`: the recommended V3.1 renderer. It builds real annual bands, keeps a central pith, reserves quiet ring channels, fills each band with weighted blue-noise cells, and uses only acute excess mortality to create scar-like local growth.

The design is inspired by Pedro Cruz's [Simulated Dendrochronology](https://pmcruz.com/dendrochronology/) and the VISAP paper, [Process of simulating tree rings for immigration in the U.S.](https://pmcruz.com/download/portfolio-camera-ready.pdf). This project adapts the principles to Belgian mortality seasonality rather than cloning the immigration visualization.

![Belgium mortality tree, v3 art](examples/belgium_mortality_tree_1992_2025_v3_art.png)

Animated preview: [Belgium mortality tree v3 art GIF](examples/belgium_mortality_tree_1992_2025_v3_art.gif)

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Quick Start

Generate the Belgian chart directly from Statbel:

```bash
mortality-rings --statbel --output-dir outputs/belgium --name belgium_mortality_tree_1992_2025
```

By default this writes:

- `belgium_mortality_tree_1992_2025_v3_art.png`
- `belgium_mortality_tree_1992_2025_v3_art.gif`
- `belgium_mortality_tree_1992_2025_v3_analytical.png`
- `belgium_mortality_tree_1992_2025_v3_analytical.gif`
- matching `.mp4` files when ffmpeg is installed
- `weekly_mortality_summary.csv`

Use `--renderer v1`, `--renderer v2`, `--renderer v3`, `--renderer both`, or `--renderer all` to choose renderers. `both` means the legacy v1 plus v2 comparison. `all` writes v1, v2, and v3. Use `--v2-views art`, `--v2-views analytical`, or `--v2-views both` for v2, and the matching `--v3-views` flag for v3.

Generate legacy comparison outputs:

```bash
mortality-rings --statbel --renderer both --output-dir outputs/legacy --name belgium_mortality_tree_1992_2025
```

Generate debug fields for checking the simulation:

```bash
mortality-rings --statbel --renderer v3 --v3-debug-fields --no-gif --no-mp4 --output-dir outputs/debug
```

Generate a V3.1 contact sheet for tuning hierarchy, gaps, and scar strength:

```bash
mortality-rings --statbel --renderer v3 --v3-preset-sweep scar-gap-texture --no-gif --no-mp4 --output-dir outputs/sweeps/v3_1
```

## Use Your Own Data

Your input needs one date column and one numeric count column. CSV, TXT, and ZIP files containing one CSV/TXT file are supported.

```bash
mortality-rings ^
  --input path/to/daily_deaths.csv ^
  --date-column date ^
  --count-column deaths ^
  --sep "," ^
  --title "Mortality tree" ^
  --baseline-start 2015 ^
  --baseline-end 2019 ^
  --output-dir outputs/custom
```

For European day-first dates, add `--dayfirst`.

## V3.1 Encoding

V3.1 separates three mortality questions that v1/v2 used to blur together:

- `seasonal_concentration_pct`: how concentrated a week is within its own year. This is preserved for v1/v2 compatibility.
- `absolute_excess_deaths`: how far a week is above a robust pre-2020 expected mortality baseline. This supports analytical summaries and broad annual stress.
- `acute_excess_deaths`: the part of weekly excess above a robust weekly threshold. This is the only signal that creates visible local scar cells and scar geometry.

The absolute baseline uses the 1992-2019 annual trend and distributes each expected annual total by the baseline seasonal shares. It is not forced to sum to observed deaths in a year, so 2020, 2021, 2022, and other high-mortality years can create real positive excess instead of being normalized away.

Each year becomes a physical band:

- The renderer starts outside a small empty pith and reserves a real, quiet annual channel before each band.
- Observed deaths determine how many cells are placed inside the band.
- Expected deaths form quiet wood/tissue marks with subtle seasonal tint.
- General annual excess thickens the whole band slightly.
- Acute weekly excess becomes longer, darker, warmer, more saturated cells and locally thickens the band into scars.
- Deficits are not rendered as deficit cells. They reduce density/opacity and only weakly affect geometry.

Cells are sampled with weighted blue-noise rejection and then locally relaxed with a SciPy `cKDTree` neighbor pass while constrained to their annual band. The result is closer to simulated cambium growth than to a polar chart: the white rings are actual empty space, normal tissue is the primary texture, and only acute mortality shocks become memorable scars.

## V2 Encoding

V2 uses weekly mortality as the simulation grain. The tool learns a baseline seasonal profile from the reference years, scales that profile to each year's total deaths, and compares each observed week with its expected value.

Expected mortality becomes colored seasonal tissue rather than grey filler: winter and early-year weeks lean cool blue, spring/summer weeks move through teal and green, and late-year weeks warm toward ochre and rust. Excess mortality pushes those cells toward ochre, dried red, and burgundy. Deficit periods are represented mostly by lower density; the analytical view also marks some deficit cells in muted blue-green.

Cells are not placed at exact calendar axes. Each week has a broad seasonal region around the trunk, and cells are sampled from probability fields. Event weeks use tighter angular distributions, so COVID waves, winter waves, and heat-wave periods form visible local clusters.

New v2 cells are inserted below the current bark/cambium and packed into tight non-overlapping dash fields. A SciPy `cKDTree` neighbor search verifies the final collision tolerance. Thin pale annual channels trace bark growth, with every fifth year only slightly emphasized so the tree structure breathes without becoming a chart grid.

## V1 Encoding

V1 is the previous polar renderer. It samples cells from daily deaths, assigns day-of-year to an angle, and grows a smoothed radial boundary. It remains useful as a comparison baseline, but v3 is the preferred artistic direction.

## Useful Options

```bash
mortality-rings --help
```

Common settings:

- `--renderer`: `v1`, `v2`, `v3`, `both`, or `all`; defaults to `v3`.
- `--v2-views`: `art`, `analytical`, or `both`; defaults to `both`.
- `--v3-views`: `art`, `analytical`, or `both`; defaults to `both`.
- `--baseline-start` and `--baseline-end`: reference years for seasonal medians.
- `--first-year` and `--last-year`: visible year range.
- `--people-per-cell`: deaths represented by one cell.
- `--seed`: deterministic cell placement seed.
- `--v2-cell-radius`: v2 deposited-cell collision size; smaller values create finer high-resolution dashes.
- `--v2-relax-iterations` and `--v2-outward-strength`: legacy v2 tuning flags kept for experimentation.
- `--v3-people-per-cell`: deaths represented by one v3 cambium cell; defaults to `220`.
- `--v3-cell-radius`: v3 collision radius; defaults to `0.0072`.
- `--v3-relax-iterations`: local relaxation passes per annual band; defaults to `60`.
- `--v3-debug-fields`: write outlines, year-colored cells, role-colored cells, growth profile, absolute excess, and acute excess debug images.
- `--v3-preset-sweep scar-gap-texture`: render 24 V3.1 thumbnails plus a contact sheet and CSV manifest.
- `--colormap`: v1 color scale; defaults to `wood-blood`.
- `--clip-low` and `--clip-high`: v1 color-scale clipping bounds as proportions.
- `--png-dpi`: high-resolution static PNG export.
- `--no-gif` or `--no-mp4`: skip animation formats.
- `--fps`: animation frame rate.
- `--title` and `--subtitle`: visible analytical chart text.

## Data Source

The Belgian example uses Statbel open data: [Number of deaths per day](https://statbel.fgov.be/en/open-data/number-deaths-day).

The `--statbel` option downloads the Statbel daily deaths open-data file at render time. For reproducible publications, save the downloaded source file and report the snapshot date.

## Design Limitations

This is a symbolic biological simulation, not a geographic or epidemiological model. Angles encode seasonal position unless you supply a different stratified data design. The art view is interpretive; use the analytical view and debug fields to read exact mortality patterns and verify that scars correspond to acute weekly excess.

## License

Code is released under the MIT License. Check the license terms of your data source before publishing derived charts.
