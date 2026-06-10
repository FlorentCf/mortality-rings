# Mortality Rings

Create dendrochronology-inspired mortality charts and animations from daily death counts.

The Belgian example downloads Statbel's open-data file and turns daily deaths into simulated tree sections. The current default writes both renderers:

- `v1`: the older polar/radial cell renderer, kept as a comparison baseline.
- `v2`: a Cruz-inspired cambium simulation where weekly mortality deposits colored cells below the bark, packs them without visible overlap, and renders both an art view and an analytical view.

The design is inspired by Pedro Cruz's [Simulated Dendrochronology](https://pmcruz.com/dendrochronology/) and the VISAP paper, [Process of simulating tree rings for immigration in the U.S.](https://pmcruz.com/download/portfolio-camera-ready.pdf). This project adapts the principles to Belgian mortality seasonality rather than cloning the immigration visualization.

![Belgium mortality tree, v2 art](examples/belgium_mortality_tree_1992_2025_v2_art.png)

Animated preview: [Belgium mortality tree v2 art GIF](examples/belgium_mortality_tree_1992_2025_v2_art.gif)

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

- `belgium_mortality_tree_1992_2025_v1.png`
- `belgium_mortality_tree_1992_2025_v1.gif`
- `belgium_mortality_tree_1992_2025_v2_art.png`
- `belgium_mortality_tree_1992_2025_v2_art.gif`
- `belgium_mortality_tree_1992_2025_v2_analytical.png`
- `belgium_mortality_tree_1992_2025_v2_analytical.gif`
- matching `.mp4` files when ffmpeg is installed
- `weekly_mortality_summary.csv`

Use `--renderer v1`, `--renderer v2`, or `--renderer both` to choose the renderer. Use `--v2-views art`, `--v2-views analytical`, or `--v2-views both` for v2.

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

## V2 Encoding

V2 uses weekly mortality as the simulation grain. The tool learns a baseline seasonal profile from the reference years, scales that profile to each year's total deaths, and compares each observed week with its expected value.

Expected mortality becomes colored seasonal tissue rather than grey filler: winter and early-year weeks lean cool blue, spring/summer weeks move through teal and green, and late-year weeks warm toward ochre and rust. Excess mortality pushes those cells toward ochre, dried red, and burgundy. Deficit periods are represented mostly by lower density; the analytical view also marks some deficit cells in muted blue-green.

Cells are not placed at exact calendar axes. Each week has a broad seasonal region around the trunk, and cells are sampled from probability fields. Event weeks use tighter angular distributions, so COVID waves, winter waves, and heat-wave periods form visible local clusters.

New v2 cells are inserted below the current bark/cambium and packed into tight non-overlapping dash fields. A SciPy `cKDTree` neighbor search verifies the final collision tolerance. Thin pale annual channels trace bark growth, with every fifth year only slightly emphasized so the tree structure breathes without becoming a chart grid.

## V1 Encoding

V1 is the previous polar renderer. It samples cells from daily deaths, assigns day-of-year to an angle, and grows a smoothed radial boundary. It remains useful as a comparison baseline, but v2 is the preferred artistic direction.

## Useful Options

```bash
mortality-rings --help
```

Common settings:

- `--renderer`: `v1`, `v2`, or `both`; defaults to `both`.
- `--v2-views`: `art`, `analytical`, or `both`; defaults to `both`.
- `--baseline-start` and `--baseline-end`: reference years for seasonal medians.
- `--first-year` and `--last-year`: visible year range.
- `--people-per-cell`: deaths represented by one cell.
- `--seed`: deterministic cell placement seed.
- `--v2-cell-radius`: v2 deposited-cell collision size; smaller values create finer high-resolution dashes.
- `--v2-relax-iterations` and `--v2-outward-strength`: legacy v2 tuning flags kept for experimentation.
- `--colormap`: v1 color scale; defaults to `wood-blood`.
- `--clip-low` and `--clip-high`: v1 color-scale clipping bounds as proportions.
- `--png-dpi`: high-resolution static PNG export.
- `--no-gif` or `--no-mp4`: skip animation formats.
- `--fps`: animation frame rate.
- `--title` and `--subtitle`: visible analytical chart text.

## Data Source

The Belgian example uses Statbel open data: [Number of deaths per day](https://statbel.fgov.be/en/open-data/number-deaths-day).

Statbel's broader mortality page reported 112,923 deaths in Belgium in 2025, matching the latest daily open-data file used by this project.

## License

Code is released under the MIT License. Check the license terms of your data source before publishing derived charts.
