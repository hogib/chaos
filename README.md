# CHAOS — Seismic Chaotic Feature Extraction

A Python pipeline for seismic signal processing and nonlinear (chaotic)
feature extraction from MiniSEED files. Built for earthquake precursor
research on three-component (E, N, Z) broadband seismometers.

MSEED goes in, one feature table comes out. The recording is cleaned,
filtered and decimated as it streams past, and a sliding window computes
statistical and nonlinear-dynamics features over one continuous timeline.
Nothing is written between the two: there is no intermediate tree to
regenerate, stale, or keep in step.

---

## Requirements

Python **3.12+**. The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # create .venv and install everything
```

Dependencies: `numpy`, `scipy`, `pandas`, `obspy`, `joblib`, `pytest`.
ObsPy may need extra system libraries on some platforms — see the
[ObsPy install guide](https://docs.obspy.org/install.html).

---

## Layout

```
chaos/
├── config.json              # every tunable parameter lives here
├── src/
│   ├── main.py              # `python src/main.py` shim
│   └── chaos/
│       ├── cli.py           # the `chaos` command line
│       ├── pipeline.py      # Settings, config validation, run planning
│       ├── periods.py       # target periods and quiet period selection
│       ├── mean_period.py   # Rosenstein mean period from quiet data
│       ├── preprocess.py    # one file → decimated samples on the global grid
│       ├── recording.py     # stitches files into a streamed recording
│       ├── cache.py         # optional .npz cache of preprocessed blocks
│       ├── extraction.py    # sliding-window features → results CSV
│       ├── chaotic_features.py  # per-window feature wrappers
│       └── chaos_algorithms.py  # Wolf, Rosenstein, SampEn, CorrDim, FNN, AMI
├── tests/                   # pytest suite
├── raw/<STATION>/<EVENT>/*.mseed        # event-mode input (not in the repo)
├── data/<NET>_<STATION>_*.mseed         # continuous data for period runs
├── cache/...                            # mean period, optional block cache
└── results/<STATION>/<RUN>/ENZ/...      # the output, with settings.json
```

---

## Running

Put your MSEED files here:

```
raw/<STATION>/<EARTHQUAKE_NAME>/*.mseed
```

Point `config.json` at them and run either form:

```bash
uv run chaos            # installed console script
python src/main.py      # equivalent shim
```

The project root — the directory holding `config.json`, `raw/` and
`results/` — is found in this order:

1. the `CHAOS_ROOT` environment variable, if set;
2. the nearest ancestor of the source tree containing a `config.json`;
3. the current working directory.

`CHAOS_ROOT` is the easy way to run against a dataset outside the checkout:

```bash
CHAOS_ROOT=/data/campaign-2025 uv run chaos
```

### What to run on

There are three modes, chosen by `target`:

| Mode | When | Input | Output folder (`<RUN>`) |
|---|---|---|---|
| Event | `target.start_date` is `null` | `raw/<STATION>/<EVENT>/` | `<EVENT>` |
| Period | `target.start_date` is set | `[start_date, start_date + duration_days)` cut out of `data/` | `20250416_7d` |
| Quiet | `target.quiet_window` is `true` | `quiet_window.samples` quiet periods of `duration_days` each, from `data/` | `quiet_7d/<YYYYMMDD>` per period |

With `"process_all": false`, only the `single_run` station (and event) is
processed. With `"process_all": true`, event mode discovers every
`raw/<STATION>/<EVENT>/` folder, and the period modes every station with
`NET_STA_*.mseed` files in `data/`. Several runs go at a time; each reports
`OK` or `FAIL` and a failure never stops the rest.

Periods are read straight out of the long continuous files one day at a time
(plus filter padding from either side), so a period can start anywhere in a
21-day file without the whole file ever being decoded.

A quiet period has no catalog event of at least `quiet_window.min_mag` within
`quiet_window.max_dist_km` of the station (coordinates come from
`paths.station_coords`), and lies, with its filter padding, entirely inside
recorded data. Candidates start at UTC midnights; the pick is random but
seeded and never overlapping. Near an active fault a range with no event at
all does not exist, which is why there is a magnitude floor. For ELBA with the
defaults (M3, 300 km) there are about 17 non-overlapping 7-day periods, 3 of
14 days; the run warns when it finds fewer than asked for. A quiet run also
writes `periods.csv`/`periods.json` (the pick and its parameters) and
`summary.csv` (rows and status per period) beside the period folders.

### Every run records its settings

Each run writes `settings.json` next to its features CSV: the configuration
exactly as the run used it — command-line overrides applied, the estimated
mean period filled in, the period resolved — together with the command line,
where the mean period came from, and for quiet runs the selection parameters.
Passing its `config` block back as a config file reproduces the run.

### Command line

`chaos` with no arguments does exactly what `config.json` says. Anything on
the command line overrides the file for that invocation; `--save` writes the
result back. Precedence, lowest first: `config.json`, `--set`, the dedicated
flags.

```bash
chaos run --start 2025-04-16 --days 7            # one period
chaos run --quiet --days 7 --samples 15          # 15 quiet weeks
chaos run --quiet --days 14 --min-mag 4          # longer periods need a higher floor
chaos run --event 20250423_M6.3_Silivri          # back to event mode
chaos run --station CAME --all --jobs 8
chaos run --mean-period 2.6                      # pin it; `auto` estimates it
chaos run --set feature_extraction.win_sec=300 --dry-run
chaos quiet --days 7                             # list the quiet periods only
chaos mean-period --refresh                      # re-estimate the mean period
chaos config --days 3 --min-mag 3.5 --save       # edit config.json from the shell
```

`--set KEY=VALUE` reaches any key (values are JSON: `--set
feature_extraction.channels='["N","E"]'`); unknown keys are rejected rather
than ignored. `--dry-run` lists what would run without extracting anything.
`chaos <command> --help` lists every flag.

---

## Configuration

Everything is in `config.json`, re-read on every run. Settings are validated
before any work starts, and a configuration that could only produce
meaningless output is rejected with an explanation rather than run.

### `paths`

| Key | Meaning |
|---|---|
| `raw_dir` | Where input MSEED lives (`raw`) |
| `results_dir` | Output root (`results`) |
| `results_subdir` | Sub-folder for the feature CSV (`ENZ`) |
| `cache_dir` | Where the mean period and optional block cache live (`cache`) |
| `data_dir` | Continuous MSEED for period runs and the mean period (`data`) |
| `data_pattern` | File glob inside it; `{station}` becomes the station (`*_{station}_*.mseed`) |
| `catalog` | Earthquake catalog CSV (`catalog_current.csv`) |
| `station_coords` | CSV with `station`, `latitude`, `longitude` (and optionally `network`) columns (`station_coords.csv`) |

### `target`

| Key | Default | Meaning |
|---|---|---|
| `start_date` | `null` | Period start, `YYYY-MM-DD` or ISO timestamp, UTC; `null` = event mode |
| `duration_days` | `7` | Period length; may be fractional |
| `quiet_window` | `false` | Run on quiet periods instead of `start_date` |

### `quiet_window`

| Key | Default | Meaning |
|---|---|---|
| `samples` | `15` | Periods to pick |
| `min_mag` | `3.0` | Events at or above this magnitude break quiet |
| `max_dist_km` | `300` | Only events this close to the station count; `null` = all |
| `seed` | `0` | Seed for the random pick |

Catalog parsing (`dayfirst`, `catalog_utc_offset`, event exclusion padding)
follows `mean_period_estimation.options`.

### `cache`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Keep preprocessed blocks as `.npz` between runs |

Preprocessing is roughly 2% of a run, so the cache earns its keep only when
you are re-running repeatedly while tuning feature parameters. An entry is
used only when the preprocessing settings *and* the source file are both
unchanged, so editing `config.json` or replacing a raw file invalidates it on
its own.

### `preprocessing`

| Key | Default | Meaning |
|---|---|---|
| `freq_min` | `0.1` | Bandpass lower corner (Hz) |
| `freq_max` | `2.0` | Bandpass upper corner (Hz) — must stay below `fs/2` |
| `gap_threshold_sec` | `2.0` | Below this a gap is interpolated, at or above it it is kept as NaN |
| `filter_pad_sec` | `null` | Context borrowed from neighbouring files when filtering; `null` derives it from `freq_min` |
| `channels` | `["E","N","Z"]` | Components to decode |

### `feature_extraction`

| Key | Default | Meaning |
|---|---|---|
| `fs` | `5.0` | Target sampling frequency (Hz) |
| `win_sec` | `200` | Sliding window length (s) |
| `step_sec` | `50` | Window step (s) |
| `max_gap_sec` | `null` | Outages longer than this end a block instead of being filled with NaN; `null` means one window |
| `channels` | `["N"]` | Components to extract features from |
| `n_jobs` | `-1` | Worker processes (`-1` = all cores) |
| `warmup_count` | `3` | Windows discarded at the very start of a run |

`fs` should divide the instrument's sampling rate. Decimation can only divide
by an integer, so a target that does not divide evenly produces a different
real rate than configured; the pipeline warns when this happens.

### `feature_extraction.features`

| Metric | Keys |
|---|---|
| `wolf` | `tau`, `m`, `evolve`, `min_samples` |
| `rosenstein` | `tau`, `m`, `slope`, `mean_period` |
| `sample_entropy` | `m`, `r` |
| `corr_dim` | `tau`, `m` |

`slope` is the Rosenstein fit window **in mean periods**, either
`[short_start, short_end]` or `[short_start, short_end, long_start, long_end]`.
The shipped `[0, 1, 4, 10]` is the usual pairing: a short-term exponent over
the first mean period and a long-term one over periods 4–10.

`mean_period` is used as given when it is a number. Left out or `null`, it is
estimated from seismically quiet data (see below) before any window is
processed, and the fit window is then validated against the estimate.

A window must cover at least three samples, or the least-squares fit runs
through two points and reports a meaningless R² of exactly 1.0. At
`fs = 5 Hz` and `mean_period = 1.0 s` one mean period is five samples, so
`[0, 0.2]` would span two — this is rejected at load time rather than written
into the results.

### `mean_period_estimation`

Used only when `rosenstein.mean_period` is unset. The station's continuous
data are cut into `chunk_sec` chunks, every chunk touching a catalog event
(from `pre_sec` before the origin to a magnitude-scaled coda after it) is
dropped, and a random sample of what is left is filtered and decimated exactly
as the pipeline does. The mean period is the median over all `win_sec`
windows of the reciprocal of the power-weighted mean frequency.

The catalog, data folder, file pattern and station coordinates come from
`paths`. The only key here is `options`: any option of `python -m
chaos.mean_period`, underscored — `chunks`, `chunk_sec`, `dayfirst`,
`catalog_utc_offset`, `min_mag`, `station_lat`, `station_lon`, `max_dist_km`,
`jobs`...

With `station_coords` and `max_dist_km` set, only catalog events within that
distance of the station exclude data; without them every event in the catalog
does, which leaves little quiet time for a nationwide catalog. A station code
listed under more than one network is resolved with the network in the data's
MSEED headers; if that still leaves two different locations the run stops
and asks for `station_lat`/`station_lon` in `options`. Values given there
always win over the lookup, and a station missing from the file only warns.

One value is estimated per station and shared by all of its events, so their
features stay comparable. The result, the per-window table and a plot land in
`<cache_dir>/mean_period/<STATION>/<key>/`; the key covers the catalog, the
data files and every relevant setting, so the estimate is reused until one of
them changes. The same estimate can be run by hand:

```bash
uv run python -m chaos.mean_period catalog_current.csv data --dayfirst \
    --split-files --pattern '*_ELBA_*.mseed'
```

---

## How a run works

```
raw/*.mseed ──(parallel, each padded with its neighbours' edges)──┐
                                                                  │
                     time-ordered contiguous blocks on one        │
                     global sample grid, NaN inside gaps  ◄───────┘
                                   │
                  rolling buffer one window deep
                                   │
                       batches of windows → worker pool
                                   │
                         rows appended to features.csv
```

### One global sample grid

Sample index `k` is the instant `k / fs` seconds after the UTC epoch. Every
file computes its own position on that grid, so files decimated independently
still line up to the sample when they are stitched together, and the window
lattice does not depend on which files happen to be present. Re-running with
an extra day prepended leaves every other window exactly where it was.

Files are ordered by the time they actually start recording, read from their
headers — not by name. A station writing `DDMMYYYY` filenames sorts 01 May
before 08 April, and joining files in that order would splice the wrong months
together.

### Preprocessing, per file

1. Read the file together with `filter_pad_sec` of its neighbours.
2. Classify gaps against `gap_threshold_sec`. Short gaps are filled by cubic
   interpolation; long gaps stay NaN and are never filtered across.
3. Detrend (mean, then linear) and apply a zero-phase 4-corner Butterworth
   bandpass over `freq_min`–`freq_max`, to each contiguous stretch separately.
   Stretches shorter than one filter warm-up (`3 / freq_min` seconds) are left
   as NaN.
4. Decimate onto the global grid and discard the padding.

The padding is what makes step 3 both parallel and correct. A zero-phase
bandpass reaches back about `3 / freq_min` seconds, so a file filtered alone is
wrong at both ends — measurably so: against a single-pass filter of the whole
recording, filtering hour by hour with no context is off by **98% of the
signal's standard deviation** near each join, while 60 s of real context brings
that to **0.03%**. Each channel's padding reaches to its own first sample,
since the three components rarely open together.

### Stitching

Consecutive files are joined into one timeline. A seam shorter than
`max_gap_sec` is filled with NaN, so windows crossing it report as empty rather
than being silently spliced; a longer outage ends the block instead, so a week
off air does not become a million NaN rows. Files that overlap — day-long
recordings routinely overhang midnight — contribute each instant once.

Blocks are handed downstream as soon as nothing still to come can change them,
so peak memory follows the worker count, not the length of the recording. A
month and a year cost the same.

### Extraction

A window of `win_sec` slides in `step_sec` steps over the timeline. Window
starts sit on a fixed lattice of whole steps from the epoch, so they land on
round times and stay put between runs. Every `(channel, window)` pair goes to a
worker pool that lives for the whole run, and rows are appended to the CSV in
batches rather than accumulated.

A window containing any NaN, or with zero standard deviation, yields NaN for
every feature — it is never computed on partial data.

### Output columns

`window_start` and `window_end` (ISO UTC), then `Window_ID`
(`<date>_<hour>_w<n>`) and `Time_min` (minutes from the start of that hour to
the end of the window), then per channel, prefixed with the channel name
(`N_wolf_lye`, `N_samp_ent`, …):

| Column | Description |
|---|---|
| `ham_mean` | Mean of the raw signal |
| `ham_std` | Standard deviation (unbiased) |
| `ham_min` / `ham_max` | Min and max amplitude |
| `aktivite_std` | Activity indicator (= `ham_std`) |
| `norm_min` / `norm_max` | Min/max after z-score normalisation |
| `wolf_lye` | Maximum Lyapunov exponent — Wolf (1985) |
| `ros_short` | Short-term Lyapunov exponent — Rosenstein (1993), per mean period |
| `ros_r2` | R² of the short-term fit |
| `ros_n_points` | Samples the short-term fit used |
| `ros_low_fit_quality` | True when the short fit has fewer than 3 points or R² < 0.8 |
| `ros_long` | Long-term Lyapunov exponent, per mean period |
| `ros_long_r2` | R² of the long-term fit |
| `ros_long_n_points` | Samples the long-term fit used |
| `samp_ent` | Sample Entropy |
| `corr_dim` | Correlation Dimension |

Output: `results/<STATION>/<EVENT>/ENZ/<STATION>_<EVENT>_features.csv`, with
the gap report in `results/<STATION>/<EVENT>/ENZ/logs/`.

`ros_low_fit_quality` is worth filtering on. The Lyapunov exponents are slopes
of a least-squares fit to the divergence curve, and a low R² means the curve
had no straight stretch to fit — the number is still produced, but it is not
describing exponential divergence. A fit through fewer than three points is
flagged separately, because two points always report R² = 1.0 regardless of
the data.

`corr_dim` picks its fit range by thresholding the correlation integral, so
roughly one window in 200 sits on a bin edge and moves by ~1% under changes
far below the noise floor. Treat differences of a percent or two in `corr_dim`
alone as estimator noise rather than signal.

---

## Tests

```bash
uv run pytest
```

The suite covers the numerical cores against known-good behaviour, gap
handling, stitching and the rolling buffer, the cache, config validation, and
end-to-end runs on synthetic MSEED. Most tests are written as regressions for
a specific bug and say so in their docstring.

Two carry most of the weight:

- **padded filtering matches a continuous filter** — the property that lets
  files be preprocessed in parallel without changing the answer;
- **splitting the recording into more files changes nothing** — where a
  recording happens to be cut into files is an accident of how it was
  archived, and must not move a feature value.

---

## Data Availability

`raw/` (raw MiniSEED recordings) is **not** included in this repository due to
data sharing restrictions. Supply your own and place it as described above.

---

## References

Nonlinear algorithms adapted from:

> S. Sarwar, A. Likens, N. Stergiou, S. Mastorakis, *"A nonlinear analysis
> software toolkit for biomechanical data"*, arXiv:2311.06723, 2023.
> https://arxiv.org/abs/2311.06723

Methods:

> M. T. Rosenstein, J. J. Collins, C. J. De Luca, *"A practical method for
> calculating largest Lyapunov exponents from small data sets"*, Physica D 65
> (1993) 117–134.

> A. Wolf, J. B. Swift, H. L. Swinney, J. A. Vastano, *"Determining Lyapunov
> exponents from a time series"*, Physica D 16 (1985) 285–317.
