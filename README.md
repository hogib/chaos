# CHAOS — Seismic Chaotic Feature Extraction

A two-stage Python pipeline for seismic signal processing and nonlinear
(chaotic) feature extraction from MiniSEED files. Built for earthquake
precursor research on three-component (E, N, Z) broadband seismometers.

Stage 1 turns raw MSEED into clean, gap-aware, decimated hourly CSVs.
Stage 2 slides a window over those CSVs and computes statistical and
nonlinear-dynamics features for every window.

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
│       ├── pipeline.py      # entry point, Settings, config validation
│       ├── preprocess.py    # stage 1: MSEED → hourly CSV
│       ├── extraction.py    # stage 2: sliding-window features
│       ├── chaotic_features.py  # per-window feature wrappers
│       └── chaos_algorithms.py  # Wolf, Rosenstein, SampEn, CorrDim, FNN, AMI
├── tests/                   # pytest suite
├── raw/<STATION>/<EVENT>/*.mseed        # input (not in the repo)
├── proceeded/<STATION>/<EVENT>/...      # stage 1 output
└── results/<STATION>/<EVENT>/ENZ/...    # stage 2 output
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

The project root — the directory holding `config.json`, `raw/`, `proceeded/`
and `results/` — is found in this order:

1. the `CHAOS_ROOT` environment variable, if set;
2. the nearest ancestor of the source tree containing a `config.json`;
3. the current working directory.

`CHAOS_ROOT` is the easy way to run against a dataset outside the checkout:

```bash
CHAOS_ROOT=/data/campaign-2025 uv run chaos
```

### One event or all of them

With `"process_all": false`, only the `single_run` station/event is processed.
With `"process_all": true`, every `raw/<STATION>/<EVENT>/` folder is discovered
and processed, several events at a time; each one reports `OK` or `FAIL` and a
failure never stops the rest.

---

## Configuration

Everything is in `config.json`, re-read on every run. Settings are validated
before any work starts, and a configuration that could only produce
meaningless output is rejected with an explanation rather than run.

### `paths`

| Key | Meaning |
|---|---|
| `raw_dir` | Where input MSEED lives (`raw`) |
| `processed_dir` | Stage 1 output root (`proceeded`) |
| `results_dir` | Stage 2 output root (`results`) |
| `results_subdir` | Sub-folder for the feature CSV (`ENZ`) |

### `preprocessing`

| Key | Default | Meaning |
|---|---|---|
| `window_sec` | `3600.0` | Length of one output file, in seconds |
| `freq_min` | `0.1` | Bandpass lower corner (Hz) |
| `freq_max` | `2.0` | Bandpass upper corner (Hz) — must stay below `fs/2` |
| `gap_threshold_sec` | `2.0` | Below this a gap is interpolated, at or above it it is kept as NaN |
| `channels` | `["E","N","Z"]` | Components written to disk |

### `feature_extraction`

| Key | Default | Meaning |
|---|---|---|
| `fs` | `5.0` | Target sampling frequency (Hz) |
| `win_sec` | `200` | Sliding window length (s) |
| `step_sec` | `50` | Window step (s) |
| `prev_sec` | `150` | Carry-over kept from the previous hour (s) |
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

A window must cover at least three samples, or the least-squares fit runs
through two points and reports a meaningless R² of exactly 1.0. At
`fs = 5 Hz` and `mean_period = 1.0 s` one mean period is five samples, so
`[0, 0.2]` would span two — this is rejected at load time rather than written
into the results.

---

## Stage 1 — Preprocessing

For each MSEED file:

1. Read and cast every trace to `float64`.
2. Classify gaps against `gap_threshold_sec`. Short gaps are filled by cubic
   interpolation; long gaps stay as NaN and are never filtered across.
3. Detrend (mean, then linear) and apply a zero-phase 4-corner Butterworth
   bandpass over `freq_min`–`freq_max`. Around a long gap, each clean segment
   is filtered on its own so the gap does not smear into the good data.
   Segments shorter than one filter warm-up (`3 / freq_min` seconds) are left
   as NaN.
4. Decimate to `fs`. Every segment is placed on one global decimation grid, so
   samples after a gap stay in phase with samples before it.
5. Split into `window_sec` windows aligned to midnight and write one CSV per
   window per component to
   `proceeded/<STATION>/<EVENT>/<YYYY_MM_DD>/<C>/<YYYYMMDD_HHMMSS>_<C>.csv`.

Each window is written half-open — `[start, end)` — so consecutive files never
share a sample, and every channel is laid out on the window's own sample grid,
padded with NaN where it has no data. Every CSV of a given window therefore has
exactly `window_sec * fs` rows, and row *i* is the same instant in every
channel.

Windows a file barely touches are skipped. Day-long recordings routinely
overhang midnight by a second or two, and that sliver is an hour the
neighbouring day's file writes in full to the very same path. Anything shorter
than one filter warm-up is dropped and noted in the log rather than raced
against the file that owns the hour.

A gap report is written to `proceeded/<STATION>/<EVENT>/logs/`, listing every
long gap and every skipped window, ordered by input file.

Files are processed in parallel, one worker per file.

---

## Stage 2 — Feature Extraction

Each hourly CSV is joined to the last `prev_sec` seconds of the previous one,
so a window straddling an hour boundary still sees continuous signal. A window
of `win_sec` slides over the result in `step_sec` steps. Every
`(channel, window)` pair is dispatched to a worker pool that lives for the
whole run.

A window containing any NaN, or with zero standard deviation, yields NaN for
every feature — it is never computed on partial data. Channels that disagree
on length are padded with NaN rather than sliced past their end.

### Output columns

`Window_ID` (`<date>_<hour>_w<n>`) and `Time_min` (minutes from the start of
that hour to the end of the window), then per channel, prefixed with the
channel name (`N_wolf_lye`, `N_samp_ent`, …):

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

Output: `results/<STATION>/<EVENT>/ENZ/<STATION>_<first_date>-<last_date>_ENZ_features.csv`

`ros_low_fit_quality` is worth filtering on. The Lyapunov exponents are slopes
of a least-squares fit to the divergence curve, and a low R² means the curve
had no straight stretch to fit — the number is still produced, but it is not
describing exponential divergence. A fit through fewer than three points is
flagged separately, because two points always report R² = 1.0 regardless of
the data.

---

## Tests

```bash
uv run pytest
```

The suite covers the numerical cores against known-good behaviour, the gap and
decimation-phase handling, window alignment across file boundaries, config
validation, and end-to-end runs of both stages on synthetic MSEED. Most tests
are written as regressions for a specific bug and say so in their docstring.

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
