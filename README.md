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
│       ├── pipeline.py      # entry point, Settings, config validation
│       ├── preprocess.py    # one file → decimated samples on the global grid
│       ├── recording.py     # stitches files into a streamed recording
│       ├── cache.py         # optional .npz cache of preprocessed blocks
│       ├── extraction.py    # sliding-window features → results CSV
│       ├── chaotic_features.py  # per-window feature wrappers
│       └── chaos_algorithms.py  # Wolf, Rosenstein, SampEn, CorrDim, FNN, AMI
├── tests/                   # pytest suite
├── raw/<STATION>/<EVENT>/*.mseed        # input (not in the repo)
├── cache/<STATION>/<EVENT>/...          # optional, off by default
└── results/<STATION>/<EVENT>/ENZ/...    # the output
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
| `results_dir` | Output root (`results`) |
| `results_subdir` | Sub-folder for the feature CSV (`ENZ`) |
| `cache_dir` | Where the optional block cache lives (`cache`) |

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

A window must cover at least three samples, or the least-squares fit runs
through two points and reports a meaningless R² of exactly 1.0. At
`fs = 5 Hz` and `mean_period = 1.0 s` one mean period is five samples, so
`[0, 0.2]` would span two — this is rejected at load time rather than written
into the results.

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
