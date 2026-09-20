# Signal Processing & Chaotic Feature Extraction

A two-stage Python pipeline for seismic signal processing and nonlinear (chaotic) feature extraction from MiniSEED (MSEED) files. Designed for research on earthquake precursor analysis using three-component (E, N, Z) broadband seismometers.

---

## Project Structure

```
signal-process-chaotic-feature-extraction/
├── main.py               # Entry point — configures and runs the full pipeline
├── preprocess.py         # Stage 1: MSEED → per-hour CSV (gap handling, filter, downsample)
├── extraction.py         # Stage 2: sliding-window chaotic feature extraction
├── chaotic_features.py   # Feature computation functions (statistical + chaotic metrics)
├── chaos_algorithms.py   # Core nonlinear algorithms (Wolf, Rosenstein, SampEn, CorrDim)
└── raw/
    └── <STATION>/
        └── <EARTHQUAKE_NAME>/
            └── <file>.mseed
```

---

## Pipeline Overview

### Stage 1 — MSEED Preprocessing (`preprocess.py`)

- Reads a raw MSEED file and forces all traces to `float64`.
- Detects and handles data gaps:
  - **Small gaps** (< `GAP_THRESHOLD` s): filled with cubic spline interpolation.
  - **Large gaps** (≥ `GAP_THRESHOLD` s): masked with `NaN`.
- Applies a zero-phase Butterworth bandpass filter (`FREQMIN`–`FREQMAX` Hz).
- Downsamples to the target sampling frequency `Fs`.
- Saves each component (E, N, Z) as hourly CSVs under `proceeded/<EARTHQUAKE_NAME>/`.
- Writes a gap report log to `proceeded/<EARTHQUAKE_NAME>/logs/`.

### Stage 2 — Feature Extraction (`extraction.py`)

Applies a sliding window of `WIN_SEC` seconds (step `STEP_SEC`) over each hourly CSV for all three channels. Features computed per window:

| Feature | Description |
|---|---|
| `ham_mean` | Mean of the raw signal |
| `ham_std` | Standard deviation (unbiased) |
| `ham_min` / `ham_max` | Min and max amplitude |
| `aktivite_std` | Activity indicator (= `ham_std`) |
| `norm_min` / `norm_max` | Min/max after z-score normalisation |
| `wolf_lye` | Maximum Lyapunov exponent — Wolf (1985) |
| `ros_short` / `ros_long` | Short- and long-term Lyapunov exponents — Rosenstein (1993) |
| `samp_ent` | Sample Entropy |
| `corr_dim` | Correlation Dimension |

Each column is prefixed with the channel name (e.g. `E_wolf_lye`, `N_samp_ent`, `Z_corr_dim`).  
Output: `results/<STATION>/ENZ/<STATION>_<date_range>_ENZ_features.csv`

---

## Configuration

All parameters are set in the `Settings` class in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `STATION` | `'ELZG'` | Seismic station code |
| `EARTHQUAKE_NAME` | `'24012020_M6.8_Sivrice__Elazig_'` | Event sub-folder name (also used to auto-derive catalog date range) |
| `RAW_FILE_NAME` | `'TU_ELZG_...HH.mseed'` | Input MSEED filename |
| `Fs` | `5.0` | Target sampling frequency (Hz) |
| `FREQMIN` | `0.1` | Bandpass lower bound (Hz) |
| `FREQMAX` | `2.0` | Bandpass upper bound (Hz) |
| `GAP_THRESHOLD` | `2.0` | Gap threshold for interpolation vs. NaN masking (s) |
| `WIN_SEC` | `200` | Window length (s) |
| `STEP_SEC` | `50` | Sliding step (s) |
| `PREV_SEC` | `150` | Rolling context buffer length (s) |
| `WARMUP_COUNT` | `3` | Warm-up windows to discard at the start |
| `N_JOBS` | `-1` | Parallel jobs for feature extraction (`-1` = all cores) |

---

## Requirements

Python 3.9+ is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** ObsPy may require additional system libraries on some platforms. See the [ObsPy installation guide](https://docs.obspy.org/install.html) for details.

---

## How to Run

1. **Place your MSEED file** at:

   ```
   raw/<STATION>/<EARTHQUAKE_NAME>/<RAW_FILE_NAME>
   ```

   Example (default config):

   ```
   raw/ELZG/24012020_M6.8_Sivrice__Elazig_/TU_ELZG_24012020_000000_25012020_000000_HH.mseed
   ```

2. **Adjust parameters** in `main.py` (`Settings` class) if needed.

3. **Run the pipeline:**

   ```bash
   python main.py
   ```

4. **Output files** (all under `results/<STATION>/ENZ/`):

   | File | Description |
   |---|---|
   | `<STATION>_<date>_ENZ_features.csv` | Sliding-window feature matrix |

   Preprocessed hourly CSVs and gap logs are saved under `proceeded/<EARTHQUAKE_NAME>/`.

---

## Data Availability

The following files are **not included** in this repository due to data sharing restrictions:

| File / Folder | Description |
|---|---|
| `raw/` | Raw MiniSEED seismic recordings |

To run the pipeline, you must supply these files yourself and place them in the paths described above.

---

## References

Nonlinear algorithms adapted from:

> S. Sarwar, A. Likens, N. Stergiou, S. Mastorakis, *"A nonlinear analysis software toolkit for biomechanical data"*, arXiv:2311.06723, 2023. https://arxiv.org/abs/2311.06723
