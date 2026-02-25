# Unified SP 800-90B Validation Framework (CMOS TRNG vs QRNG)

This folder contains a small **Python-only** reproduction pack to regenerate the **tables**
used in the accompanying manuscript/project write-up:

- **Table**: QRNG variations → observable symptoms → affected bound → intuition on Hmin
- **Table**: synthetic case definitions (artifact parameters)
- **Table**: aggregated proxy validation results (N=200k bits, 5 seeds)

> Note: This is **not** the official NIST SP 800-90B entropy assessment tool.
> It is a **benchmark proxy** aligned with the framework narrative (IID screening proxy + conservative bounds).

## Quick Start

### 1) Create environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

### 3) Generate artifacts
```bash
python generate_artifacts.py --outdir out
```

Outputs will appear under:
- `out/tables/`

## Parameters (optional)

You can change:
- `--N` bits per case (default 200000)
- `--seeds` number of seeds (default 5)
- `--bias_thr`, `--lag1_thr` for IID screening proxy
- `--apt_W`, `--apt_z` for APT proxy
- `--rct_alpha` for RCT proxy threshold

Example:
```bash
python generate_artifacts.py --outdir out --N 300000 --seeds 10 --bias_thr 0.01 --lag1_thr 0.03
```

## What each output represents

### Tables
- `table_qrng_variations.csv`  
  Implementation variation mapping (efficiency mismatch, dead time, afterpulsing, drift) to symptoms and which bound tightens.

- `table_case_definitions.csv`  
  Synthetic benchmark cases: IID bias, Markov dependence, and drift.

- `table_aggregated_results.csv`  
  Aggregated (mean) metrics across seeds: IID pass rate, lag-1, p1, selected conservative Hmin bound, APT/RCT proxy alarm rates.

## License
This pack includes an MIT `LICENSE` file at the repo root.
