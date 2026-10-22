# vibration-noise-detection

Leakage-free condition monitoring of compressors from two-channel accelerometer recordings:
**healthy (OK) vs. faulty (NG)**, decided per *physical engine*.

This is a ground-up, tested re-engineering of the original
[`vibration_noise_detection`](https://github.com/MiladSoleymani/vibration_noise_detection)
project (see [NOTICE.md](NOTICE.md)). The original idea (high-pass → PSD → classical models →
majority vote) is kept; the evaluation protocol, code structure and tooling are not.
Every problem found in the original and how it was fixed is listed, with the test that guards
the fix, in [docs/CHANGES_FROM_ORIGINAL.md](docs/CHANGES_FROM_ORIGINAL.md).

## Why the evaluation protocol is the point

With a dozen engines, *how* you validate matters more than *which* model you use:

| Pitfall in the original | What this package does |
|---|---|
| K-fold over windows: windows of one engine land in train **and** validation | every split is grouped by physical **engine id** |
| Hyper-parameters tuned on all data, then "evaluated" with leave-one-engine-out | **nested** Leave-One-Engine-Out: the held-out engine never influences tuning |
| Accuracy only, 12 engines → steps of 8.3 % | Wilson 95 % CI, balanced accuracy, sensitivity on NG, AUC, optional **label-permutation test** |
| Ties in the vote silently became "OK" | ties go to NG (missed faults cost more than false alarms) |
| No way to see how wrong a naive CV is | `--leakage-audit` prints naive-vs-grouped CV for your data |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,xgboost]"      # xgboost / tracking (wandb) are optional extras
pytest                                # 53 tests, ~1 min
```

## Quick start (no private data needed)

```bash
vibnoise synth demo/data --engines-per-class 6 --recordings-per-engine 2   # synthetic compressors
vibnoise inspect demo/data                                                  # check engine grouping!
vibnoise evaluate demo/data --config demo/data/synth_config.yaml \
        --models logreg svm --leakage-audit --out reports/demo
vibnoise train    demo/data --config demo/data/synth_config.yaml --model svm --out models/svm.joblib
vibnoise predict  models/svm.joblib demo/data --out preds.csv
```

> The synthetic set only proves the pipeline works end to end (and it is easy: expect ~100 %).
> It is **not** evidence about real compressors.

## Using your real data

Expected layout (same as the original project): one folder per class, CSVs with
`time,sensor_1,sensor_2` at 1 kHz.

```
data/
├── OK_clean/DAS64100_OK_2_1st.CSV ...
└── NG_clean/DAS56284_NG_3_2nd.CSV ...
```

1. **Verify the engine grouping.** `vibnoise inspect data/` prints the engine id parsed from every
   file name (default regex turns `DAS64100_OK_2_1st.CSV` into engine `OK_2`). If your files
   encode the engine differently, change `data.engine_regex`. *If several recordings of one engine
   were treated as different engines, evaluation would leak again.*
2. Put your 7 operating intervals into the config (`segment.intervals`), or use fixed windows
   (`segment.window_s`) over the whole recording.
3. To compare like-for-like with the original pipeline use `configs/original_compat.yaml`.

## Pipeline

```
CSV ─► strict label/engine discovery ─► zero-phase Butterworth high-pass (+edge trim)
    ─► operating intervals ─► fixed-length windows ─► features (psd | bands | stats, per sensor)
    ─► Pipeline(StandardScaler [→ PCA] → classifier) ─► window P(NG)
    ─► per-engine aggregation (mean-probability + majority vote) ─► OK / NG
```

* **Features**: `psd` (log10 Welch PSD, band-limited), `bands` (log band energies, ~5 dims per
  sensor - fights p ≫ n), `stats` (log-RMS, log-peak, crest factor, skewness, excess kurtosis).
* **Models**: `logreg`, `svm` (calibrated), `knn`, `dt`, `rf`, `xgb` (optional) - one valid grid each.
* **Everything is configurable** through YAML plus `--set section.key=value`; the exact config,
  library versions and a SHA-256 data fingerprint are stored in every report and model bundle.

## Layout

```
src/vibnoise/   config · io · signal · features · dataset · models · evaluation
                training · inference · synth · report · cli
tests/          53 tests (signal, features, io, grouping, leakage regression, every model, CLI e2e)
configs/        default.yaml · original_compat.yaml
docs/           CHANGES_FROM_ORIGINAL.md
```

## Limitations (please read)

* A dozen engines cannot establish generalisation to new machines or operating conditions.
  Report the confidence interval, collect more engines, and treat results as preliminary.
* The synthetic generator is a pipeline test, not a physical model.
* Model bundles are pickles (`joblib`): only load files you created or trust.
* Hyper-parameter grids are small and hand-chosen; wider/Bayesian search is future work.

## Development

```bash
make lint test        # ruff + pytest
```

MIT licensed (see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md)).
