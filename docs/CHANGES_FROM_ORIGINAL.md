# Changes from the original project

Each row: problem found in the original code → fix → where it is guarded.
"Test" names refer to `tests/`.

## A. Scientific validity

| # | Original problem | Fix | Guard |
|---|---|---|---|
| 1 | `GridSearchCV` / `cross_val_score` over windows: windows of one engine in train and validation | `StratifiedGroupKFold` on engine id everywhere (`evaluation.make_group_splits`) | `test_group_splits_never_share_an_engine`, `test_naive_window_cv_is_inflated_but_grouped_cv_is_not` |
| 2 | Hyper-parameters chosen on all data, then leave-one-engine-out voting reused them | Nested Leave-One-Engine-Out: tuning happens inside each outer training set (`evaluation.nested_loeo`) | `test_nested_loeo_does_not_hallucinate_signal` |
| 3 | `classification_report` printed on the *training* data | Removed; only held-out metrics are reported | - |
| 4 | "Engine" = file name, so several recordings of one engine could be split | Engine id parsed from the file name by a configurable regex; `inspect` shows the grouping | `test_recordings_of_one_engine_share_a_group` |
| 5 | 12 engines → accuracy in steps of 8.3 %, no uncertainty | Wilson 95 % CI, balanced accuracy, sensitivity/specificity, AUC, label-permutation test | `test_wilson_interval_properties`, `test_permutation_test_is_calibrated` |
| 6 | p ≫ n (2×513 PSD bins vs. tens of samples) | band-limited PSD, low-dimensional `bands`/`stats` features, optional PCA, fixed-length windows to get more samples | feature tests |
| 7 | No scaling for SVM/KNN | `Pipeline(StandardScaler → clf)`; scaler fitted on training folds only | `test_pipeline_scaler_is_fitted_on_training_data_only` |
| 8 | Vote ties → OK | Ties → NG (documented, configurable threshold) | `test_ties_go_to_ng` |
| 9 | Accuracy only | + balanced accuracy, sensitivity on NG, specificity, AUC, confusion matrix, leakage audit | `test_end_to_end` |
| 10 | Best parameters on grid edges, DT with 44 leaves on tiny data | Depth / leaf-size grids that regularise; edge hits are visible in `best_params_per_engine` of `report.json` | - |

## B. Bugs

| # | Original problem | Fix | Guard |
|---|---|---|---|
| 11 | `predict.py` crashed for KNN / DT / XGB (`probability=True` only exists on SVC) | all models expose `predict_proba`; SVM uses `CalibratedClassifierCV` | `test_every_model_persists_and_predicts_on_new_data[*]` |
| 12 | Predict loop re-predicted the whole test set for every engine | decisions are computed per engine from its own windows only | `test_prediction_is_per_engine_not_pooled` |
| 13 | Model not persisted; training data needed at inference | self-describing `joblib` bundle (pipeline, config, feature names, versions, data hash) | persistence test, `load_bundle` version warning |
| 14 | `0 if folder == "OK_clean" else 1`: any typo became NG | strict `label_map`; unknown folder or file-name/folder disagreement raises | `test_unknown_folder_is_refused_not_labelled_ng`, `test_label_and_filename_must_agree` |
| 15 | Double `sort_values` (second overrides first, quicksort unstable) | no sorting: sensors are concatenated per window by construction, so alignment is guaranteed | `test_features_match_names_and_are_finite_on_silence` |
| 16 | `select_model(OFF)` returned shared module-level estimators | models are built per call | `make_pipeline` |
| 17 | Crash after hours of GridSearch (`out_file` missing, `wandb` login) | output dirs created up front; `wandb` is an opt-in extra | `test_end_to_end` |
| 18 | `max_features="auto"`, invalid LogReg combos, hidden by `filterwarnings("ignore")` | only valid grids, `error_score="raise"`, no global warning suppression | `test_grids_only_use_valid_parameters[*]` |
| 19 | `np.reshape` of ragged lists in `__len__` | removed | - |
| 20 | `log` of kurtosis (can be negative → NaN), "wavelet energy" was RMS, `log(0) = -inf` PSD | bounded, NaN-free features with `eps`; honest names | `test_features_match_names_and_are_finite_on_silence`, `test_constant_signal_stats_not_nan` |

## C. Signal processing

| # | Original problem | Fix | Guard |
|---|---|---|---|
| 21 | Causal order-20 filter (start-up transient inside the first interval, phase distortion) | zero-phase `sosfiltfilt` (default) + `trim_s` edge trim; causal filter still selectable | `test_zero_phase_has_no_start_transient_or_lag` |
| 22 | Magic `time_points`, last sample dropped by `-1` | intervals live in the config, python-style negative indices, out-of-range intervals reported | `test_resolve_intervals_semantics` |
| 23 | README promised small chunks, code produced 7 big ones | real fixed-length windows (`window_s`, `hop_s`) inside each interval | `test_iter_windows` |
| 24 | Notebook used cutoff 2 Hz, CLI 10 Hz | single source of truth: the config | `test_config` |

## D. Engineering

| # | Original problem | Fix |
|---|---|---|
| 25 | `sys.path.append(os.getcwd())`, star imports | `src/` layout, installable package, `vibnoise` console script |
| 26 | Wrong paths / argument names in README | README commands are exercised by `test_end_to_end` |
| 27 | No tests, no CI, no seeds | 53 tests, GitHub Actions (3.10-3.13), ruff, seeded everywhere |
| 28 | Incomplete `requirements.txt`, `pywt` missing, unused deps | dependencies declared in `pyproject.toml`; extras for xgboost / wandb |
| 29 | No `.gitignore`, no LICENSE file | added |
| 30 | Data not described | documented layout, `inspect`, synthetic generator for demos |

## E. Not changed on purpose

* The data are not included (private), so **no claim about real-world accuracy is made here**.
  Run `vibnoise evaluate` on your data and report the engine-level CI.
* The original exploratory notebook is not shipped; it duplicated library code and referenced
  stale files. Its plots can be recreated from `FeatureTable` / `signal` functions.
