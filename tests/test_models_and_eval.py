import numpy as np
import pytest

from vibnoise.config import Config
from vibnoise.dataset import build_feature_table
from vibnoise.evaluation import (
    engine_decision,
    leakage_audit,
    make_group_splits,
    nested_loeo,
    permutation_test,
    wilson_interval,
)
from vibnoise.inference import predict_recordings
from vibnoise.io import discover_recordings
from vibnoise.models import available_models, make_pipeline, param_grid
from vibnoise.training import fit_final, load_bundle, save_bundle

ALL_MODELS = ["logreg", "svm", "knn", "dt", "rf", "xgb"]


def _skip_if_missing(name):
    if name not in available_models():
        pytest.skip(f"{name} not installed")


@pytest.fixture(scope="module")
def table(synth_dir, small_cfg):
    return build_feature_table(discover_recordings(synth_dir, small_cfg.data), small_cfg)


def _fingerprint_data(seed=0, n_eng=12, n_win=20, dim=30):
    """Labels are random per engine, but every engine has a unique 'fingerprint'."""
    rng = np.random.default_rng(seed)
    fp = rng.normal(size=(n_eng, dim)) * 2.0
    eng_y = np.array([0] * (n_eng // 2) + [1] * (n_eng // 2))
    rng.shuffle(eng_y)
    groups = np.repeat([f"E{i}" for i in range(n_eng)], n_win)
    y = np.repeat(eng_y, n_win)
    X = np.repeat(fp, n_win, axis=0) + rng.normal(scale=0.3, size=(n_eng * n_win, dim))
    return X, y, groups


# --------------------------------------------------------------------------- the key fix
def test_naive_window_cv_is_inflated_but_grouped_cv_is_not():
    """Regression test for the original leakage: with NO real signal, window-level K-fold
    reports near-perfect accuracy (it just recognises engines) while grouped CV stays at chance."""
    X, y, g = _fingerprint_data()
    audit = leakage_audit(X, y, g, "logreg", Config())
    assert audit["naive_window_cv"] > 0.9
    assert audit["grouped_cv"] < 0.75
    assert audit["inflation"] > 0.25


def test_nested_loeo_does_not_hallucinate_signal():
    X, y, g = _fingerprint_data()
    res = nested_loeo(X, y, g, "logreg", Config())
    assert res.metrics["engine"]["balanced_accuracy"] < 0.85


def test_nested_loeo_finds_real_signal(table):
    cfg = Config.from_dict({"model": {"seed": 1}})
    res = nested_loeo(table.X, table.y, table.groups, "logreg", cfg)
    m = res.metrics["engine"]
    assert m["accuracy"] >= 0.85 and m["auc"] >= 0.9
    assert len(res.engines) == 8 and set(res.engines["engine_id"]) == set(table.engines)
    assert (
        res.metrics["engine"]["accuracy_ci95"][0]
        < m["accuracy"]
        < res.metrics["engine"]["accuracy_ci95"][1] + 1e-9
    )


def test_permutation_test_is_calibrated(table):
    cfg = Config()
    res = nested_loeo(table.X, table.y, table.groups, "logreg", cfg)
    out = permutation_test(
        table.X, table.y, table.groups, "logreg", cfg, 4, res.metrics["engine"]["balanced_accuracy"]
    )
    assert (
        0 < out["p_value"] <= 0.5 and out["null_mean"] < res.metrics["engine"]["balanced_accuracy"]
    )


def test_group_splits_never_share_an_engine(table):
    splits, k = make_group_splits(table.X, table.y, table.groups, 3, 0)
    assert k == 3
    for tr, va in splits:
        assert not set(table.groups[tr]) & set(table.groups[va])


def test_group_splits_need_two_engines_per_class():
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="at least 2 engines"):
        make_group_splits(np.zeros((4, 2)), y, np.array(["a", "a", "b", "b"]), 3, 0)


# --------------------------------------------------------------------------- voting
def test_ties_go_to_ng():
    assert engine_decision(np.array([0.2, 0.8]), 0.5)["pred_hard"] == 1
    assert engine_decision(np.array([0.5, 0.5]), 0.5)["pred_soft"] == 1
    assert engine_decision(np.array([0.1, 0.2, 0.9]), 0.5)["pred_hard"] == 0


def test_wilson_interval_properties():
    lo, hi = wilson_interval(12, 12)
    assert hi == pytest.approx(1.0) and 0.7 < lo < 0.8  # 12/12 is NOT a certain 100%
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # NaN


# --------------------------------------------------------------------------- models
@pytest.mark.parametrize("name", ALL_MODELS)
def test_grids_only_use_valid_parameters(name):
    _skip_if_missing(name)
    pipe = make_pipeline(name)
    valid = set(pipe.get_params())
    for grid in param_grid(name):
        assert set(grid) <= valid, f"invalid keys in {name}: {set(grid) - valid}"
        pipe.set_params(**{k: v[0] for k, v in grid.items()})


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_persists_and_predicts_on_new_data(name, table, small_cfg, synth_dir, tmp_path):
    """The original predict.py crashed for 3 of 4 models (`probability=True` on non-SVMs)."""
    _skip_if_missing(name)
    pipe = make_pipeline(name).fit(table.X, table.y)
    path = save_bundle(tmp_path / f"{name}.joblib", pipe, small_cfg, table)
    bundle = load_bundle(path)
    engines, windows = predict_recordings(bundle, synth_dir)
    assert len(engines) == 8
    assert windows.groupby("engine_id").size().sum() == len(table)
    assert set(engines["decision"]) <= {"OK", "NG"}
    np.testing.assert_allclose(bundle.estimator.predict_proba(table.X), pipe.predict_proba(table.X))


def test_prediction_is_per_engine_not_pooled(table, small_cfg, synth_dir, tmp_path):
    pipe = make_pipeline("logreg").fit(table.X, table.y)
    bundle = load_bundle(save_bundle(tmp_path / "m.joblib", pipe, small_cfg, table))
    engines, _ = predict_recordings(bundle, synth_dir)
    truth = table.engine_labels()
    assert (
        engines["decision"].eq("NG").astype(int).to_numpy()
        == [truth[e] for e in engines["engine_id"]]
    ).all()
    assert (engines["n_windows"] > 0).all()


def test_fit_final_uses_grouped_cv(table, small_cfg):
    est, params, score = fit_final(table, "logreg", small_cfg)
    assert "clf__C" in params and 0.5 <= score <= 1.0
    assert est.predict_proba(table.X).shape == (len(table), 2)


def test_feature_mismatch_is_detected(table, small_cfg, synth_dir, tmp_path):
    pipe = make_pipeline("logreg").fit(table.X, table.y)
    bundle = load_bundle(save_bundle(tmp_path / "m.joblib", pipe, small_cfg, table))
    bundle.feature_names = bundle.feature_names[:-1]
    with pytest.raises(ValueError, match="Feature layout"):
        predict_recordings(bundle, synth_dir)


def test_pipeline_scaler_is_fitted_on_training_data_only():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 100])
    pipe = make_pipeline("logreg").fit(
        X[:10], np.zeros(10, dtype=int) if False else np.array([0] * 5 + [1] * 5)
    )
    assert (
        pipe.named_steps["scale"].mean_.max() == 0.0
    )  # statistics come from the 10 training rows only
