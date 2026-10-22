import pytest

from vibnoise.config import Config, apply_overrides, load_config


def test_defaults_are_valid():
    Config().validate()


def test_unknown_section_and_key_rejected():
    with pytest.raises(ValueError, match="Unknown config section"):
        Config.from_dict({"nope": {}})
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_dict({"filter": {"cutoff": 10}})


@pytest.mark.parametrize(
    "bad",
    [
        {"filter": {"cutoff_hz": 600}},
        {"features": {"names": ["wavelet"]}},
        {"features": {"fmin_hz": 300, "fmax_hz": 100}},
        {"eval": {"threshold": 1.5}},
        {"segment": {"window_s": -1}},
        {"data": {"label_map": {"OK_clean": 0, "NG_clean": 2}}},
    ],
)
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        Config.from_dict(bad)


def test_overrides_parse_yaml_values():
    d = apply_overrides(
        {}, ["segment.window_s=5", "features.names=[psd, stats]", "segment.hop_s=null"]
    )
    cfg = Config.from_dict(d)
    assert cfg.segment.window_s == 5
    assert cfg.features.names == ["psd", "stats"]
    assert cfg.segment.hop_s is None


def test_override_syntax_errors():
    with pytest.raises(ValueError):
        apply_overrides({}, ["nokey"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["a.b.c=1"])


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("filter: {order: 4}\n", encoding="utf-8")
    assert load_config(p).filter.order == 4
    assert Config.from_dict(Config().to_dict()) == Config()
