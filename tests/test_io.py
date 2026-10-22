import numpy as np
import pandas as pd
import pytest

from vibnoise.config import DataConfig
from vibnoise.io import discover_recordings, parse_engine_id, read_recording


def _csv(path, n=50, cols=("time", "sensor_1", "sensor_2")):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {c: np.arange(n, dtype=float) / 1000 if c == "time" else np.zeros(n) for c in cols}
    ).to_csv(path, index=False)


def test_parse_engine_id():
    assert parse_engine_id("DAS64100_OK_2_1st.CSV", DataConfig().engine_regex) == ("OK_2", "OK")
    assert parse_engine_id("DAS56284_NG_3_2nd.CSV", DataConfig().engine_regex) == ("NG_3", "NG")
    assert parse_engine_id("whatever.csv", None) == ("whatever", None)
    with pytest.raises(ValueError):
        parse_engine_id("garbage.csv", DataConfig().engine_regex)


def test_recordings_of_one_engine_share_a_group(tmp_path):
    _csv(tmp_path / "OK_clean" / "DAS1_OK_2_1st.CSV")
    _csv(tmp_path / "OK_clean" / "DAS9_OK_2_2nd.CSV")
    _csv(tmp_path / "NG_clean" / "DAS5_NG_1_1st.CSV")
    recs = discover_recordings(tmp_path, DataConfig())
    assert {r.engine_id for r in recs} == {"OK_2", "NG_1"}
    assert sum(r.engine_id == "OK_2" for r in recs) == 2
    assert {r.label for r in recs if r.engine_id == "NG_1"} == {1}


def test_unknown_folder_is_refused_not_labelled_ng(tmp_path):
    _csv(tmp_path / "OK_clean" / "DAS1_OK_1_1st.CSV")
    _csv(tmp_path / "typo_clean" / "DAS2_NG_1_1st.CSV")
    with pytest.raises(ValueError, match="label_map"):
        discover_recordings(tmp_path, DataConfig())


def test_label_and_filename_must_agree(tmp_path):
    _csv(tmp_path / "OK_clean" / "DAS1_NG_1_1st.CSV")
    with pytest.raises(ValueError, match="file name says NG"):
        discover_recordings(tmp_path, DataConfig())


def test_unlabeled_mode_falls_back_to_file_stem(tmp_path):
    _csv(tmp_path / "mystery.csv")
    (rec,) = discover_recordings(tmp_path, DataConfig(), labeled=False)
    assert rec.label is None and rec.engine_id == "mystery"


def test_missing_path_and_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_recordings(tmp_path / "nope", DataConfig())
    with pytest.raises(FileNotFoundError):
        discover_recordings(tmp_path, DataConfig())


def test_read_recording_validation(tmp_path):
    _csv(tmp_path / "a.csv", cols=("time", "sensor_1"))
    with pytest.raises(ValueError, match="missing column"):
        read_recording(tmp_path / "a.csv", DataConfig())
    pd.DataFrame({"sensor_1": [1.0, np.nan], "sensor_2": [1.0, 2.0]}).to_csv(
        tmp_path / "b.csv", index=False
    )
    with pytest.raises(ValueError, match="non-finite"):
        read_recording(tmp_path / "b.csv", DataConfig())
    _csv(tmp_path / "c.csv")
    assert read_recording(tmp_path / "c.csv", DataConfig()).shape == (50, 2)
