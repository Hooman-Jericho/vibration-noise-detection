from __future__ import annotations

import pytest

from vibnoise.config import Config
from vibnoise.synth import generate_dataset

SMALL_INTERVALS = (5, 4, 4)


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    cfg = Config.from_dict(
        {
            "segment": {"window_s": 2},
            "features": {"nperseg": 256},
            "filter": {"trim_s": 0.5},
        }
    )
    return cfg


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("synth")
    generate_dataset(
        out,
        engines_per_class=4,
        recordings_per_engine=2,
        interval_seconds=SMALL_INTERVALS,
        severity=3.0,
        seed=1,
    )
    return str(out)
