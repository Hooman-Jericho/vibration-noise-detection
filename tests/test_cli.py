import json

import pandas as pd

from vibnoise.cli import main

FAST = [
    "--set",
    "segment.window_s=2",
    "--set",
    "features.nperseg=256",
    "--set",
    "filter.trim_s=0.5",
]


def test_end_to_end(tmp_path, capsys):
    data = tmp_path / "data"
    assert (
        main(
            [
                "synth",
                str(data),
                "--engines-per-class",
                "3",
                "--recordings-per-engine",
                "2",
                "--interval-seconds",
                "5",
                "4",
                "4",
                "--severity",
                "1.5",
                "-q",
            ]
        )
        == 0
    )
    assert (data / "synth_config.yaml").exists()

    assert main(["inspect", str(data), "-q"]) == 0
    assert "engines with several recordings" in capsys.readouterr().out

    out = tmp_path / "report"
    assert (
        main(
            [
                "evaluate",
                str(data),
                "--models",
                "logreg",
                "--out",
                str(out),
                "--permutations",
                "2",
                "--leakage-audit",
                "-q",
                *FAST,
            ]
        )
        == 0
    )
    report = json.loads((out / "report.json").read_text())
    res = report["models"]["logreg"]
    assert report["data"]["n_engines"] == 6
    assert report["data"]["engines_per_class"] == {"OK": 3, "NG": 3}
    assert 0 <= res["permutation"]["p_value"] <= 1 and "inflation" in res["leakage_audit"]
    assert "Wilson" in (out / "report.md").read_text()
    assert len(pd.read_csv(out / "engines_logreg.csv")) == 6

    model = tmp_path / "m.joblib"
    assert main(["train", str(data), "--model", "logreg", "--out", str(model), "-q", *FAST]) == 0
    preds = tmp_path / "p.csv"
    assert main(["predict", str(model), str(data), "--out", str(preds), "-q"]) == 0
    df = pd.read_csv(preds)
    assert set(df["decision"]) <= {"OK", "NG"} and len(df) == 6


def test_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing"), "-q"]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["inspect", str(tmp_path), "--set", "bogus.key=1", "-q"]) == 2
