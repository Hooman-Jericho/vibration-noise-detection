"""Command line interface: ``vibnoise {synth,inspect,evaluate,train,predict}``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from . import __version__
from .config import Config, load_config
from .dataset import FeatureTable, build_feature_table
from .evaluation import leakage_audit, nested_loeo, permutation_test
from .inference import predict_recordings
from .io import discover_recordings, fingerprint
from .models import available_models
from .report import render_markdown
from .synth import generate_dataset
from .training import fit_final, load_bundle, save_bundle
from .utils import clean_json, env_info, seed_everything, setup_logging

log = logging.getLogger("vibnoise")


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, help="YAML config (see configs/default.yaml)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set segment.window_s=5 (repeatable)",
    )
    p.add_argument(
        "--features",
        nargs="+",
        choices=["psd", "bands", "stats"],
        help="feature set (overrides features.names)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vibnoise", description=__doc__)
    ap.add_argument("--version", action="version", version=f"vibnoise {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    common = _common()

    s = sub.add_parser("synth", parents=[common], help="write a synthetic demo data set")
    s.add_argument("out", type=Path)
    s.add_argument("--engines-per-class", type=int, default=6)
    s.add_argument("--recordings-per-engine", type=int, default=2)
    s.add_argument("--interval-seconds", type=float, nargs="+", default=[12, 10, 8, 8, 6, 6, 6])
    s.add_argument("--severity", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)

    i = sub.add_parser("inspect", parents=[common], help="show recordings, engines and classes")
    i.add_argument("data", type=Path)

    e = sub.add_parser("evaluate", parents=[common], help="nested Leave-One-Engine-Out evaluation")
    e.add_argument("data", type=Path)
    e.add_argument("--models", nargs="+", choices=available_models())
    e.add_argument("--out", type=Path, default=Path("reports/latest"))
    e.add_argument("--permutations", type=int, help="label-permutation test with N shuffles")
    e.add_argument("--leakage-audit", action="store_true", help="also quantify naive-CV inflation")
    e.add_argument(
        "--wandb", action="store_true", help="log metrics to Weights & Biases (optional)"
    )

    t = sub.add_parser("train", parents=[common], help="fit the final model on all engines")
    t.add_argument("data", type=Path)
    t.add_argument("--model", choices=available_models())
    t.add_argument("--out", type=Path, default=Path("models/model.joblib"))

    p = sub.add_parser("predict", parents=[common], help="predict unseen recordings")
    p.add_argument("model", type=Path)
    p.add_argument("path", type=Path, help="CSV file or directory")
    p.add_argument("--threshold", type=float, help="decision threshold on mean P(NG)")
    p.add_argument("--out", type=Path, help="write engine table to .csv or .json")
    p.add_argument("--windows", action="store_true", help="also print window-level probabilities")
    return ap


def _load_cfg(args: argparse.Namespace) -> Config:
    overrides = list(args.set)
    cfg = load_config(args.config, overrides)
    if getattr(args, "features", None):
        cfg.features.names = list(args.features)
    if getattr(args, "model", None):
        cfg.model.name = args.model
    cfg.validate()
    return cfg


def _table(data: Path, cfg: Config) -> tuple[FeatureTable, list, str]:
    recs = discover_recordings(data, cfg.data)
    return build_feature_table(recs, cfg), recs, fingerprint(recs)


def _cmd_synth(args: argparse.Namespace) -> int:
    suggested = generate_dataset(
        args.out,
        args.engines_per_class,
        args.recordings_per_engine,
        interval_seconds=tuple(args.interval_seconds),
        severity=args.severity,
        seed=args.seed,
    )
    print(
        f"Wrote synthetic data to {args.out} (config suggestion: {args.out / 'synth_config.yaml'})"
    )
    print(f"intervals: {suggested['segment']['intervals']}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    recs = discover_recordings(args.data, cfg.data)
    df = pd.DataFrame(
        [{"engine_id": r.engine_id, "label": r.label, "recording": r.recording_id} for r in recs]
    )
    print(df.to_string(index=False))
    per_engine = df.drop_duplicates("engine_id")["label"].value_counts().to_dict()
    print(
        f"\n{len(recs)} recordings, {df['engine_id'].nunique()} engines; "
        f"engines per class (0=OK, 1=NG): {per_engine}"
    )
    multi = {k: v for k, v in Counter(df["engine_id"]).items() if v > 1}
    if multi:
        print(f"engines with several recordings (kept in the same CV group): {multi}")
    if min(per_engine.values(), default=0) < 2 or len(per_engine) < 2:
        print("WARNING: fewer than 2 engines in a class - grouped evaluation is impossible.")
        return 1
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    seed_everything(cfg.model.seed)
    if args.permutations is not None:
        cfg.eval.permutations = args.permutations
    table, recs, fp = _table(args.data, cfg)
    labels = table.engine_labels()
    per_class = {
        "OK": sum(v == 0 for v in labels.values()),
        "NG": sum(v == 1 for v in labels.values()),
    }
    models = args.models or [cfg.model.name]
    report: dict = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": env_info(),
        "config": cfg.to_dict(),
        "data": {
            "n_recordings": len(recs),
            "n_engines": len(labels),
            "n_windows": len(table),
            "n_features": table.X.shape[1],
            "engines_per_class": per_class,
            "fingerprint": fp,
        },
        "models": {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name in models:
        res = nested_loeo(table.X, table.y, table.groups, name, cfg)
        entry: dict = {"metrics": res.metrics, "best_params_per_engine": res.best_params}
        res.engines.to_csv(args.out / f"engines_{name}.csv", index=False)
        res.windows.to_csv(args.out / f"windows_{name}.csv", index=False)
        if cfg.eval.permutations > 0:
            entry["permutation"] = permutation_test(
                table.X,
                table.y,
                table.groups,
                name,
                cfg,
                cfg.eval.permutations,
                res.metrics["engine"]["balanced_accuracy"],
            )
        if args.leakage_audit:
            entry["leakage_audit"] = leakage_audit(table.X, table.y, table.groups, name, cfg)
        report["models"][name] = entry
        e = res.metrics["engine"]
        print(
            f"{name:7s} engine acc={e['accuracy']:.3f} "
            f"CI95=[{e['accuracy_ci95'][0]:.3f}, {e['accuracy_ci95'][1]:.3f}] "
            f"bal.acc={e['balanced_accuracy']:.3f} auc={e['auc']:.3f}"
        )
    report = clean_json(report)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if args.wandb:
        import wandb  # optional dependency

        run = wandb.init(project="vibration-noise-detection", config=cfg.to_dict())
        for name, entry in report["models"].items():
            run.log(
                {
                    f"{name}/{k}": v
                    for k, v in entry["metrics"]["engine"].items()
                    if isinstance(v, (int, float))
                }
            )
        run.finish()
    print(f"Report written to {args.out}/report.md")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    seed_everything(cfg.model.seed)
    table, _, fp = _table(args.data, cfg)
    est, params, score = fit_final(table, cfg.model.name, cfg)
    path = save_bundle(
        args.out,
        est,
        cfg,
        table,
        {
            "model": cfg.model.name,
            "best_params": clean_json(params),
            "grouped_cv_score": score,
            "data_fingerprint": fp,
        },
    )
    print(f"Saved {cfg.model.name} bundle to {path} (grouped CV {cfg.eval.scoring}={score:.3f})")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    engines, windows = predict_recordings(bundle, args.path, args.threshold)
    print(engines.drop(columns=["recordings"]).to_string(index=False, float_format="%.3f"))
    if args.windows:
        print(windows.to_string(index=False, float_format="%.3f"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".json":
            args.out.write_text(
                json.dumps(clean_json(engines.to_dict("records")), indent=2), encoding="utf-8"
            )
        else:
            engines.to_csv(args.out, index=False)
    return 0


_COMMANDS = {
    "synth": _cmd_synth,
    "inspect": _cmd_inspect,
    "evaluate": _cmd_evaluate,
    "train": _cmd_train,
    "predict": _cmd_predict,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)
    try:
        return _COMMANDS[args.command](args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
