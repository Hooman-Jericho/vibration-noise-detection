"""Human-readable evaluation report."""

from __future__ import annotations

from typing import Any


def _fmt(v: Any, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    d = report["data"]
    lines = [
        "# Vibration noise detection - evaluation report",
        "",
        f"- Created (UTC): {report['created_utc']}",
        f"- Engines: **{d['n_engines']}** ({d['engines_per_class']['OK']} OK / "
        f"{d['engines_per_class']['NG']} NG), recordings: {d['n_recordings']}, "
        f"windows: {d['n_windows']}",
        f"- Features: `{', '.join(report['config']['features']['names'])}` "
        f"({d['n_features']} dims) | validation: nested Leave-One-Engine-Out, "
        f"inner {report['config']['eval']['inner_splits']}-fold grouped CV",
        f"- Data fingerprint: `{d['fingerprint'][:16]}...`",
        "",
        "## Engine-level results (unit of independence = physical engine)",
        "",
        "| model | accuracy [95% Wilson CI] | balanced acc. | sensitivity (NG) "
        "| specificity (OK) | AUC |"
        " permutation p |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, res in report["models"].items():
        e = res["metrics"]["engine"]
        ci = e["accuracy_ci95"]
        perm = res.get("permutation", {}).get("p_value")
        lines.append(
            f"| {name} | {_fmt(e['accuracy'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{_fmt(e['balanced_accuracy'])} | {_fmt(e['sensitivity_ng'])} | "
            f"{_fmt(e['specificity_ok'])} | {_fmt(e['auc'])} | {_fmt(perm)} |"
        )
    leaks = {n: r["leakage_audit"] for n, r in report["models"].items() if "leakage_audit" in r}
    if leaks:
        lines += [
            "",
            "## Leakage audit (naive window K-fold vs. grouped CV, untuned model)",
            "",
            "| model | naive window CV | grouped CV | inflation |",
            "|---|---|---|---|",
        ]
        for n, a in leaks.items():
            lines.append(
                f"| {n} | {_fmt(a['naive_window_cv'])} | {_fmt(a['grouped_cv'])} | "
                f"{_fmt(a['inflation'])} |"
            )
    lines += [
        "",
        "With few engines every engine is worth 1/N of the accuracy; read the confidence "
        "interval, not just the point estimate.",
        "",
    ]
    return "\n".join(lines)
