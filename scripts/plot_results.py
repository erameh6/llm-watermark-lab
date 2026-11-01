#!/usr/bin/env python3
"""Create publication-style diagnostic plots from one experiment report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to report_*.json")
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples_path = Path(report["artifacts"]["samples_csv"])
    attacks_path = Path(report["artifacts"]["attacks_csv"])
    if not samples_path.is_absolute():
        samples_path = report_path.parent / samples_path
    if not attacks_path.is_absolute():
        attacks_path = report_path.parent / attacks_path
    samples = pd.read_csv(samples_path)
    attacks = pd.read_csv(attacks_path)
    out_dir = report_path.parent / "figures" / report["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in samples.groupby("label"):
        ax.hist(group["z_score"].dropna(), bins=12, alpha=0.55, label=label)
    ax.axvline(report["configuration"]["z_threshold"], linestyle="--",
               label="detection threshold")
    ax.set(title="Watermark detector z-score distributions",
           xlabel="Detector z-score", ylabel="Sample count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "z_score_distribution.png", dpi=160)
    plt.close(fig)

    length_summary = report.get("length_sensitivity", {})
    if length_summary:
        lengths = sorted(length_summary, key=lambda x: int(x))
        tpr = [length_summary[x]["true_positive_rate"] for x in lengths]
        fpr = [length_summary[x]["false_positive_rate"] for x in lengths]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lengths, tpr, marker="o", label="True-positive rate")
        ax.plot(lengths, fpr, marker="s", label="False-positive rate")
        ax.set(title="Detection performance by generation length",
               xlabel="Requested generated tokens", ylabel="Rate", ylim=(0, 1.05))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "length_sensitivity.png", dpi=160)
        plt.close(fig)

    if not attacks.empty:
        summary = attacks.groupby("attack")["prediction"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(summary.index, summary.values)
        ax.set(title="Watermark detection after text perturbations",
               xlabel="Perturbation", ylabel="Detection rate", ylim=(0, 1.05))
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "attack_robustness.png", dpi=160)
        plt.close(fig)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
