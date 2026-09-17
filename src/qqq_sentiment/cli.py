from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import (
    aggregate_grid_metrics,
    build_event_reaction_panel,
    build_predictive_panel,
    grid_metrics,
    robust_test_grid_metrics,
    reaction_curve,
    reaction_time_metrics,
)
from .config import load_config
from .io import read_table, write_table
from .plots import plot_ic_heatmap, plot_reaction_curve
from .synthetic import generate_demo


def _run(news: pd.DataFrame, prices: pd.DataFrame, output_dir: Path, config_path: str | None) -> None:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictive = build_predictive_panel(news, prices, config)
    all_folds = sorted(predictive["fold"].dropna().unique())
    if len(all_folds) < 2:
        raise RuntimeError("At least two chronological folds are required for validation/test separation.")

    test_fold = int(all_folds[-1])
    test_start = predictive.loc[predictive["fold"] == test_fold, "asof"].min()
    max_horizon = max(pd.Timedelta(h) for h in config.horizons)
    validation_panel = predictive[
        (predictive["fold"] != test_fold)
        & (predictive["asof"] + max_horizon < test_start)
    ].copy()
    fold_grid = grid_metrics(validation_panel, config)
    folds = sorted(fold_grid["fold"].dropna().unique())
    if not folds:
        raise RuntimeError("The test embargo removed all validation observations; provide more history.")

    validation_fold_grid = fold_grid.copy()
    test_grid = robust_test_grid_metrics(predictive, config, test_fold)
    validation_grid = aggregate_grid_metrics(validation_fold_grid)

    write_table(predictive, output_dir / "predictive_panel.csv")
    write_table(fold_grid, output_dir / "grid_metrics_by_fold.csv")
    write_table(validation_grid, output_dir / "grid_metrics_validation.csv")
    write_table(test_grid, output_dir / "grid_metrics_test.csv")

    plot_ic_heatmap(
        validation_grid,
        lookback_order=config.lookbacks,
        horizon_order=config.horizons,
        output=output_dir / "ic_heatmap_validation.png",
        subtitle="Validation folds: mean Spearman IC; earth = negative, forest = positive",
    )
    test_for_plot = test_grid.rename(columns={"spearman": "mean_spearman"})
    plot_ic_heatmap(
        test_for_plot,
        lookback_order=config.lookbacks,
        horizon_order=config.horizons,
        output=output_dir / "ic_heatmap_test.png",
        subtitle="Final untouched test fold: Spearman IC; earth = negative, forest = positive",
    )

    valid = validation_grid[
        (validation_grid["mean_n"] >= config.validation.min_observations)
        & validation_grid["mean_spearman"].notna()
    ].copy()
    if valid.empty:
        raise RuntimeError("No validation grid cell has enough observations; lower min_observations or provide more data.")

    selected = valid.sort_values(
        ["stability_score", "positive_fold_fraction", "mean_spearman"], ascending=False
    ).iloc[0]
    selected_lookback = str(selected["lookback"])
    selected_horizon = str(selected["horizon"])
    selected_test = test_grid[
        (test_grid["lookback"] == selected_lookback) & (test_grid["horizon"] == selected_horizon)
    ]
    test_spearman = float(selected_test.iloc[0]["spearman"]) if len(selected_test) else float("nan")
    test_fdr_p = float(selected_test.iloc[0]["fdr_p"]) if len(selected_test) else float("nan")

    event_panel = build_event_reaction_panel(news, prices, config)
    test_events = event_panel[event_panel["asof"] >= test_start].copy()
    curve = reaction_curve(
        test_events,
        lookback=selected_lookback,
        horizons=config.horizons,
        fold=None,
        min_observations=config.validation.min_observations,
    )
    metrics = reaction_time_metrics(curve, terminal_horizon="4h" if "4h" in config.horizons else None)

    write_table(event_panel, output_dir / "event_reaction_panel.csv")
    write_table(curve, output_dir / "reaction_curve_test.csv")
    pd.DataFrame([metrics]).to_csv(output_dir / "reaction_speed_metrics_test.csv", index=False)
    if not curve.empty:
        plot_reaction_curve(curve, metrics, output_dir / "reaction_curve_test.png")

    summary = pd.DataFrame(
        [
            {
                "validation_folds": ",".join(map(str, folds)),
                "test_fold": test_fold,
                "test_start_utc": test_start,
                "embargo": str(max_horizon),
                "selected_lookback": selected_lookback,
                "selected_horizon": selected_horizon,
                "validation_mean_spearman": selected["mean_spearman"],
                "validation_positive_fold_fraction": selected["positive_fold_fraction"],
                "validation_stability_score": selected["stability_score"],
                "test_spearman": test_spearman,
                "test_fdr_p": test_fdr_p,
                **metrics,
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="News sentiment versus forward abnormal-return study")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate deterministic synthetic data and run the full pipeline")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--config", default=None)
    demo.add_argument("--days", type=int, default=15)

    run = sub.add_parser("run", help="Run the study on normalized news and price files")
    run.add_argument("--news", required=True)
    run.add_argument("--prices", required=True)
    run.add_argument("--output-dir", default="outputs/run")
    run.add_argument("--config", default=None)

    args = parser.parse_args()
    if args.command == "demo":
        news, prices = generate_demo(n_days=args.days)
        demo_data = Path(args.output_dir) / "input"
        write_table(news, demo_data / "news.csv")
        write_table(prices, demo_data / "prices.csv")
        _run(news, prices, Path(args.output_dir), args.config)
    elif args.command == "run":
        _run(read_table(args.news), read_table(args.prices), Path(args.output_dir), args.config)


if __name__ == "__main__":
    main()
