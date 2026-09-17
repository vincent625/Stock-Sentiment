# QQQ News Sentiment → Price Reaction Study

A reproducible research pipeline for answering two questions:

1. **Which news-sentiment lookback `x` has the strongest relationship with forward abnormal return over horizon `y`?**
2. **How quickly does the market incorporate the sentiment signal?**

The project correlates sentiment with **future abnormal returns**, not price levels. It uses chronological folds for selection, then block-bootstrap confidence intervals, HAC statistics, and Benjamini-Hochberg FDR correction on the untouched final test fold so the result is not just the largest in-sample correlation from a large `(x, y)` search.

## What it produces

- `grid_metrics_by_fold.csv` — lightweight validation stability metrics for each lookback × horizon × chronological fold
- `grid_metrics_validation.csv` — validation-fold IC and stability statistics
- `grid_metrics_test.csv` — untouched final-fold grid with bootstrap CI, HAC and FDR inference
- `ic_heatmap_validation.png` and `ic_heatmap_test.png` — sentiment lookback × forward-return horizon
- `reaction_curve_test.csv` — final-fold Q5 minus Q1 cumulative abnormal-return response
- `reaction_speed_metrics_test.csv` — final-fold `T50` and `T80`
- `reaction_curve_test.png` — final-fold market-reaction-speed curve
- `predictive_panel.csv` and `event_reaction_panel.csv` — inspectable research panels

Plots use a restrained nature-style earth/forest palette.

## Core definitions

For article `j` affecting stock `i`:

```text
article_score = sentiment × relevance × novelty × source_quality
```

At time `t`, sentiment over lookback `x` is an exponentially decayed weighted average of only news that was already public by `t`.

The default outcome is market-adjusted forward log return:

```text
abnormal_return(i,t,y) = stock_log_return(i,t→t+y) - QQQ_log_return(t→t+y)
```

Set `returns.abnormal_method: beta_adjusted` in the YAML config to use a rolling pre-event beta instead.

Reaction speed is measured from the cumulative abnormal-return spread between the highest and lowest sentiment quintiles:

```text
SpreadCAR(h) = mean(CAR | top sentiment quintile) - mean(CAR | bottom quintile)
```

`T50` and `T80` are the earliest horizons that reach 50% and 80% of the pre-specified terminal response.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run the synthetic end-to-end demo

This is only a pipeline test. It deliberately injects a decaying sentiment response into synthetic intraday prices.

```bash
qqq-sentiment demo --output-dir outputs/demo
```

Then inspect:

```text
outputs/demo/ic_heatmap_validation.png
outputs/demo/ic_heatmap_test.png
outputs/demo/reaction_curve_test.png
outputs/demo/summary.csv
```

## Run on real data

Normalize your feeds to the schemas in [`docs/data_contract.md`](docs/data_contract.md), then:

```bash
qqq-sentiment run \
  --news /path/to/news.parquet \
  --prices /path/to/intraday_prices.parquet \
  --config config.example.yaml \
  --output-dir outputs/real
```

CSV is supported with the base install. Parquet is also supported with `pip install -e ".[parquet]"`.

`config.example.yaml` contains the full 5-minute-to-20-day lookback and 1-minute-to-20-day forward-horizon grid. The built-in no-config defaults are shorter so the synthetic demo runs quickly.

## Recommended production data

For the definitive intraday study, use a timestamped institutional news feed with entity mapping, relevance, novelty and story versioning, plus institutional intraday trades/quotes. A strong setup is **LSEG Machine Readable News / News Analytics + LSEG Tick History**; **RavenPack News Analytics + institutional tick data** is a strong alternative. For long-history daily robustness, CRSP is preferable to convenience price APIs.

This repository does not embed paid-provider credentials or assume proprietary column names. Map vendor exports to the small normalized contract instead.

## Important methodology choices

- News is never allowed into a signal before `first_timestamp_utc`.
- Predictive tests start from the **first stock bar strictly after publication**.
- Reaction-speed tests use the **last stock price known at or before publication** so the immediate response is retained.
- Repeated syndicated stories are deduplicated by `story_id × ticker`.
- GOOG/GOOGL should share an `issuer_id` and should not be treated as two independent issuer observations in a cross-sectional robustness study.
- Use minute/finer data for market-reaction speed; daily close-to-close data is too coarse for that question.
- Whole market-calendar dates are assigned to chronological folds using `market_timezone`. The last fold is held out as a final test, and validation observations whose maximum forward-return horizon would cross the test boundary are purged. `(x, y)` selection uses only the remaining earlier folds and is ranked by fold stability, not raw `argmax(correlation)` alone.
- Treat the current QQQ top 10 as a convenience universe only. A historical production backtest should use **point-in-time QQQ constituents** to avoid survivorship bias.


For publication-quality inference, increase `validation.bootstrap_samples` in the YAML (for example to 1,000-5,000). The example defaults to 500 to keep exploratory runs practical.

## Tests

```bash
pytest -q
```

## Repository layout

```text
src/qqq_sentiment/
  analysis.py   # predictive grid + event reaction study
  returns.py    # event/bar alignment and abnormal returns
  signals.py    # lagged decayed sentiment
  stats.py      # bootstrap, HAC, FDR, chronological folds
  plots.py      # heatmap and reaction-speed visualizations
  synthetic.py  # deterministic end-to-end demo
  cli.py        # command-line entry point

tests/
docs/data_contract.md
config.example.yaml
```

## Upload to GitHub

After unzipping the project:

```bash
git init
git add .
git commit -m "Initial news sentiment reaction study"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

The included GitHub Actions workflow runs unit tests plus an end-to-end synthetic smoke test on pushes and pull requests.

## Research interpretation

A useful result is not one isolated bright heatmap cell. Look for a **stable region** that keeps the same sign across chronological folds, has adequate coverage, survives FDR/uncertainty checks, and remains present in genuinely unseen periods. Correlation is predictive association, not proof that news sentiment caused the price move.
