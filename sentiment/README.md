# Sentiment-Informed Forecast-Driven Portfolio Optimization — Indian Market Replication

Replication of Manish & Chahal (2026), *"Sentiment-informed forecast-driven portfolio
optimization with alternative risk measures and strategic commodity diversification"*
(Research in International Business and Finance, 91, 103620), adapted to Indian equities
and commodity futures.

## Mapping from the original (US) study to the Indian market

| Original (US)                                            | Indian replication                                                                 |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------|
| 9 SPDR sectoral ETFs (XLK, XLF, XLY, XLV, XLI, XLP, XLE, XLB, XLU) | 9 Indian sectors (Bank, IT, FMCG, Pharma, Auto, Energy, Metal, Realty, Financial Services), **self-built**: market-cap-weighted basket of top constituent stocks priced from raw NSE OHLCV, not a licensed index/ETF feed |
| 12 commodity futures: 4 energy, 4 metal, 4 agri (CME/NYMEX/COMEX) | MCX futures (energy: crude oil, natural gas, ...) + MCX metals (gold, silver, copper, zinc) + NCDEX agri (soybean, mustard seed, cotton, guar seed / chana) |
| Bloomberg (ETFs) + Yahoo Finance (futures)                | Self-built stock baskets (via yfinance) for sectors + MCX/NCDEX historical data for commodities |
| LSEG MarketPsych Analytics sentiment (9 indicators)        | **Self-built**: scrape Indian financial news (Economic Times, Moneycontrol, Business Standard, Mint), score with FinBERT / LLM, aggregate into the same 9 indicators |
| 3-month US T-bill risk-free rate                           | 91-day Government of India T-bill rate |
| DJI benchmark                                               | Nifty 50 / Sensex benchmark |
| VIX (regime split)                                          | India VIX (regime split) |
| High-vol regime: Jul 2014–Dec 2022 (test Jan 2020–Dec 2022) | **User-provided** — placeholder dates in `config/config.yaml`, to be filled in directly |
| Normal regime: Jul 2017–Dec 2025 (test Jan 2023–Dec 2025)   | **User-provided** — placeholder dates in `config/config.yaml`, to be filled in directly |

Decisions locked in:
- **Sector universe**: self-built constituent baskets, not ETFs/index feeds (see
  `config/assets.yaml -> sectoral`, `src/data_collection/fetch_equity_data.py`).
- **Sentiment source**: self-built news-scraping + FinBERT/LLM scoring pipeline (see
  `config/assets.yaml -> sentiment_source`, `src/data_collection/fetch_sentiment_data.py`).
- **Regime windows**: left as placeholders in `config/config.yaml -> regimes` — fill in
  the actual start/end dates before running the pipeline.

Still open: MCX/NCDEX contract continuity confirmation for the two lower-liquidity
energy legs (Brent Crude, Gasoline — see `config/assets.yaml -> commodities.energy`),
and the FinBERT-vs-LLM choice for sentiment scoring. Sector constituent lists are
filled in and every ticker has been validated against yfinance (see "How to run" below).

## How to run

A project-local virtual environment lives in `.venv/` (Python 3.12). Only the
packages needed for what's implemented so far are installed; see `requirements.txt`
for the full eventual dependency list (forecasting/portfolio libraries not installed yet).

```powershell
# from the project root, E:\Project\sentiment
.venv\Scripts\python.exe -m scripts.run_data_pipeline          # runs all implemented stages (equities only, for now)
.venv\Scripts\python.exe -m scripts.run_data_pipeline --stage equities --start 2020-01-01 --end 2020-06-01   # smaller/faster run
```

This builds the 9 self-built sector baskets (config/assets.yaml -> sectoral) over
the full regime date range from config/config.yaml, and writes one parquet file
per sector to `data/raw/equities/`.

### What's implemented vs. not

| Stage | Status |
|---|---|
| `src/data_collection/fetch_equity_data.py` | **Implemented** — market-cap-weighted sector baskets from yfinance |
| `src/data_collection/fetch_commodity_data.py` | Not implemented (MCX/NCDEX bhavcopy scraping + rollover logic) |
| `src/data_collection/fetch_sentiment_data.py` | Not implemented (news scraping + FinBERT/LLM scoring) |
| `src/data_collection/fetch_macro_data.py` | Not implemented (T-bill rate, India VIX) |
| `src/features/`, `src/regimes/` | Not implemented |
| `src/forecasting/`, `src/portfolio/`, `src/evaluation/` | Not implemented |

Everything else in `scripts/` and `src/` beyond the equity fetcher is still a
docstring-only stub — see each file's TODO list for what it needs.

## Methodology (unchanged from the original paper)

1. **Feature engineering** — 25 variables per asset: OHLCV-derived returns + 11 technical
   indicators (SMA, EMA, RSI, CCI, MFI, ROC, Williams %R, OBV, A/D, MACD, ATR) + 9 sentiment
   indicators (buzz, mentions, emotion-vs-fact, sentiment, optimism, pessimism, volatility,
   uncertainty, fear).
2. **Forecasting** — 9 models: LR, RF, SVR (ML benchmarks) and GRU, LSTM, BiGRU, BiLSTM,
   GRU-LSTM, BiGRU-GRU (DL / hybrid). Hyperparameters tuned with Optuna + time-series CV;
   dropout, early stopping (patience=20), gradient clipping. Best model (GRU in the original
   paper) selected by MAE/RMSE/DA on the out-of-sample window, separately per regime.
3. **Portfolio optimization**:
   - `F-MSR-X`: forecast-driven Maximum Sharpe Ratio portfolios, one per risk measure
     X ∈ {Variance, MSV, MAD, FLPM, CVaR, CDaR, EVaR}.
   - `F-BL`: modified Black-Litterman with forecasted returns as investor views
     (confidence = 99%, τ = 0.05, δ = 1, identity pick matrix).
   - Historical counterparts `MSR-X` (same risk measures, historical mean returns) as
     baselines, plus EWP, HRP, and the market benchmark.
   - Daily rebalancing, top-6 assets by forecast return per configuration, 30% single-asset
     cap, long-only, 5 bps transaction cost, portfolio formation skipped if all 6 candidates
     have negative forecasts.
   - 8 portfolio configurations: Sect, Sect+Ene, Sect+Met, Sect+Agri, Sect+Ene+Met,
     Sect+Ene+Agri, Sect+Met+Agri, Combined.
4. **Evaluation**:
   - Forecast accuracy: MAE, RMSE, Directional Accuracy.
   - Portfolio performance: AMR, ASD, Sharpe Ratio, Sortino Ratio, CVaR(95%), Maximum
     Drawdown — computed separately for the high-volatility and normal regimes.

## Project layout

```
config/            Global + asset + indicator + hyperparameter configuration (YAML)
data/raw/          Untouched source data (equities, commodities, sentiment, macro)
data/interim/       Cleaned & time-aligned data, pre-regime-split
data/processed/     Regime-split, model-ready datasets (train/test per regime)
src/data_collection/  Fetchers for NSE indices, MCX/NCDEX futures, sentiment, macro/risk-free rate
src/features/       Technical + sentiment indicator construction, feature matrix assembly
src/regimes/        Regime definition logic (high-volatility vs normal)
src/forecasting/    ML/DL/hybrid forecasting models, training, evaluation
src/portfolio/      Risk measures, F-MSR-X and F-BL optimizers, benchmarks, rebalancing engine
src/evaluation/     Portfolio performance metrics and regime comparison
src/utils/          IO, logging, plotting helpers
notebooks/          Exploratory analysis and results notebooks
scripts/            End-to-end pipeline entry points
results/            Output tables, figures, saved model checkpoints
tests/              Unit tests for indicators, risk measures, optimizers
docs/               Notes on data sourcing, assumptions, deviations from the original paper
```

## Status

Scaffold only — no data collected or code implemented yet. Next steps:
1. Fill in `config/config.yaml -> regimes` with the actual high-volatility and
   normal regime start/end dates (in-sample + out-of-sample test windows).
2. Fill in `config/assets.yaml -> sectoral[*].constituents` with the top stocks
   per sector (and confirm MCX/NCDEX contract continuity/history length for commodities).
3. Pick FinBERT vs. LLM-based scoring for the sentiment pipeline
   (`config/assets.yaml -> sentiment_source.scoring_model`).
4. Implement `src/data_collection/` fetchers, then `src/features/`, then work through
   `src/forecasting/` and `src/portfolio/` in order.
