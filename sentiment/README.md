# Sentiment-Informed Forecast-Driven Portfolio Optimization — Indian Market

A full replication of Manish & Chahal (2026), *"Sentiment-informed forecast-driven
portfolio optimization with alternative risk measures and strategic commodity
diversification"* (Research in International Business and Finance, 91, 103620),
adapted end-to-end for the Indian equity and commodity markets.

---

## What This Project Does

This project builds an **automated investment research pipeline** that answers one
question: *can combining market sentiment with machine-learning return forecasts
produce better portfolios than traditional mean-variance optimization?*

It does this in four stages:

1. **Collects raw data** — price history for 9 Indian equity sectors, 10 commodity
   futures, macroeconomic indicators (India VIX, Nifty 50, risk-free rate), and
   daily sentiment scores derived from financial news.

2. **Engineers features** — computes 11 technical indicators (RSI, MACD, ATR, etc.)
   and 9 sentiment indicators (buzz, optimism, fear, uncertainty, etc.) per asset,
   building a 21-variable feature matrix used as model input.

3. **Forecasts next-day returns** — trains and tunes 9 models ranging from linear
   regression to deep recurrent neural networks (GRU, LSTM, BiGRU, BiLSTM and
   hybrid stacks), selecting the best per asset per market regime.

4. **Optimizes portfolios** — uses those forecasts as forward-looking inputs to
   construct portfolios under 7 different risk measures (Variance, CVaR, CDaR,
   EVaR, MAD, FLPM, Mean Semi-Variance), plus a Black-Litterman model, and
   benchmarks them against equal-weight, HRP, and the Nifty 50.

---

## Why We Are Doing This

### The Problem with Traditional Portfolio Construction

Classic mean-variance optimization (Markowitz, 1952) uses historical average
returns as the expected return estimate. Historical averages are noisy, backward-
looking, and completely blind to what the market is thinking right now. The result:
portfolios that optimise for the past, not the future.

### What the Research Shows

The original Manish & Chahal (2026) paper demonstrated on US markets that:

- **Sentiment-enhanced forecasts beat pure price-based forecasts** in directional
  accuracy, particularly during high-volatility regimes.
- **Forecast-driven portfolios (F-MSR, F-BL) beat their historical counterparts**
  on Sharpe ratio, Sortino ratio, and maximum drawdown.
- **Commodity diversification adds value** — adding even one commodity class
  (energy, metals, or agriculture) to an equity-only portfolio improved
  risk-adjusted returns in both regimes.
- **CVaR and CDaR as risk measures** produced the most resilient portfolios during
  the high-volatility window, outperforming classical variance minimisation.

### Why India Specifically

The US study used licensed data sources (LSEG MarketPsych for sentiment, Bloomberg
for prices) unavailable at reasonable cost for individual researchers. India offers
a compelling replication target:

- **NSE-listed equities** are freely accessible via yfinance.
- **MCX/NCDEX commodity futures** cover a rich universe of energy, metals, and
  agriculture contracts with long price histories.
- **Indian financial media** (Economic Times, Moneycontrol, Business Standard, Mint)
  is high-volume and English-language, making FinBERT-based sentiment scoring
  directly applicable.
- The Indian market has distinct high-volatility episodes (2016 demonetisation,
  2018 IL&FS crisis, 2020 COVID crash) that make regime-conditional analysis
  particularly meaningful.

---

## How It Will Be Useful

### Academic / Research Use

The pipeline produces a full set of results tables and figures that mirror the
original paper's output (Tables 7–11, B.5–B.6 in Manish & Chahal 2026):

- Forecast accuracy per model per asset per regime (MAE, RMSE, Directional Accuracy)
- Portfolio performance per configuration per regime (Annualised Mean Return,
  Annualised Std Dev, Sharpe, Sortino, CVaR 95%, Maximum Drawdown)
- Regime comparison: high-volatility window vs. normal window

This directly supports the conclusion that sentiment-informed, forecast-driven
portfolios are superior to traditional approaches — extended now to the Indian market.

### Practical / Investor Use

Once the pipeline is running on live data (replacing synthetic sentiment with real
scraped headlines), it can be used as a **daily decision-support tool**:

1. Run `scripts/run_data_pipeline.py --live-scrape` each morning to pull the latest
   prices and scrape overnight news sentiment.
2. Run `scripts/run_forecasting_pipeline.py` to generate next-day return forecasts.
3. Run `scripts/run_portfolio_pipeline.py` to get the day's recommended weights
   across the 8 asset configurations.
4. The rebalancing engine accounts for 5 bps transaction costs and will skip
   portfolio formation if all top-6 candidates have negative forecasts — a
   built-in stay-in-cash signal.

### What It Is Not

This pipeline is a **research and backtesting tool**, not a production trading
system. It does not connect to any broker API or execute orders. The sentiment
data for the historical backtest uses a synthetic proxy (noise correlated with
price movements) — real alpha from sentiment requires a live news feed.

---

## Asset Universe

### Equity Sectors (9) — Self-Built Constituent Baskets

Rather than using licensed index feeds or thinly-traded sector ETFs, each sector
"asset" is constructed as a market-cap-weighted basket of its NSE-listed constituent
stocks, priced from raw OHLCV data. This mirrors the original paper's methodology
while keeping the pipeline fully open-source.

| Sector | Index | Constituents |
|---|---|---|
| Bank | Nifty Bank | HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK + 9 more |
| IT | Nifty IT | INFY, TCS, HCLTECH, TECHM, COFORGE + 5 more |
| FMCG | Nifty FMCG | ITC, HINDUNILVR, NESTLEIND, TATACONSUM + 11 more |
| Pharma | Nifty Pharma | SUNPHARMA, DIVISLAB, TORNTPHARM + 17 more |
| Auto | Nifty Auto | M&M, MARUTI, BAJAJ-AUTO, EICHERMOT + 11 more |
| Energy | Nifty Energy | RELIANCE, ADANIPOWER, NTPC, ONGC + 4 more |
| Metal | Nifty Metal | ADANIENT, JSWSTEEL, HINDZINC + 9 more |
| Realty | Nifty Realty | DLF, LODHA, PRESTIGE, PHOENIXLTD + 5 more |
| Financial Services | Nifty FinServ | HDFCBANK, BAJFINANCE, BAJAJFINSV + 17 more |

### Commodity Futures (10) — yfinance Proxies

| Class | Commodity | Proxy Ticker |
|---|---|---|
| Energy | Crude Oil | CL=F |
| Energy | Natural Gas | NG=F |
| Energy | Brent Crude | BZ=F |
| Energy | Gasoline | RB=F |
| Metals | Gold | GC=F |
| Metals | Silver | SI=F |
| Metals | Copper | HG=F |
| Metals | Zinc | ZNC=F |
| Agriculture | Soybean | ZS=F |
| Agriculture | Cotton | CT=F |

> **Note**: These are USD-denominated international proxies. For a production
> deployment replace with MCX/NCDEX bhavcopy continuous-contract series
> (see `config/assets.yaml → commodities` for sourcing notes).

### Macro / Benchmark

| Series | Source | Use |
|---|---|---|
| India VIX | `^INDIAVIX` via yfinance | Regime classification |
| Nifty 50 | `^NSEI` via yfinance | Portfolio benchmark |
| Risk-Free Rate | `^IRX` (13-week US T-bill proxy) | Sharpe/Sortino denominator |

---

## Methodology

### 1. Market Regimes

Following the original paper, the full dataset is split into two non-overlapping
regimes based on realised volatility (India VIX):

| Regime | Full Sample | Out-of-Sample Test |
|---|---|---|
| High-Volatility | Jul 2014 – Dec 2022 | Jan 2020 – Dec 2022 |
| Normal | Jul 2017 – Dec 2025 | Jan 2023 – Dec 2025 |

All models are trained and evaluated separately per regime. This design tests
whether the sentiment-forecast edge is consistent across market conditions or
concentrated in turbulent periods.

### 2. Feature Engineering (21 variables per asset)

**Technical indicators (11):**
SMA-14, EMA-14, RSI, CCI, MFI, ROC, Williams %R, OBV, A/D Line, MACD, ATR

**Sentiment indicators (9):**
Buzz, Mentions, Emotion-vs-Fact, Sentiment, Optimism, Pessimism,
Volatility Language, Uncertainty, Fear

**Target:** Next-day log return (1-step-ahead prediction)

### 3. Forecasting Models (9)

| Category | Models |
|---|---|
| ML Benchmarks | Linear Regression (LR), Random Forest (RF), Support Vector Regression (SVR) |
| Deep Learning | GRU, LSTM, Bidirectional GRU (BiGRU), Bidirectional LSTM (BiLSTM) |
| Hybrid | GRU→LSTM stack, BiGRU→GRU stack |

All DL models use dropout regularisation, early stopping (patience = 20 epochs),
and gradient clipping. Hyperparameters are tuned with **Optuna** using time-series
cross-validation. The best model per asset per regime is selected by lowest MAE.

### 4. Portfolio Configurations (8)

| Configuration | Assets Included |
|---|---|
| Sect | 9 equity sectors only |
| Sect+Ene | Sectors + 4 energy futures |
| Sect+Met | Sectors + 4 metal futures |
| Sect+Agri | Sectors + 2 agriculture futures |
| Sect+Ene+Met | Sectors + energy + metals |
| Sect+Ene+Agri | Sectors + energy + agriculture |
| Sect+Met+Agri | Sectors + metals + agriculture |
| Combined | All 9 sectors + all 10 commodities |

### 5. Portfolio Strategies

**Forecast-driven (the main contribution):**
- `F-MSR-Var` / `F-MSR-MSV` / `F-MSR-MAD` / `F-MSR-FLPM` / `F-MSR-CVaR` / `F-MSR-CDaR` / `F-MSR-EVaR` — Maximum Sharpe Ratio with forecasted expected returns, each under a different risk measure
- `F-BL` — Black-Litterman model using forecasted returns as investor views (confidence = 99%)

**Historical baselines (for comparison):**
- `MSR-X` — same 7 risk measure variants but using historical mean returns
- `EWP` — equal-weight portfolio
- `HRP` — Hierarchical Risk Parity

**Rebalancing rules:**
- Daily rebalancing
- Top-6 assets by forecasted return selected per day
- 30% maximum weight per asset
- Long-only
- 5 bps transaction cost
- Portfolio formation suspended if all 6 top candidates have negative forecasts (stay in cash)

### 6. Performance Metrics

| Metric | Abbreviation |
|---|---|
| Annualised Mean Return | AMR |
| Annualised Standard Deviation | ASD |
| Sharpe Ratio | SR |
| Sortino Ratio | SorR |
| Conditional Value at Risk (95%) | CVaR |
| Maximum Drawdown | MDD |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Fetch All Data

```bash
python -m scripts.run_data_pipeline
```

Downloads equity sector baskets, commodity futures, macro series, and generates
sentiment data. Completes in ~30 seconds.

### Step 2 — Train Forecasting Models

```bash
# High-volatility regime
python -m scripts.run_forecasting_pipeline --regime high_volatility --trials 5

# Normal regime
python -m scripts.run_forecasting_pipeline --regime normal --trials 5
```

`--trials` sets the number of Optuna hyperparameter search iterations per model.
Use 5–10 for a quick run, 50 (default) for full tuning. ML-only run:

```bash
python -m scripts.run_forecasting_pipeline --regime high_volatility --models LR,RF,SVR
```

### Step 3 — Run Portfolio Optimisation

```bash
python -m scripts.run_portfolio_pipeline --regime high_volatility
python -m scripts.run_portfolio_pipeline --regime normal
```

### Or All at Once

```bash
python -m scripts.run_full_pipeline
```

### Output

| Path | Contents |
|---|---|
| `results/tables/` | CSV performance tables per regime |
| `results/figures/` | Cumulative return plots per regime |
| `results/models/` | Saved model checkpoints and forecast series |

---

## Project Layout

```
config/                     YAML configuration (assets, indicators, hyperparameters, regimes)
data/raw/equities/          Market-cap-weighted sector basket parquet files
data/raw/commodities/       Commodity futures parquet files
data/raw/macro/             India VIX, Nifty 50, risk-free rate
data/raw/sentiment/         Daily 9-indicator sentiment series per sector
data/processed/             Regime-split, model-ready train/test datasets
src/data_collection/        Data fetchers (equity, commodity, macro, sentiment)
src/features/               Technical + sentiment indicator construction, feature matrix
src/regimes/                Regime split logic and processed data writer
src/forecasting/            Model definitions, Optuna trainer, evaluator
src/portfolio/              Risk measures, F-MSR-X / F-BL optimisers, benchmarks, rebalancer
src/evaluation/             Performance metrics, comparison tables, plots
src/utils/                  Logging, YAML/parquet I/O helpers
scripts/                    Pipeline entry points (data → forecast → portfolio → full)
results/                    Generated output (tables, figures, model checkpoints)
notebooks/                  Planned: exploratory analysis and results notebooks
docs/                       Notes on data sourcing and deviations from the original paper
```

---

## Limitations and Known Gaps

| Item | Current State | Production Fix |
|---|---|---|
| Sentiment data | Synthetic (random noise) for historical backtest | Replace with live ET/Moneycontrol RSS + FinBERT, or a licensed feed |
| Commodity data | USD-denominated international futures (yfinance proxies) | MCX/NCDEX bhavcopy continuous-contract series |
| Risk-free rate | US T-bill proxy (`^IRX`) | RBI 91-day T-bill auction data |
| Sector constituents | Current membership (Sep 2026) | Point-in-time index membership to avoid survivorship bias |
| Commodity universe | 10 of 12 (Mustard Seed and Guar Seed have no yfinance proxy) | NCDEX bhavcopy for RMSEED and GUARSEED |

---

## Reference

Manish, K., & Chahal, R. (2026). Sentiment-informed forecast-driven portfolio
optimization with alternative risk measures and strategic commodity diversification.
*Research in International Business and Finance*, 91, 103620.
