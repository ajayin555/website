"""
End-to-end forecasting pipeline entry point:
  1. Load regime-split, model-ready data (data/processed/)
  2. Train + tune all forecasting models per asset per regime (src/forecasting/)
  3. Evaluate and select the best model; save forecasts for the portfolio stage.

Usage: python scripts/run_forecasting_pipeline.py --config config/config.yaml --regime high_volatility
"""
