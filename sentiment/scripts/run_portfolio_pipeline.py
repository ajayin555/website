"""
End-to-end portfolio pipeline entry point:
  1. Load selected-model forecasts (results/models/ or data/processed/)
  2. Run F-MSR-X, F-BL, MSR-X, EWP, HRP, benchmark for all 8 configurations
  3. Compute performance metrics and save results tables (results/tables/)

Usage: python scripts/run_portfolio_pipeline.py --config config/config.yaml --regime normal
"""
