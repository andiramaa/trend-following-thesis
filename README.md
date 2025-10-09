# Trend-Following Strategies with Dynamic Risk Management

**Master's Thesis Implementation**  
Andi Rama - University of Parma, 2024/2025  
Supervisor: Prof. Giulio Tagliavini

## Overview

This repository contains the complete implementation of systematic trend-following strategies with dynamic risk management, as presented in my Master's thesis. The framework tests multiple trend detection methods across diverse asset classes and implements comprehensive risk controls including stop-loss mechanisms and volatility targeting.

## Key Features

- **Asset-Specific Signal Optimization**: Calibrated parameters for cryptocurrency, equity indices, and commodities
- **Dynamic Risk Management**: ATR-based stop-loss and volatility targeting with position scaling
- **Multi-Asset Portfolio Construction**: Equal-weighted and risk-parity aggregation methods
- **Overlay Integration**: Framework for applying trend-following as tactical overlay on traditional portfolios
- **Robustness Testing**: Walk-forward analysis and parameter sensitivity studies

## Repository Structure

├── src/
│   ├── data_processor.py          # Data loading and preprocessing
│   ├── signals_optimizer.py       # Trend signal generation
│   ├── risk_management.py         # Stop-loss and vol targeting
│   ├── portfolio_construction.py  # Multi-asset aggregation
│   ├── overlay_integration.py     # Portfolio overlay framework
│   ├── robustness_analysis.py     # Walk-forward testing
│   ├── diagnostics.py             # Performance diagnostics
│   └── signals.py                 # Base signal module
├── data/
│   ├── raw/                       # Raw market data
│   ├── processed/                 # Cleaned data
│   └── results/                   # Analysis outputs
└── charts/                        # Visualization outputs

## Installation

### Requirements
- Python 3.9+
- pandas, numpy, matplotlib, scipy

### Setup
```bash
git clone https://github.com/andiramaa/trend-following-thesis.git
cd trend-following-thesis
pip install -r requirements.txt
