# Event Contract Forecasting Dashboard

A real-time forecasting dashboard for SPX (S&P 500) and NDX (NASDAQ-100) event contracts with price prediction and volume analysis.

## Features

- **Price Forecasting**: Predicts whether the price will be above or below a certain threshold
- **Multiple Timeframes**: Analyze 5-minute and 10-minute candles separately
- **Volume Analysis**: Visualize buyer vs seller volume to identify market sentiment
- **Technical Indicators**: Uses SMA, RSI, momentum, and volume trends for predictions
- **Interactive Charts**: Real-time price and volume charts using Chart.js
- **Comparison Mode**: Compare different timeframes side-by-side

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the dashboard buttons to:
   - View individual symbol/interval combinations (SPX 5m, SPX 10m, NDX 5m, NDX 10m)
   - Compare timeframes for a single symbol (Compare SPX, Compare NDX)

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/analysis/<symbol>/<interval>` - Get analysis for specific symbol and interval
- `GET /api/compare/<symbol>` - Compare 5m and 10m intervals for a symbol

## How It Works

### Price Prediction
The system uses multiple technical indicators to predict price movement:
- **Moving Average Crossover**: 5-period vs 10-period SMA
- **Momentum**: Recent price change trends
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **Volume Confirmation**: Volume trend analysis

### Volume Analysis
- **Buying Volume**: Volume on candles where close > open (green candles)
- **Selling Volume**: Volume on candles where close < open (red candles)
- **Net Volume**: Difference between buying and selling pressure

## Symbols

- **SPX**: S&P 500 Index (^GSPC)
- **NDX**: NASDAQ-100 Index (^NDX)

## Note

Market data is fetched from Yahoo Finance with a 5-day historical period. The predictions are based on technical analysis and should not be used as sole basis for trading decisions.
