from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Configuration
SYMBOLS = {
    'SPX': '^GSPC',  # S&P 500 Index
    'NDX': '^NDX'    # NASDAQ-100 Index
}

# ML Model storage
ML_MODELS = {}
ML_SCALERS = {}
MODEL_DIR = 'models'

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def fetch_data(symbol, interval='5m', period='5d'):
    """Fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(interval=interval, period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def fetch_intraday_data(symbol, interval='5m'):
    """Fetch only today's intraday data"""
    try:
        ticker = yf.Ticker(symbol)
        # Try 1d first
        data = ticker.history(interval=interval, period='1d')
        
        # If no data, try 2d as fallback
        if data is None or len(data) == 0:
            print(f"No data for 1d period, trying 2d for {symbol}")
            data = ticker.history(interval=interval, period='2d')
        
        # If still no data, try 5d
        if data is None or len(data) == 0:
            print(f"No data for 2d period, trying 5d for {symbol}")
            data = ticker.history(interval=interval, period='5d')
        
        print(f"Fetched {len(data) if data is not None else 0} rows for {symbol} with interval {interval}")
        return data
    except Exception as e:
        print(f"Error fetching intraday data for {symbol}: {e}")
        return None

def calculate_volume_analysis(data):
    """Analyze buyer vs seller volume"""
    if data is None or len(data) == 0:
        return None
    
    # Calculate buying and selling pressure
    data['Price_Change'] = data['Close'] - data['Open']
    data['Buying_Volume'] = data.apply(
        lambda row: row['Volume'] if row['Price_Change'] > 0 else 0, axis=1
    )
    data['Selling_Volume'] = data.apply(
        lambda row: row['Volume'] if row['Price_Change'] < 0 else 0, axis=1
    )
    
    total_buying = data['Buying_Volume'].sum()
    total_selling = data['Selling_Volume'].sum()
    total_volume = total_buying + total_selling
    
    if total_volume > 0:
        buying_percentage = (total_buying / total_volume) * 100
        selling_percentage = (total_selling / total_volume) * 100
    else:
        buying_percentage = 0
        selling_percentage = 0
    
    return {
        'buying_volume': float(total_buying),
        'selling_volume': float(total_selling),
        'buying_percentage': float(buying_percentage),
        'selling_percentage': float(selling_percentage),
        'net_volume': float(total_buying - total_selling)
    }

def predict_price_movement(data, threshold_price=None):
    """Predict if price will be above or below threshold using volume-based analysis"""
    if data is None or len(data) == 0:
        return None
    
    current_price = float(data['Close'].iloc[-1])
    
    if threshold_price is None:
        threshold_price = current_price
    
    # Calculate buying and selling volume
    data['Price_Change'] = data['Close'] - data['Open']
    data['Buying_Volume'] = data.apply(
        lambda row: row['Volume'] if row['Price_Change'] > 0 else 0, axis=1
    )
    data['Selling_Volume'] = data.apply(
        lambda row: row['Volume'] if row['Price_Change'] < 0 else 0, axis=1
    )
    
    # Get volume percentages
    total_buying = data['Buying_Volume'].sum()
    total_selling = data['Selling_Volume'].sum()
    total_volume = total_buying + total_selling
    
    buying_percentage = (total_buying / total_volume * 100) if total_volume > 0 else 50
    selling_percentage = (total_selling / total_volume * 100) if total_volume > 0 else 50
    
    # Technical indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    # MACD
    macd_line, signal_line, macd_histogram = calculate_macd(data['Close'])
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Histogram'] = macd_histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_upper
    data['BB_Middle'] = bb_middle
    data['BB_Lower'] = bb_lower
    
    # Supertrend
    supertrend_values, supertrend_trend = calculate_supertrend(data, multiplier=3, period=10)
    data['Supertrend'] = supertrend_values
    data['Supertrend_Trend'] = supertrend_trend
    
    # Price momentum
    price_change = data['Close'].pct_change()
    recent_momentum = price_change.tail(5).mean()
    
    # Inside Bar Detection (reversal signal)
    inside_bar_signal, volatility_reduction = detect_inside_bar(data)
    
    # Recent volume trend (last 10 candles)
    recent_buying = data['Buying_Volume'].tail(10).sum()
    recent_selling = data['Selling_Volume'].tail(10).sum()
    recent_total = recent_buying + recent_selling
    recent_buying_pct = (recent_buying / recent_total * 100) if recent_total > 0 else 50
    
    # Prediction logic based primarily on volume
    bullish_signals = 0
    bearish_signals = 0
    
    # Calculate net volume
    net_volume = total_buying - total_selling
    
    # PRIMARY SIGNAL: Net Volume (most important) - 40% weight
    if net_volume > 0:  # Positive net volume = more buyers
        volume_ratio = total_buying / total_selling if total_selling > 0 else 10
        if volume_ratio > 2.0:  # Buyers dominating heavily (2x or more)
            bullish_signals += 4
        elif volume_ratio > 1.5:  # Strong buying
            bullish_signals += 3
        elif volume_ratio > 1.2:  # Moderate buying
            bullish_signals += 2
        else:  # Slight buying edge
            bullish_signals += 1
    else:  # Negative net volume = more sellers
        volume_ratio = total_selling / total_buying if total_buying > 0 else 10
        if volume_ratio > 2.0:  # Sellers dominating heavily
            bearish_signals += 4
        elif volume_ratio > 1.5:  # Strong selling
            bearish_signals += 3
        elif volume_ratio > 1.2:  # Moderate selling
            bearish_signals += 2
        else:  # Slight selling edge
            bearish_signals += 1
    
    # SECONDARY SIGNAL: Overall volume percentage
    if buying_percentage > selling_percentage:
        if buying_percentage > 55:  # Strong buying
            bullish_signals += 2
        else:  # Moderate buying
            bullish_signals += 1
    else:
        if selling_percentage > 55:  # Strong selling
            bearish_signals += 2
        else:  # Moderate selling
            bearish_signals += 1
    
    # MACD SIGNAL - 30% weight (3 signals)
    if not pd.isna(data['MACD'].iloc[-1]) and not pd.isna(data['MACD_Signal'].iloc[-1]):
        macd_current = data['MACD'].iloc[-1]
        macd_signal_current = data['MACD_Signal'].iloc[-1]
        macd_histogram = data['MACD_Histogram'].iloc[-1]
        
        # MACD crossover
        if macd_current > macd_signal_current:  # Bullish crossover
            if macd_histogram > 0:  # Strong bullish
                bullish_signals += 3
            else:  # Moderate bullish
                bullish_signals += 2
        else:  # Bearish crossover
            if macd_histogram < 0:  # Strong bearish
                bearish_signals += 3
            else:  # Moderate bearish
                bearish_signals += 2
    
    # BOLLINGER BANDS SIGNAL - 30% weight (3 signals)
    if not pd.isna(data['BB_Upper'].iloc[-1]) and not pd.isna(data['BB_Lower'].iloc[-1]):
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        
        # Calculate position within bands
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            price_position = (current_price - bb_lower) / bb_range
            
            if price_position > 0.8:  # Near upper band - overbought
                bearish_signals += 3  # Likely to come down
            elif price_position > 0.6:  # Above middle, heading to upper
                bearish_signals += 1
            elif price_position < 0.2:  # Near lower band - oversold
                bullish_signals += 3  # Likely to bounce up
            elif price_position < 0.4:  # Below middle, heading to lower
                bullish_signals += 1
            else:  # In the middle - neutral
                pass  # No signal
    
    # Recent volume trend (last 10 candles)
    if recent_buying_pct > 52:
        bullish_signals += 2
    elif recent_buying_pct < 48:
        bearish_signals += 2
    
    # SMA crossover (secondary signal)
    if len(data) >= 10:
        if data['SMA_5'].iloc[-1] > data['SMA_10'].iloc[-1]:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    # Momentum (secondary signal)
    if recent_momentum > 0.001:  # Significant positive momentum
        bullish_signals += 1
    elif recent_momentum < -0.001:  # Significant negative momentum
        bearish_signals += 1
    
    # RSI (minor signal)
    if not pd.isna(data['RSI'].iloc[-1]):
        if data['RSI'].iloc[-1] < 30:
            bullish_signals += 1
        elif data['RSI'].iloc[-1] > 70:
            bearish_signals += 1
    
    # INSIDE BAR REVERSAL SIGNAL (2 signals for reversal detection)
    if inside_bar_signal == 'BULLISH_REVERSAL':
        bullish_signals += 2  # Strong reversal signal
    elif inside_bar_signal == 'BEARISH_REVERSAL':
        bearish_signals += 2  # Strong reversal signal
    elif inside_bar_signal == 'BULLISH_CONTINUATION':
        bullish_signals += 1  # Trend continuation
    elif inside_bar_signal == 'BEARISH_CONTINUATION':
        bearish_signals += 1  # Trend continuation
    
    # SUPERTREND SIGNAL - 25% weight (3 signals for strong trend following)
    supertrend_signal = None
    if not pd.isna(data['Supertrend_Trend'].iloc[-1]):
        current_trend = data['Supertrend_Trend'].iloc[-1]
        supertrend_value = data['Supertrend'].iloc[-1]
        
        # Check for trend change
        if len(data) >= 2 and not pd.isna(data['Supertrend_Trend'].iloc[-2]):
            prev_trend = data['Supertrend_Trend'].iloc[-2]
            
            # Trend reversal detection
            if current_trend == 1 and prev_trend == -1:
                # Bullish reversal - strong signal
                bullish_signals += 3
                supertrend_signal = 'BULLISH_REVERSAL'
            elif current_trend == -1 and prev_trend == 1:
                # Bearish reversal - strong signal
                bearish_signals += 3
                supertrend_signal = 'BEARISH_REVERSAL'
            elif current_trend == 1:
                # Continuing uptrend
                bullish_signals += 2
                supertrend_signal = 'UPTREND'
            elif current_trend == -1:
                # Continuing downtrend
                bearish_signals += 2
                supertrend_signal = 'DOWNTREND'
        else:
            # No previous trend data, just use current
            if current_trend == 1:
                bullish_signals += 2
                supertrend_signal = 'UPTREND'
            else:
                bearish_signals += 2
                supertrend_signal = 'DOWNTREND'
    
    prediction = 'ABOVE' if bullish_signals > bearish_signals else 'BELOW'
    
    # Calculate confidence based on signal strength
    total_signals = bullish_signals + bearish_signals
    confidence = abs(bullish_signals - bearish_signals) / total_signals * 100 if total_signals > 0 else 50
    
    # Adjust confidence based on volume dominance
    volume_dominance = abs(buying_percentage - selling_percentage)
    if volume_dominance > 20:  # Very clear volume signal
        confidence = min(confidence + 15, 95)
    elif volume_dominance > 10:  # Clear volume signal
        confidence = min(confidence + 10, 90)
    
    # Get MACD values for return
    macd_value = float(data['MACD'].iloc[-1]) if not pd.isna(data['MACD'].iloc[-1]) else None
    macd_signal_value = float(data['MACD_Signal'].iloc[-1]) if not pd.isna(data['MACD_Signal'].iloc[-1]) else None
    
    # Get Bollinger Band position
    bb_position = None
    if not pd.isna(data['BB_Upper'].iloc[-1]) and not pd.isna(data['BB_Lower'].iloc[-1]):
        bb_range = data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]
        if bb_range > 0:
            bb_position = float((current_price - data['BB_Lower'].iloc[-1]) / bb_range * 100)
    
    # Get Supertrend values
    supertrend_value = float(data['Supertrend'].iloc[-1]) if not pd.isna(data['Supertrend'].iloc[-1]) else None
    
    return {
        'current_price': current_price,
        'threshold_price': float(threshold_price),
        'prediction': prediction,
        'confidence': float(confidence),
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'rsi': float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else None,
        'momentum': float(recent_momentum * 100),
        'buying_percentage': float(buying_percentage),
        'selling_percentage': float(selling_percentage),
        'recent_buying_percentage': float(recent_buying_pct),
        'net_volume': float(net_volume),
        'macd': macd_value,
        'macd_signal': macd_signal_value,
        'bb_position': bb_position,
        'inside_bar': inside_bar_signal,
        'volatility_reduction': float(volatility_reduction) if volatility_reduction else None,
        'supertrend': supertrend_value,
        'supertrend_signal': supertrend_signal
    }

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def detect_inside_bar(data):
    """Detect Inside Bar pattern for reversal signals"""
    if len(data) < 2:
        return None, None
    
    # Get last two candles
    current = data.iloc[-1]
    previous = data.iloc[-2]
    
    # Inside Bar: Current candle is completely within previous candle's range
    is_inside_bar = (current['High'] <= previous['High'] and 
                     current['Low'] >= previous['Low'])
    
    if not is_inside_bar:
        return None, None
    
    # Determine direction based on close position and previous trend
    inside_bar_signal = None
    
    # Check if we have enough data for trend
    if len(data) >= 3:
        prev_prev = data.iloc[-3]
        
        # Uptrend with inside bar (potential bearish reversal)
        if previous['Close'] > prev_prev['Close']:
            # If current closes near bottom of inside bar = bearish reversal
            inside_bar_range = current['High'] - current['Low']
            close_position = (current['Close'] - current['Low']) / inside_bar_range if inside_bar_range > 0 else 0.5
            
            if close_position < 0.4:  # Closing near bottom
                inside_bar_signal = 'BEARISH_REVERSAL'
            elif close_position > 0.6:  # Closing near top (continuation)
                inside_bar_signal = 'BULLISH_CONTINUATION'
        
        # Downtrend with inside bar (potential bullish reversal)
        elif previous['Close'] < prev_prev['Close']:
            # If current closes near top of inside bar = bullish reversal
            inside_bar_range = current['High'] - current['Low']
            close_position = (current['Close'] - current['Low']) / inside_bar_range if inside_bar_range > 0 else 0.5
            
            if close_position > 0.6:  # Closing near top
                inside_bar_signal = 'BULLISH_REVERSAL'
            elif close_position < 0.4:  # Closing near bottom (continuation)
                inside_bar_signal = 'BEARISH_CONTINUATION'
    
    # Calculate volatility reduction
    current_range = current['High'] - current['Low']
    previous_range = previous['High'] - previous['Low']
    volatility_reduction = (previous_range - current_range) / previous_range * 100 if previous_range > 0 else 0
    
    return inside_bar_signal, volatility_reduction

def calculate_atr(data, period=10):
    """Calculate Average True Range (ATR)"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_supertrend(data, multiplier=3, period=10):
    """Calculate Supertrend indicator"""
    if len(data) < period:
        return None, None
    
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # Calculate ATR
    df['ATR'] = calculate_atr(df, period)
    
    # Calculate HL2 (average of high and low)
    df['HL2'] = (df['High'] + df['Low']) / 2
    
    # Calculate basic bands
    df['Basic_Upper'] = df['HL2'] + (multiplier * df['ATR'])
    df['Basic_Lower'] = df['HL2'] - (multiplier * df['ATR'])
    
    # Initialize bands and trend using numpy arrays for speed
    upper_band = np.zeros(len(df))
    lower_band = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend = np.ones(len(df))  # 1 = uptrend, -1 = downtrend
    
    # Convert to numpy for faster iteration
    basic_upper = df['Basic_Upper'].values
    basic_lower = df['Basic_Lower'].values
    close_prices = df['Close'].values
    
    for i in range(period, len(df)):
        # Upper Band calculation
        if i == period:
            upper_band[i] = basic_upper[i]
        else:
            if basic_upper[i] < upper_band[i-1] or close_prices[i-1] > upper_band[i-1]:
                upper_band[i] = basic_upper[i]
            else:
                upper_band[i] = upper_band[i-1]
        
        # Lower Band calculation
        if i == period:
            lower_band[i] = basic_lower[i]
        else:
            if basic_lower[i] > lower_band[i-1] or close_prices[i-1] < lower_band[i-1]:
                lower_band[i] = basic_lower[i]
            else:
                lower_band[i] = lower_band[i-1]
        
        # Trend calculation
        if i == period:
            trend[i] = 1  # Start with uptrend
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close_prices[i] > upper_band[i]:
                    trend[i] = 1  # Uptrend
                else:
                    trend[i] = -1  # Downtrend
            else:
                if close_prices[i] < lower_band[i]:
                    trend[i] = -1  # Downtrend
                else:
                    trend[i] = 1  # Uptrend
        
        # Supertrend value
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    df['Supertrend'] = supertrend
    df['Trend'] = trend
    
    return df['Supertrend'], df['Trend']

def extract_ml_features(data, current_price, target_price, time_horizon_minutes):
    """
    Extract comprehensive features for ML model
    Returns a feature vector suitable for classification
    """
    if data is None or len(data) < 30:
        return None
    
    features = {}
    
    # Basic price features
    features['current_price'] = current_price
    features['price_distance_pct'] = ((target_price - current_price) / current_price) * 100
    features['needs_to_go_up'] = 1 if target_price > current_price else 0
    features['time_horizon_minutes'] = time_horizon_minutes
    
    # Price momentum features (multiple timeframes)
    features['momentum_5_candles'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100) if len(data) >= 5 else 0
    features['momentum_10_candles'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100) if len(data) >= 10 else 0
    features['momentum_20_candles'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100) if len(data) >= 20 else 0
    
    # Volume features
    data_copy = data.copy()
    data_copy['Price_Change'] = data_copy['Close'] - data_copy['Open']
    data_copy['Buying_Volume'] = data_copy.apply(lambda row: row['Volume'] if row['Price_Change'] > 0 else 0, axis=1)
    data_copy['Selling_Volume'] = data_copy.apply(lambda row: row['Volume'] if row['Price_Change'] < 0 else 0, axis=1)
    
    total_buying = data_copy['Buying_Volume'].sum()
    total_selling = data_copy['Selling_Volume'].sum()
    total_volume = total_buying + total_selling
    
    features['buying_pressure'] = (total_buying / total_volume * 100) if total_volume > 0 else 50
    features['net_volume_ratio'] = ((total_buying - total_selling) / total_volume) if total_volume > 0 else 0
    
    # Recent volume trend (last 10 candles vs previous 10)
    recent_volume = data_copy['Volume'].tail(10).mean()
    prev_volume = data_copy['Volume'].iloc[-20:-10].mean() if len(data_copy) >= 20 else recent_volume
    features['volume_increase'] = ((recent_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
    
    # Technical indicators
    features['rsi'] = calculate_rsi(data_copy['Close'], 14).iloc[-1] if len(data_copy) >= 14 else 50
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(data_copy['Close'])
    features['macd'] = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    features['macd_signal'] = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    features['macd_histogram'] = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
    features['macd_bullish'] = 1 if features['macd'] > features['macd_signal'] else 0
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data_copy['Close'])
    if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]):
        bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        features['bb_position'] = ((current_price - bb_lower.iloc[-1]) / bb_range) if bb_range > 0 else 0.5
        features['bb_width'] = (bb_range / bb_middle.iloc[-1]) if bb_middle.iloc[-1] > 0 else 0
    else:
        features['bb_position'] = 0.5
        features['bb_width'] = 0
    
    # Supertrend
    supertrend_values, supertrend_trend = calculate_supertrend(data_copy)
    if supertrend_trend is not None and not pd.isna(supertrend_trend.iloc[-1]):
        features['supertrend_bullish'] = 1 if supertrend_trend.iloc[-1] == 1 else 0
        features['distance_to_supertrend'] = ((current_price - supertrend_values.iloc[-1]) / current_price * 100) if not pd.isna(supertrend_values.iloc[-1]) else 0
    else:
        features['supertrend_bullish'] = 0
        features['distance_to_supertrend'] = 0
    
    # Volatility features
    returns = data_copy['Close'].pct_change()
    features['volatility'] = returns.std() * 100 if len(returns) > 1 else 0
    features['avg_candle_size'] = ((data_copy['High'] - data_copy['Low']) / data_copy['Close'] * 100).mean()
    
    # Price position features
    high_20 = data_copy['High'].tail(20).max()
    low_20 = data_copy['Low'].tail(20).min()
    features['price_position_20'] = ((current_price - low_20) / (high_20 - low_20)) if (high_20 - low_20) > 0 else 0.5
    
    # Trend strength
    data_copy['SMA_5'] = data_copy['Close'].rolling(window=5).mean()
    data_copy['SMA_10'] = data_copy['Close'].rolling(window=10).mean()
    data_copy['SMA_20'] = data_copy['Close'].rolling(window=20).mean()
    
    features['above_sma5'] = 1 if current_price > data_copy['SMA_5'].iloc[-1] else 0
    features['above_sma10'] = 1 if len(data_copy) >= 10 and current_price > data_copy['SMA_10'].iloc[-1] else 0
    features['above_sma20'] = 1 if len(data_copy) >= 20 and current_price > data_copy['SMA_20'].iloc[-1] else 0
    
    # Convert to array in consistent order
    feature_names = sorted(features.keys())
    feature_vector = [features[name] for name in feature_names]
    
    return feature_vector, feature_names

def train_ml_model(symbol, interval='5m', lookback_days=30):
    """
    Train ML model using historical data
    Creates labeled training data by looking at whether price reached targets
    """
    print(f"Training ML model for {symbol} with {interval} interval...")
    
    # Fetch extended historical data
    ticker = yf.Ticker(SYMBOLS[symbol])
    historical_data = ticker.history(interval=interval, period=f'{lookback_days}d')
    
    if historical_data is None or len(historical_data) < 100:
        print(f"Insufficient data for training: {len(historical_data) if historical_data is not None else 0} rows")
        return None, None
    
    X = []  # Features
    y = []  # Labels (1 = reached target, 0 = didn't reach)
    feature_names = None
    
    # Create training samples
    for i in range(50, len(historical_data) - 30):  # Leave room for lookback and lookahead
        current_price = historical_data['Close'].iloc[i]
        
        # Simulate different target prices and time horizons
        for price_move_pct in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:  # Different target distances
            for time_horizon in [5, 10, 15, 30, 60]:  # Different time horizons in minutes
                if interval == '5m':
                    candles_ahead = time_horizon // 5
                elif interval == '15m':
                    candles_ahead = time_horizon // 15
                elif interval == '30m':
                    candles_ahead = time_horizon // 30
                else:
                    candles_ahead = 1
                
                if i + candles_ahead >= len(historical_data):
                    continue
                
                # Create targets in both directions
                for direction in [1, -1]:  # 1 = up, -1 = down
                    target_price = current_price * (1 + direction * price_move_pct / 100)
                    
                    # Extract features at time i
                    lookback_data = historical_data.iloc[max(0, i-50):i+1]
                    features, fn = extract_ml_features(lookback_data, current_price, target_price, time_horizon)
                    
                    if features is None:
                        continue
                    
                    if feature_names is None:
                        feature_names = fn
                    
                    # Check if target was reached in the time horizon
                    future_data = historical_data.iloc[i:i+candles_ahead+1]
                    if direction == 1:
                        target_reached = future_data['High'].max() >= target_price
                    else:
                        target_reached = future_data['Low'].min() <= target_price
                    
                    X.append(features)
                    y.append(1 if target_reached else 0)
    
    if len(X) < 100:
        print(f"Insufficient training samples: {len(X)}")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training with {len(X)} samples, {sum(y)} positive cases ({sum(y)/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting model (better for imbalanced data)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Model trained! Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f'{symbol}_{interval}_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'{symbol}_{interval}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, f'{symbol}_{interval}_features.pkl'))
    
    return model, scaler

def load_ml_model(symbol, interval='5m'):
    """Load trained ML model or train a new one"""
    model_key = f'{symbol}_{interval}'
    
    # Check if model is already loaded
    if model_key in ML_MODELS:
        return ML_MODELS[model_key], ML_SCALERS[model_key]
    
    # Try to load from disk
    model_path = os.path.join(MODEL_DIR, f'{symbol}_{interval}_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'{symbol}_{interval}_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            ML_MODELS[model_key] = model
            ML_SCALERS[model_key] = scaler
            print(f"Loaded existing model for {model_key}")
            return model, scaler
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Train new model
    print(f"No existing model found for {model_key}, training new model...")
    model, scaler = train_ml_model(symbol, interval)
    
    if model is not None:
        ML_MODELS[model_key] = model
        ML_SCALERS[model_key] = scaler
    
    return model, scaler

def predict_with_ml(data, current_price, target_price, time_horizon_minutes, symbol, interval='5m'):
    """
    Use ML model to predict probability of reaching target
    """
    # Extract features
    features, feature_names = extract_ml_features(data, current_price, target_price, time_horizon_minutes)
    
    if features is None:
        return None
    
    # Load model
    model, scaler = load_ml_model(symbol, interval)
    
    if model is None or scaler is None:
        print(f"ML model not available for {symbol} {interval}")
        return None
    
    # Scale features and predict
    features_scaled = scaler.transform([features])
    
    # Get probability
    probability = model.predict_proba(features_scaled)[0][1]  # Probability of reaching target
    
    # Also get feature importance for debugging
    prediction = model.predict(features_scaled)[0]
    
    return {
        'probability': float(probability * 100),  # Convert to percentage
        'will_reach': bool(prediction),
        'confidence': float(probability * 100) if prediction else float((1 - probability) * 100)
    }

def prepare_chart_data(data, symbol_name):
    """Prepare data for charting"""
    if data is None or len(data) == 0:
        return None
    
    chart_data = {
        'labels': [dt.strftime('%I:%M %p') for dt in data.index],  # 12-hour format with AM/PM
        'prices': data['Close'].tolist(),
        'volumes': data['Volume'].tolist(),
        'buying_volumes': data['Buying_Volume'].tolist() if 'Buying_Volume' in data.columns else [],
        'selling_volumes': data['Selling_Volume'].tolist() if 'Selling_Volume' in data.columns else []
    }
    return chart_data

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_price_target():
    """Predict if price will reach target by specific time"""
    from flask import request
    
    data = request.get_json()
    symbol = data.get('symbol')
    target_price = float(data.get('target_price'))
    target_time = data.get('target_time')  # Format: 'HH:MM' like '15:00' for 3 PM
    interval = data.get('interval', '5m')
    
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    # Fetch only today's intraday data
    data_5m = fetch_intraday_data(SYMBOLS[symbol], interval='5m')
    data_10m = fetch_intraday_data(SYMBOLS[symbol], interval='15m')
    
    if data_5m is None or len(data_5m) == 0:
        return jsonify({'error': 'Failed to fetch data or no data available for today'}), 500
    
    current_price = float(data_5m['Close'].iloc[-1])
    current_time = data_5m.index[-1]
    
    # Parse target time
    from datetime import datetime, time
    target_hour, target_minute = map(int, target_time.split(':'))
    
    # Create target datetime (today at target time)
    target_datetime = datetime.now().replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # Calculate time difference
    time_diff = (target_datetime - datetime.now()).total_seconds() / 60  # minutes
    
    # Analyze both intervals
    prediction_5m = predict_price_movement(data_5m, target_price)
    volume_5m = calculate_volume_analysis(data_5m)
    
    # Check if we have valid 15m data
    prediction_10m = None
    volume_10m = None
    if data_10m is not None and len(data_10m) > 0:
        prediction_10m = predict_price_movement(data_10m, target_price)
        volume_10m = calculate_volume_analysis(data_10m)
    
    # Enhanced prediction combining both timeframes
    price_distance = abs((target_price - current_price) / current_price) * 100
    needs_to_go_up = target_price > current_price  # True if price needs to rise to hit target
    
    # Calculate probability based on momentum, volume, and technical indicators
    if prediction_5m is None:
        return jsonify({'error': 'Unable to generate prediction with available data'}), 500
    
    # Check if prediction aligns with needed direction
    prediction_5m_aligned = (prediction_5m['prediction'] == 'ABOVE' and needs_to_go_up) or \
                           (prediction_5m['prediction'] == 'BELOW' and not needs_to_go_up)
    
    # If prediction aligns with needed direction, use confidence as-is; otherwise invert it
    confidence_5m = prediction_5m['confidence'] if prediction_5m_aligned else (100 - prediction_5m['confidence'])
    
    # Use 15m data if available, otherwise use only 5m
    if prediction_10m is not None:
        prediction_10m_aligned = (prediction_10m['prediction'] == 'ABOVE' and needs_to_go_up) or \
                                (prediction_10m['prediction'] == 'BELOW' and not needs_to_go_up)
        confidence_10m = prediction_10m['confidence'] if prediction_10m_aligned else (100 - prediction_10m['confidence'])
    else:
        confidence_10m = confidence_5m  # Fallback to 5m confidence
    
    # Fetch 30m data for longer-term view
    data_30m = fetch_intraday_data(SYMBOLS[symbol], interval='30m')
    confidence_30m = confidence_5m  # Default fallback
    
    if data_30m is not None and len(data_30m) > 0:
        prediction_30m = predict_price_movement(data_30m, target_price)
        if prediction_30m is not None:
            prediction_30m_aligned = (prediction_30m['prediction'] == 'ABOVE' and needs_to_go_up) or \
                                    (prediction_30m['prediction'] == 'BELOW' and not needs_to_go_up)
            confidence_30m = prediction_30m['confidence'] if prediction_30m_aligned else (100 - prediction_30m['confidence'])
    
    # Weighted average based on time remaining (adjust weights dynamically)
    if time_diff < 10:  # Less than 10 minutes
        combined_confidence = confidence_5m  # Only use 5m
    elif time_diff < 30:  # 10-30 minutes
        combined_confidence = (confidence_5m * 0.7 + confidence_10m * 0.3)
    else:  # 30+ minutes
        combined_confidence = (confidence_5m * 0.4 + confidence_10m * 0.35 + confidence_30m * 0.25)
    
    # Adjust confidence based on price distance (larger moves are harder)
    if price_distance > 2.0:  # More than 2% move required
        combined_confidence *= 0.7  # Reduce confidence significantly
    elif price_distance > 1.0:  # 1-2% move
        combined_confidence *= 0.85
    elif price_distance > 0.5:  # 0.5-1% move
        combined_confidence *= 0.95
    
    # === MACHINE LEARNING PREDICTION ===
    # Use ML model for final prediction (if available)
    ml_prediction_5m = predict_with_ml(data_5m, current_price, target_price, time_diff, symbol, '5m')
    ml_prediction_15m = None
    ml_prediction_30m = None
    
    if data_10m is not None and len(data_10m) > 0:
        ml_prediction_15m = predict_with_ml(data_10m, current_price, target_price, time_diff, symbol, '15m')
    
    if data_30m is not None and len(data_30m) > 0:
        ml_prediction_30m = predict_with_ml(data_30m, current_price, target_price, time_diff, symbol, '30m')
    
    # Combine ML predictions with rule-based confidence
    if ml_prediction_5m is not None:
        ml_confidence = ml_prediction_5m['probability']
        
        # Weight ML predictions based on time horizon
        if ml_prediction_15m is not None and time_diff >= 15:
            ml_confidence = ml_confidence * 0.6 + ml_prediction_15m['probability'] * 0.4
        
        if ml_prediction_30m is not None and time_diff >= 30:
            ml_confidence = ml_confidence * 0.5 + ml_prediction_15m['probability'] * 0.3 + ml_prediction_30m['probability'] * 0.2
        
        # Blend ML confidence (70%) with rule-based confidence (30%)
        final_ml_confidence = ml_confidence * 0.7 + combined_confidence * 0.3
        
        print(f"ML Prediction: {ml_confidence:.1f}%, Rule-based: {combined_confidence:.1f}%, Blended: {final_ml_confidence:.1f}%")
    else:
        # Fallback to rule-based if ML not available
        final_ml_confidence = combined_confidence
        print(f"ML model not available, using rule-based confidence: {combined_confidence:.1f}%")
    
    # Adjust for time remaining
    if time_diff < 0:
        prediction_result = 'TIME_PASSED'
        final_confidence = 0
        will_reach = False
    else:
        # Determine if target will be reached using ML-enhanced confidence
        will_reach = final_ml_confidence >= 50
        final_confidence = final_ml_confidence if will_reach else (100 - final_ml_confidence)
        
        # Set result based on whether target will be reached
        if needs_to_go_up:
            prediction_result = 'ABOVE' if will_reach else 'BELOW'
        else:
            prediction_result = 'BELOW' if will_reach else 'ABOVE'
    
    return jsonify({
        'symbol': symbol,
        'current_price': current_price,
        'target_price': target_price,
        'target_time': target_time,
        'time_remaining_minutes': max(0, time_diff),
        'price_distance_percent': price_distance,
        'needs_to_go_up': needs_to_go_up,
        'will_reach_target': will_reach if time_diff >= 0 else False,
        'prediction': prediction_result,
        'confidence': final_confidence,
        'ml_enabled': ml_prediction_5m is not None,
        'ml_confidence': ml_confidence if ml_prediction_5m is not None else None,
        'rule_confidence': combined_confidence,
        'analysis_5m': {
            'prediction': prediction_5m['prediction'],
            'confidence': prediction_5m['confidence'],
            'rsi': prediction_5m['rsi'],
            'momentum': prediction_5m['momentum'],
            'volume_analysis': volume_5m
        },
        'analysis_10m': {
            'prediction': prediction_10m['prediction'] if prediction_10m else 'N/A',
            'confidence': prediction_10m['confidence'] if prediction_10m else 0,
            'rsi': prediction_10m['rsi'] if prediction_10m else None,
            'momentum': prediction_10m['momentum'] if prediction_10m else 0,
            'volume_analysis': volume_10m if volume_10m else {'buying_volume': 0, 'selling_volume': 0, 'buying_percentage': 0, 'selling_percentage': 0, 'net_volume': 0}
        },
        'recommendation': get_recommendation(will_reach if time_diff >= 0 else False, final_confidence, needs_to_go_up, price_distance),
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def get_recommendation(will_reach_target, confidence, needs_to_go_up, price_distance):
    """Generate trading recommendation"""
    if confidence > 75:
        strength = 'HIGHLY LIKELY'
    elif confidence > 60:
        strength = 'LIKELY'
    elif confidence > 45:
        strength = 'POSSIBLE'
    else:
        strength = 'UNLIKELY'
    
    direction_text = 'rise' if needs_to_go_up else 'fall'
    
    if will_reach_target:
        if price_distance < 0.3:
            return f"{strength}: Target very close ({price_distance:.2f}% move needed)"
        elif price_distance < 1.0:
            return f"{strength}: Price should {direction_text} to reach target ({price_distance:.2f}% move)"
        else:
            return f"{strength}: Large move required ({price_distance:.2f}% {direction_text})"
    else:
        return f"{strength}: Target will NOT be reached (requires {price_distance:.2f}% {direction_text})"

@app.route('/api/analysis/<symbol>/<interval>')
def get_analysis(symbol, interval):
    """Get analysis for a symbol with specific interval"""
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    if interval not in ['5m', '15m', '30m']:
        return jsonify({'error': 'Invalid interval. Use 5m, 15m, or 30m'}), 400
    
    # Fetch only today's intraday data
    data = fetch_intraday_data(SYMBOLS[symbol], interval=interval)
    
    if data is None or len(data) == 0:
        return jsonify({'error': 'Failed to fetch data or no data available for today'}), 500
    
    # Calculate volume analysis
    volume_analysis = calculate_volume_analysis(data)
    
    # Predict price movement
    current_price = float(data['Close'].iloc[-1])
    prediction = predict_price_movement(data, current_price)
    
    # Prepare chart data
    chart_data = prepare_chart_data(data, symbol)
    
    return jsonify({
        'symbol': symbol,
        'interval': interval,
        'volume_analysis': volume_analysis,
        'prediction': prediction,
        'chart_data': chart_data,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/train/<symbol>/<interval>')
def train_model_endpoint(symbol, interval):
    """Endpoint to trigger model training"""
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    if interval not in ['5m', '15m', '30m']:
        return jsonify({'error': 'Invalid interval'}), 400
    
    try:
        model, scaler = train_ml_model(symbol, interval, lookback_days=30)
        
        if model is not None:
            return jsonify({
                'status': 'success',
                'message': f'Model trained successfully for {symbol} {interval}',
                'model_saved': True
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to train model - insufficient data'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/model/status')
def model_status():
    """Check which models are available"""
    models_available = {}
    
    for symbol in SYMBOLS.keys():
        models_available[symbol] = {}
        for interval in ['5m', '15m', '30m']:
            model_path = os.path.join(MODEL_DIR, f'{symbol}_{interval}_model.pkl')
            models_available[symbol][interval] = os.path.exists(model_path)
    
    return jsonify({
        'models': models_available,
        'loaded_models': list(ML_MODELS.keys())
    })

@app.route('/api/compare/<symbol>')
def compare_intervals(symbol):
    """Compare 5m and 10m intervals for a symbol"""
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    # Fetch only today's intraday data
    data_5m = fetch_intraday_data(SYMBOLS[symbol], interval='5m')
    data_10m = fetch_intraday_data(SYMBOLS[symbol], interval='15m')
    
    results = {}
    
    for interval, data in [('5m', data_5m), ('15m', data_10m)]:
        if data is not None and len(data) > 0:
            volume_analysis = calculate_volume_analysis(data)
            current_price = float(data['Close'].iloc[-1])
            prediction = predict_price_movement(data, current_price)
            chart_data = prepare_chart_data(data, symbol)
            
            results[interval] = {
                'volume_analysis': volume_analysis,
                'prediction': prediction,
                'chart_data': chart_data
            }
    
    return jsonify({
        'symbol': symbol,
        'intervals': results,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 Event Contract Prediction with Machine Learning")
    print("=" * 60)
    print("\n📊 Training ML models on startup (this may take a minute)...\n")
    
    # Pre-train models for all symbols and intervals on startup
    for symbol in SYMBOLS.keys():
        for interval in ['5m', '15m', '30m']:
            try:
                print(f"Training {symbol} {interval}...")
                train_ml_model(symbol, interval, lookback_days=30)
            except Exception as e:
                print(f"Warning: Could not train {symbol} {interval}: {e}")
    
    print("\n✅ ML models ready!")
    print("🌐 Starting Flask server on http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000)
