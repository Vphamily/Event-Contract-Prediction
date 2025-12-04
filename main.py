from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Configuration
SYMBOLS = {
    'SPX': '^GSPC',  # S&P 500 Index
    'NDX': '^NDX'    # NASDAQ-100 Index
}

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
        'volatility_reduction': float(volatility_reduction) if volatility_reduction else None
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
    price_distance = ((target_price - current_price) / current_price) * 100
    direction = 'above' if target_price > current_price else 'below'
    
    # Calculate probability based on momentum, volume, and technical indicators
    if prediction_5m is None:
        return jsonify({'error': 'Unable to generate prediction with available data'}), 500
    
    confidence_5m = prediction_5m['confidence'] if prediction_5m['prediction'] == 'ABOVE' and direction == 'above' else 100 - prediction_5m['confidence']
    
    # Use 15m data if available, otherwise use only 5m
    if prediction_10m is not None:
        confidence_10m = prediction_10m['confidence'] if prediction_10m['prediction'] == 'ABOVE' and direction == 'above' else 100 - prediction_10m['confidence']
    else:
        confidence_10m = confidence_5m  # Fallback to 5m confidence
    
    # Weighted average (5m has more weight for near-term)
    combined_confidence = (confidence_5m * 0.6 + confidence_10m * 0.4)
    
    # Adjust for time remaining
    if time_diff < 0:
        prediction_result = 'TIME_PASSED'
        final_confidence = 0
    elif time_diff < 5:
        prediction_result = prediction_5m['prediction']
        final_confidence = confidence_5m
    elif time_diff < 15:
        prediction_result = prediction_5m['prediction']
        final_confidence = combined_confidence
    else:
        prediction_result = 'ABOVE' if combined_confidence > 50 else 'BELOW'
        final_confidence = combined_confidence
    
    return jsonify({
        'symbol': symbol,
        'current_price': current_price,
        'target_price': target_price,
        'target_time': target_time,
        'time_remaining_minutes': max(0, time_diff),
        'price_distance_percent': price_distance,
        'direction': direction,
        'prediction': prediction_result,
        'confidence': final_confidence,
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
        'recommendation': get_recommendation(prediction_result, final_confidence, direction),
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def get_recommendation(prediction, confidence, direction):
    """Generate trading recommendation"""
    if confidence > 70:
        strength = 'Strong'
    elif confidence > 55:
        strength = 'Moderate'
    else:
        strength = 'Weak'
    
    if prediction == 'ABOVE':
        if direction == 'above':
            return f"{strength} signal: Price likely to reach target"
        else:
            return f"{strength} signal: Price moving away from target"
    elif prediction == 'BELOW':
        if direction == 'below':
            return f"{strength} signal: Price likely to reach target"
        else:
            return f"{strength} signal: Price unlikely to reach target"
    else:
        return "Target time has passed"

@app.route('/api/analysis/<symbol>/<interval>')
def get_analysis(symbol, interval):
    """Get analysis for a symbol with specific interval"""
    if symbol not in SYMBOLS:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    if interval not in ['5m', '15m']:
        return jsonify({'error': 'Invalid interval. Use 5m or 15m'}), 400
    
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
    app.run(debug=True, port=5000)
