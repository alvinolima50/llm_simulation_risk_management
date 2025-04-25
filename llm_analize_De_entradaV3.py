#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Natural Gas Trading Dashboard with LLM Integration

A simulation of a natural gas futures trading system that uses LLM (Large Language Model)
analytics to make trading decisions based on price action, support/resistance levels, 
and volume patterns. The dashboard visualizes price movements, trading decisions, and
LLM reasoning in real-time.

Author: Alvino
Version: 3.1.0
License: MIT
"""

# Standard library imports
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# API client setup
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

class Config:
    """Configuration settings for the application."""
    
    # LLM settings
    LLM_EVERY = 1  # Run LLM analysis on every n-th candle
    
    # Trading settings
    SUPPORT_RESISTANCE_LEVELS = [4.0, 3.9, 3.85, 3.8]
    CONTRACT_MULTIPLIER = 10_000  # Each point (1.00 USD/MMBtu) = US$10,000 per contract
    INITIAL_CONTRACTS = 5
    
    # Data settings
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "agents_project01", "operacao_agent.csv")
    
    # OpenAI API configuration
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_TIMEOUT = 5.0  # 5 second timeout
    OPENAI_MAX_RETRIES = 2  # Limit retries to avoid hanging
    
    # UI settings
    THEME = dbc.themes.DARKLY


# -----------------------------------------------------------------------------
# DATA LOADING AND SIMULATION
# -----------------------------------------------------------------------------

def create_simulated_data() -> pd.DataFrame:
    """
    Create simulated natural gas price data if no CSV exists.
    
    Returns:
        pd.DataFrame: Dataframe containing simulated price data.
    """
    logger.info("Creating simulated data...")
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    np.random.seed(42)
    base_price = 3.9
    volatility = 0.15
    opens = [base_price]
    
    # Generate price data
    for i in range(1, 100):
        drift = np.random.normal(0, volatility * 0.1)
        bias = -0.005
        opens.append(max(3.7, min(4.1, opens[-1] + drift + bias)))

    highs = [o + np.random.uniform(0.01, volatility * 0.5) for o in opens]
    lows = [o - np.random.uniform(0.01, volatility * 0.5) for o in opens]
    closes = []
    
    for i in range(100):
        if np.random.random() < 0.6:
            closes.append(opens[i] - np.random.uniform(0, (opens[i] - lows[i]) * 0.8))
        else:
            closes.append(opens[i] + np.random.uniform(0, (highs[i] - opens[i]) * 0.8))

    volumes = np.random.randint(1000, 10000, 100)
    daily_avg = pd.Series(closes).rolling(window=5).mean().fillna(method='bfill').tolist()
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'Daily Avg(5)': daily_avg
    })

    # Add support/resistance level touches
    for level in Config.SUPPORT_RESISTANCE_LEVELS:
        for i in range(5, 95, 20):
            df.loc[i:i+5, 'Low'] = level - np.random.uniform(0, 0.03, 6)
            df.loc[i:i+5, 'Close'] = level + np.random.uniform(-0.02, 0.02, 6)

    # Save to temporary file if needed
    import tempfile
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "natural_gas_data_sample.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Simulated data file created at: {csv_path}")
    
    return df


def load_data() -> pd.DataFrame:
    """
    Load price data from CSV or create simulated data if file doesn't exist.
    
    Returns:
        pd.DataFrame: Dataframe containing price data.
    """
    if not os.path.exists(Config.CSV_PATH):
        return create_simulated_data()
    
    logger.info(f"Loading data from {Config.CSV_PATH}")
    df = pd.read_csv(Config.CSV_PATH)
    
    # Try different date formats to handle variability in input files
    date_formats = [
        '%m/%d/%y %H:%M', 
        '%m/%d/%Y %H:%M', 
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt)
            logger.info(f"Successfully parsed dates with format: {fmt}")
            break
        except Exception:
            continue
    else:
        # If none of the specific formats work, try the default parser
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info("Parsed dates with default format")
        except Exception as e:
            logger.error(f"Failed to parse dates: {e}")
            raise ValueError("Could not parse dates in the CSV file.")
    
    return df


# -----------------------------------------------------------------------------
# LLM ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def ask_llm(
    current_data: Dict[str, Any], 
    price_history: List[Dict[str, Any]],
    support_resistance_levels: List[float], 
    current_contracts: int, 
    entry_price: float, 
    last_action: str
) -> Tuple[Dict[str, Any], str]:
    """
    Ask the LLM for trading advice based on market data.
    
    Args:
        current_data: Current candle data
        price_history: Recent price history
        support_resistance_levels: List of support/resistance price levels
        current_contracts: Number of contracts currently held
        entry_price: Average entry price of current position
        last_action: Description of the last action taken
        
    Returns:
        Tuple containing parsed decision dict and raw LLM response
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(
            timeout=Config.OPENAI_TIMEOUT,
            max_retries=Config.OPENAI_MAX_RETRIES
        )
        
        # Get the latest candle data
        open_price = current_data["Open"]
        high_price = current_data["High"]
        low_price = current_data["Low"]
        close_price = current_data["Close"]
        volume = current_data["Volume"]
        timestamp = current_data["Date"]
        
        # Get recent price history for context
        recent_prices = [
            {
                "time": p["time"], 
                "open": p["open"], 
                "high": p["high"], 
                "low": p["low"], 
                "close": p["close"], 
                "volume": p["volume"]
            } 
            for p in price_history
        ]
        
        # Find nearest support/resistance levels
        nearest_levels = sorted(
            [(lvl, abs(close_price - lvl)) for lvl in support_resistance_levels], 
            key=lambda x: x[1]
        )[:3]
        nearest_levels = [{"level": lvl, "distance": dist} for lvl, dist in nearest_levels]
        
        # Calculate some technical indicators for better decision making
        # Check for potential breakout patterns
        breakout_patterns = []
        for lvl in support_resistance_levels:
            # Check if current or previous candle broke through any level
            if open_price > lvl and close_price < lvl:
                breakout_patterns.append({
                    "type": "support_breakdown",
                    "level": lvl,
                    "candle_size": abs(open_price - close_price),
                    "volume": volume,
                    "strength": 0.8 if volume > 5000 else 0.5  # Arbitrary threshold for example
                })
        
        # Check for continuation patterns
        trend_strength = 0.0
        price_trend = "neutral"
        if len(recent_prices) >= 3:
            downs = sum(1 for i in range(min(5, len(recent_prices))) if 
                     recent_prices[i]["close"] < recent_prices[i]["open"])
            ups = sum(1 for i in range(min(5, len(recent_prices))) if 
                    recent_prices[i]["close"] > recent_prices[i]["open"])
            
            if downs >= 3:
                price_trend = "downtrend"
                trend_strength = downs / 5.0 * 0.8
            elif ups >= 3:
                price_trend = "uptrend"
                trend_strength = ups / 5.0 * 0.8
        
        # Prepare data for LLM
        data_for_llm = {
            "current_candle": {
                "timestamp": str(timestamp),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "size_pct": abs(open_price - close_price) / open_price * 100  # Candle size as percentage
            },
            "position": {
                "contracts": current_contracts,
                "direction": "SHORT",
                "entry_price": entry_price,
                "pnl_per_contract": entry_price - close_price,
                "last_action": last_action
            },
            "market_context": {
                "support_resistance_levels": support_resistance_levels,
                "nearest_levels": nearest_levels,
                "recent_price_history": recent_prices[:10],  # Last 10 candles for better context
                "detected_patterns": breakout_patterns,
                "price_trend": price_trend,
                "trend_strength": trend_strength,
                "avg_volume": sum(p["volume"] for p in recent_prices[:5]) / min(5, len(recent_prices)) if recent_prices else 0
            }
        }
        
        # Create prompt with detailed instructions
        prompt = f"""
You are an expert algorithmic trader specializing in natural gas futures. You make decisive trading decisions based on technical analysis patterns, support/resistance levels, and volume indicators.

Here is the current market data in JSON format:
{json.dumps(data_for_llm, indent=2)}

Analyze the data to identify actionable trading opportunities, especially looking for:

1. BREAKOUTS: When price breaks through support/resistance levels
   - Price crossing below a support level is a strong SHORT signal
   - High volume during breakouts significantly increases confidence (>0.8)
   - The closer a breakout occurs to a key level, the stronger the signal

2. CANDLESTICK PATTERNS that indicate continuation or reversal:
   - Bearish engulfing patterns suggest adding to SHORT positions
   - Long bearish candles with small wicks show strong downward momentum
   - Multiple candles touching a level before breaking it increases confidence

3. VOLUME CONFIRMATION:
   - High volume on breakout candles confirms the move (increase confidence by 0.2)
   - Low volume pullbacks to broken levels are opportunities to add positions
   - Volume increasing as price moves away from broken level is very bullish/bearish

BE DECISIVE - when you see clear signals, confidence should be 0.7 or higher.

When multiple signals align (breakout + high volume + bearish pattern), be very confident (0.9+) and take larger positions.

Respond with a JSON object containing:
1. "action": ("ADD_SHORT", "ADD_LONG", "REDUCE_SHORT", "REDUCE_LONG", or "HOLD")
2. "quantity": number of contracts to adjust (1-3 based on confidence)
3. "reasoning": brief explanation of your decision (under 50 words)
4. "confidence": your confidence level (0.0-1.0)

Your response MUST be valid JSON without any text before or after. Format:
{{"action": "ACTION", "quantity": N, "reasoning": "Your reasoning", "confidence": 0.X}}
"""
        logger.debug("Sending prompt to LLM")
        resp = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a natural gas trading algorithm that responds only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        # Get the response and parse it as JSON
        content = resp.choices[0].message.content.strip()
        try:
            # Try to parse the response as JSON
            decision = json.loads(content)
            logger.info(f"LLM decision: {decision['action']} with confidence {decision['confidence']}")
            # Also save the raw response for debugging
            return decision, content
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {content}")
            # If parsing fails, return a default "hold" decision
            return {
                "action": "HOLD", 
                "quantity": 0, 
                "reasoning": "Failed to parse LLM response", 
                "confidence": 0.0
            }, content
    except Exception as e:
        logger.error(f"LLM API error: {str(e)}")
        # If any exception occurs, return a default "hold" decision
        return {
            "action": "HOLD", 
            "quantity": 0, 
            "reasoning": f"LLM error: {str(e)}", 
            "confidence": 0.0
        }, "LLM unavailable."


# -----------------------------------------------------------------------------
# DASH APP LAYOUT
# -----------------------------------------------------------------------------

def create_app(df: pd.DataFrame) -> dash.Dash:
    """
    Create and configure the Dash application.
    
    Args:
        df: DataFrame containing price data
        
    Returns:
        dash.Dash: Configured Dash application
    """
    app = dash.Dash(__name__, external_stylesheets=[Config.THEME])
    server = app.server
    
    # Initial trading state
    initial_contracts = Config.INITIAL_CONTRACTS
    current_contracts = initial_contracts
    last_action = f"Initial entry: {initial_contracts} contracts (SHORT)"
    entry_price = df['Close'].iloc[0]

    action_history = [{
        "time": df['Date'].iloc[0],
        "action": last_action,
        "contracts": current_contracts,
        "price": df['Close'].iloc[0]
    }]
    analysis_history = []
    llm_thoughts = "Starting market analysis for natural gas…"
    
    # App layout
    app.layout = html.Div([
        # Header
        dbc.Row([
            dbc.Col(html.H1("Natural Gas Trading Dashboard with LLM", className="text-center mb-4"), width=12)
        ]),
        
        # Controls
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Simulation Controls"),
                dbc.CardBody([
                    dbc.Button("Start Simulation", id="start-button", color="success", className="me-2"),
                    dbc.Button("Pause", id="pause-button", color="warning", className="me-2"),
                    dbc.Button("Restart", id="reset-button", color="danger"),
                    html.Hr(),
                    html.Label("Speed (seconds per candle):"),
                    dcc.Slider(id="speed-slider", min=0.2, max=3, step=0.2, value=1,
                               marks={i: f"{i}s" for i in [0.2,1,2,3]})
                ])
            ]), width=12, className="mb-4")
        ]),
        
        # Main content
        dbc.Row([
            # Price chart
            dbc.Col(dbc.Card([
                dbc.CardHeader("Price Chart"),
                dbc.CardBody(dcc.Graph(id="price-chart", style={"height":"600px"}, config={"displayModeBar":False}))
            ]), width=8),
            
            # Status and analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Status"),
                    dbc.CardBody([
                        html.H5("Current Position:"), html.H3(id="current-position"),
                        html.Hr(),
                        html.H5("Current Price:"), html.H3(id="current-price"),
                        html.Hr(),
                        html.H5("P&L:"), html.H3(id="pnl")
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardHeader("LLM Analysis"),
                    dbc.CardBody([
                        html.H5("Thoughts:"), html.Pre(id="llm-thoughts", style={
                            "whiteSpace":"pre-wrap","fontSize":"0.85rem",
                            "backgroundColor":"#343a40","padding":"10px","borderRadius":"5px"}),
                        html.Hr(),
                        html.H5("Last action:"), html.Div(id="llm-action", style={"fontWeight":"bold"})
                    ])
                ])
            ], width=4)
        ]),
        
        # History
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader("Action & Analysis History"),
            dbc.CardBody(html.Div(id="action-history", style={"maxHeight":"500px","overflowY":"auto"}))
        ]), width=12), className="mt-4"),
        
        # Store component for state
        dcc.Store(id="simulation-data", data={
            "is_running": False,
            "current_index": 0,
            "speed": 1.0,
            "data": df.to_dict('records'),
            "current_contracts": current_contracts,
            "entry_price": entry_price,
            "action_history": action_history,
            "analysis_history": analysis_history,
            "llm_thoughts": llm_thoughts,
            "last_action": last_action
        }),
        
        # Interval for simulation updates
        dcc.Interval(id="simulation-interval", interval=2000, n_intervals=0, disabled=True)
    ])
    
    return app


# -----------------------------------------------------------------------------
# DASH CALLBACKS
# -----------------------------------------------------------------------------

def register_callbacks(app: dash.Dash) -> None:
    """
    Register all callbacks for the Dash application.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [Output("simulation-interval", "disabled"),
         Output("simulation-interval", "interval"),
         Output("simulation-data", "data", allow_duplicate=True)],
        [Input("start-button", "n_clicks"), Input("pause-button", "n_clicks"),
         Input("reset-button", "n_clicks"), Input("speed-slider", "value")],
        [State("simulation-data", "data")],
        prevent_initial_call=True
    )
    def control_simulation(start, pause, reset, speed, data):
        """Control the simulation (start, pause, reset) and adjust speed."""
        ctx = callback_context
        if not ctx.triggered: 
            return dash.no_update, dash.no_update, dash.no_update
            
        trg = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trg == "start-button":
            data["is_running"] = True
            return False, int(speed*1000), data
            
        if trg == "pause-button":
            data["is_running"] = False
            return True, dash.no_update, data
            
        if trg == "reset-button":
            # Get the initial data from the Config settings
            data.update({
                "is_running": False,
                "current_index": 0,
                "current_contracts": Config.INITIAL_CONTRACTS,
                "entry_price": df['Close'].iloc[0],
                "action_history": [{
                    "time": df['Date'].iloc[0],
                    "action": f"Initial entry: {Config.INITIAL_CONTRACTS} contracts (SHORT)",
                    "contracts": Config.INITIAL_CONTRACTS,
                    "price": df['Close'].iloc[0]
                }],
                "analysis_history": [],
                "llm_thoughts": "Restarting natural gas market analysis…",
                "last_action": f"Initial entry: {Config.INITIAL_CONTRACTS} contracts (SHORT)"
            })
            return True, dash.no_update, data
            
        return dash.no_update, int(speed*1000), dash.no_update

    @app.callback(
        [Output("price-chart", "figure"), Output("current-position", "children"),
         Output("current-price", "children"), Output("pnl", "children"),
         Output("llm-thoughts", "children"), Output("llm-action", "children"),
         Output("action-history", "children"), Output("simulation-data", "data")],
        [Input("simulation-interval", "n_intervals")], 
        [State("simulation-data", "data")]
    )
    def update_simulation(n, data):
        """Update the simulation state and UI elements for each time step."""
        if not data["is_running"]:
            return [dash.no_update] * 8
            
        idx = data["current_index"]
        if idx >= len(data["data"]) - 1:
            data["is_running"] = False
            return [dash.no_update] * 7 + [data]
            
        idx += 1
        data["current_index"] = idx

        recs = data["data"][:idx+1]
        current_price = recs[-1]["Close"]
        open_price = recs[-1]["Open"]

        # LLM for complete market analysis and trading decisions
        if idx % Config.LLM_EVERY == 0:
            # Prepare recent price history with full OHLCV data
            ph = [{
                "time": recs[i]["Date"],
                "open": recs[i]["Open"], 
                "high": recs[i]["High"],
                "low": recs[i]["Low"],
                "close": recs[i]["Close"],
                "volume": recs[i]["Volume"]
            } for i in range(max(0, idx-10), idx+1)]
            
            # Get LLM decision as structured JSON
            decision, raw_response = ask_llm(
                recs[-1],
                ph,
                Config.SUPPORT_RESISTANCE_LEVELS,
                data["current_contracts"],
                data["entry_price"],
                data["last_action"]
            )
            
            # Store the raw response for display
            data["llm_thoughts"] = raw_response
            data["analysis_history"].append({
                "time": recs[-1]["Date"], 
                "price": current_price, 
                "thoughts": raw_response
            })
            
            # Process the LLM's structured decision
            action = decision.get("action", "HOLD")
            quantity = decision.get("quantity", 0)
            reasoning = decision.get("reasoning", "No reasoning provided")
            confidence = decision.get("confidence", 0.0)
            
            # Debug info - always log decision even if HOLD
            logger.debug(f"LLM Decision: {action}, Qty: {quantity}, Conf: {confidence:.2f}, Reason: {reasoning}")
            
            # Execute the trading action if not HOLD and quantity > 0
            # Also log holds with high confidence so we can see what's happening
            if (action != "HOLD" and quantity > 0) or confidence > 0.6:
                old = data["current_contracts"]
                
                if action == "ADD_SHORT":
                    # For high confidence signals, consider using the suggested quantity
                    # For lower confidence, still take at least 1 contract if above threshold
                    adjusted_qty = quantity if confidence >= 0.7 else max(1, quantity-1)
                    data["current_contracts"] += adjusted_qty
                    new = data["current_contracts"]
                    # weighted average of entry_price
                    data["entry_price"] = (data["entry_price"]*old + current_price*adjusted_qty)/new
                    
                    act = (f"LLM DECISION: {action} {adjusted_qty} → {new} contracts @ {current_price:.4f}; "
                          f"AvgEntry={data['entry_price']:.4f}; "
                          f"Reasoning: {reasoning} (Conf: {confidence:.2f})")
                          
                elif action == "REDUCE_SHORT" and old >= quantity:
                    # Adjust quantity if needed based on confidence
                    adjusted_qty = min(quantity, old)  # Can't reduce below 0
                    data["current_contracts"] -= adjusted_qty
                    new = data["current_contracts"]
                    # Entry price stays the same when reducing position
                    
                    act = (f"LLM DECISION: {action} {adjusted_qty} → {new} contracts @ {current_price:.4f}; "
                          f"AvgEntry={data['entry_price']:.4f}; "
                          f"Reasoning: {reasoning} (Conf: {confidence:.2f})")
                    
                elif action == "HOLD" and confidence > 0.6:
                    # Just log the hold decision with high confidence for transparency
                    act = (f"LLM DECISION: {action} @ {current_price:.4f}; "
                          f"Reasoning: {reasoning} (Conf: {confidence:.2f})")
                    
                else:
                    # Unsupported action or invalid quantity
                    act = None
                    
                if act:  # Only update if a valid action was taken
                    data["action_history"].append({
                        "time": recs[-1]["Date"],
                        "action": act,
                        "contracts": data["current_contracts"],
                        "price": current_price
                    })
                    data["last_action"] = act

        # Calculate P&L
        delta = data["entry_price"] - current_price
        pnl = delta * Config.CONTRACT_MULTIPLIER * data["current_contracts"]

        # Create price chart
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1, 
            row_heights=[0.8, 0.2],
            specs=[[{"type": "candlestick"}], [{"type": "bar"}]]
        )
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=[r["Date"] for r in recs], 
            open=[r["Open"] for r in recs],
            high=[r["High"] for r in recs], 
            low=[r["Low"] for r in recs],
            close=[r["Close"] for r in recs], 
            name="Price"
        ), row=1, col=1)
        
        # Add support/resistance levels
        for lvl in Config.SUPPORT_RESISTANCE_LEVELS:
            fig.add_shape(
                type="line", 
                x0=recs[0]["Date"], 
                x1=recs[-1]["Date"],
                y0=lvl, 
                y1=lvl, 
                line=dict(dash="dash"), 
                row=1, 
                col=1
            )
            
        # Add volume bars
        fig.add_trace(go.Bar(
            x=[r["Date"] for r in recs], 
            y=[r["Volume"] for r in recs],
            name="Volume"
        ), row=2, col=1)
        
        # Add action markers
        actions = [a for a in data["action_history"] if a["time"] in [r["Date"] for r in recs]]
        fig.add_trace(go.Scatter(
            x=[a["time"] for a in actions], 
            y=[a["price"] for a in actions],
            mode="markers", 
            marker=dict(symbol="star", size=10), 
            name="Actions"
        ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            template="plotly_dark", 
            legend=dict(orientation="h", y=1.02)
        )
        fig.update_xaxes(type="category")

        # Generate history component
        combo = []
        for a in data["action_history"]:
            combo.append({"time": a["time"], "content": a["action"], "type": "ACTION"})
        for an in data["analysis_history"]:
            combo.append({"time": an["time"], "content": an["thoughts"], "type": "ANALYSIS"})
        combo.sort(key=lambda x: x["time"], reverse=True)
        
        hist = []
        for itm in combo[:20]:
            color = "yellow" if itm["type"] == "ACTION" else "lightblue"
            hist.append(html.Div([
                html.Span(f"{itm['time']}: ", style={"fontWeight": "bold"}),
                html.Span(itm["content"], style={"color": color})
            ], style={"marginBottom": "6px"}))

        return (
            fig,
            f"{data['current_contracts']} Contracts (SHORT)",
            f"{current_price:.4f}",
            f"{pnl:,.2f} USD" + (" (Profit)" if pnl > 0 else " (Loss)"),
            data["llm_thoughts"],
            data["last_action"],
            hist,
            data
        )


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

def main():
    """
    Main entry point for the application.
    """
    try:
        # Load or create data
        df = load_data()
        
        # Create and configure app
        app = create_app(df)
        
        # Register callbacks
        register_callbacks(app)
        
        # Run server
        logger.info("Starting server...")
        app.run_server(debug=True, port=8050)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
