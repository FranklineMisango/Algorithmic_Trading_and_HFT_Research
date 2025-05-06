import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue

from src.models.gm_model_live import GlostenMilgromModelLive
from src.data.data_processor import DataProcessor
from src.utils.config import get_model_config

def create_app():
    # Get model configuration
    model_config = get_model_config()
    
    # Initialize model with config parameters
    gm_model = GlostenMilgromModelLive(
        v_high=model_config["v_high"],
        v_low=model_config["v_low"],
        p=model_config["p"],
        alpha=model_config["alpha"],
        c_dist=lambda x: x  # Using simple linear CDF for now
    )
    
    # Initialize data processor
    data_processor = DataProcessor(
        data_path=model_config["data_path"],
        symbol=model_config["symbol"]
    )
    
    # Start data processing in a separate thread
    data_queue = queue.Queue(maxsize=1000)
    stop_event = threading.Event()
    
    data_thread = threading.Thread(
        target=data_processor.process_live_data,
        args=(data_queue, stop_event, model_config["update_frequency"]),
        daemon=True
    )
    data_thread.start()
    
    # Create Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define app layout
    app.layout = html.Div([
        dbc.Container([
            html.H1("Glosten-Milgrom Live Market Maker Dashboard", className="my-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Market Parameters"),
                    html.Div([
                        html.P(f"Asset high value: {model_config['v_high']}"),
                        html.P(f"Asset low value: {model_config['v_low']}"),
                        html.P(f"Probability of high value (p): {model_config['p']}"),
                        html.P(f"Proportion of informed traders (α): {model_config['alpha']}"),
                    ], className="border rounded p-3 bg-light")
                ], md=4),
                
                dbc.Col([
                    html.H3("Market Data"),
                    html.Div(id="market-stats", className="border rounded p-3 bg-light")
                ], md=8)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Bid-Ask Spread"),
                    dcc.Graph(id="spread-chart")
                ], md=6),
                
                dbc.Col([
                    html.H3("Price Evolution"),
                    dcc.Graph(id="price-chart")
                ], md=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Order Flow Imbalance"),
                    dcc.Graph(id="flow-imbalance-chart")
                ], md=12)
            ]),
            
            dcc.Interval(
                id="interval-component",
                interval=model_config["update_frequency"] * 1000,  # in milliseconds
                n_intervals=0
            ),
            
            # Store for keeping the latest data
            dcc.Store(id="live-data-store")
        ], fluid=True)
    ])
    
    # Define callback to update data store
    @app.callback(
        Output("live-data-store", "data"),
        Input("interval-component", "n_intervals")
    )
    def update_data_store(n):
        # Get the latest data from the queue
        try:
            new_data = data_queue.get(block=False)
            if isinstance(new_data, pd.DataFrame) and not new_data.empty:
                # Process with model
                spreads = gm_model.calculate_spreads_from_data(new_data)
                
                # Combine data and model outputs
                result_data = {
                    "timestamp": new_data["timestamp"].tolist(),
                    "bid_price": new_data["bid_price"].tolist(),
                    "ask_price": new_data["ask_price"].tolist(),
                    "mid_price": new_data["mid_price"].tolist() if "mid_price" in new_data.columns else None,
                    "trade_price": new_data["trade_price"].tolist() if "trade_price" in new_data.columns else None,
                    "volume": new_data["volume"].tolist() if "volume" in new_data.columns else None,
                    "delta_a": spreads["delta_a"],
                    "delta_b": spreads["delta_b"],
                    "gm_ask": spreads["gm_ask"],
                    "gm_bid": spreads["gm_bid"]
                }
                return result_data
        except queue.Empty:
            pass
        
        # Return empty if no new data
        return dash.no_update
    
    # Callback to update market stats
    @app.callback(
        Output("market-stats", "children"),
        Input("live-data-store", "data")
    )
    def update_market_stats(data):
        if not data:
            return "Waiting for data..."
        
        latest_idx = -1
        stats = [
            html.P(f"Last update: {data['timestamp'][latest_idx]}"),
            html.P([
                "Current prices: ",
                html.Span(f"Bid: {data['bid_price'][latest_idx]:.4f}", 
                         style={"color": "blue", "margin-right": "10px"}),
                html.Span(f"Ask: {data['ask_price'][latest_idx]:.4f}", 
                         style={"color": "red"})
            ]),
            html.P(f"GM Model Bid-Ask: {data['gm_bid'][latest_idx]:.4f} - {data['gm_ask'][latest_idx]:.4f}"),
            html.P(f"Spread: {(data['ask_price'][latest_idx] - data['bid_price'][latest_idx]):.4f}")
        ]
        
        return stats
    
    # Callback for spread chart
    @app.callback(
        Output("spread-chart", "figure"),
        Input("live-data-store", "data")
    )
    def update_spread_chart(data):
        if not data or not data["timestamp"]:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=data["ask_price"],
            mode="lines",
            name="Market Ask",
            line=dict(color="red")
        ))
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=data["bid_price"],
            mode="lines",
            name="Market Bid",
            line=dict(color="blue")
        ))
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=data["gm_ask"],
            mode="lines",
            name="GM Ask",
            line=dict(color="darkred", dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=data["gm_bid"],
            mode="lines",
            name="GM Bid",
            line=dict(color="darkblue", dash="dash")
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        
        return fig
    
    # Callback for price chart
    @app.callback(
        Output("price-chart", "figure"),
        Input("live-data-store", "data")
    )
    def update_price_chart(data):
        if not data or not data["timestamp"]:
            return go.Figure()
        
        fig = go.Figure()
        
        if data["trade_price"] and any(p is not None for p in data["trade_price"]):
            fig.add_trace(go.Scatter(
                x=data["timestamp"], 
                y=data["trade_price"],
                mode="markers",
                name="Trades",
                marker=dict(color="black", size=4)
            ))
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=data["mid_price"] if data["mid_price"] else [(a+b)/2 for a,b in zip(data["ask_price"], data["bid_price"])],
            mode="lines",
            name="Mid Price",
            line=dict(color="green")
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        
        return fig
    
    # Callback for flow imbalance chart
    @app.callback(
        Output("flow-imbalance-chart", "figure"),
        Input("live-data-store", "data")
    )
    def update_flow_imbalance(data):
        if not data or not data["timestamp"]:
            return go.Figure()
            
        # Calculate order flow imbalance from bid/ask prices
        # This is a simplified measure - in a real system you would use order book data
        if len(data["ask_price"]) > 5:  # Need some data to calculate
            mid_prices = [(a+b)/2 for a,b in zip(data["ask_price"], data["bid_price"])]
            price_changes = [0] + [mid_prices[i] - mid_prices[i-1] for i in range(1, len(mid_prices))]
            
            # Smoothed measure
            window = min(10, len(price_changes))
            imbalance = pd.Series(price_changes).rolling(window=window, min_periods=1).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data["timestamp"],
                y=imbalance,
                marker_color=['red' if x < 0 else 'green' for x in imbalance]
            ))
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Order Flow Imbalance",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            
            return fig
        else:
            return go.Figure()
    
    # Cleanup on server shutdown
    @app.server.teardown_appcontext
    def shutdown_data_thread(exception=None):
        stop_event.set()
        data_thread.join(timeout=1.0)
    
    return app