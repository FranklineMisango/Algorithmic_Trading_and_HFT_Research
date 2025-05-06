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
            
            # Add debug info panel inside the layout
            dbc.Row([
                dbc.Col([
                    html.H3("Debug Information", className="mt-3"),
                    html.Div(id="debug-info", className="border rounded p-3 bg-light")
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

    @app.callback(
        Output("live-data-store", "data"),
        Input("interval-component", "n_intervals")
    )
    def update_data_store(n):
        try:
            # Add more debugging info
            print(f"Update {n}: Queue size: {data_queue.qsize() if hasattr(data_queue, 'qsize') else 'unknown'}")
            print(f"Collector buffer size: {len(data_processor.collector.combined_data)}")
            
            # Get the latest data from the queue
            try:
                if data_queue.empty():
                    print("Queue is empty - forcing data collection")
                    # If queue is empty but collector has data, manually add it to queue
                    if len(data_processor.collector.combined_data) > 0:
                        buffer_copy = pd.DataFrame(data_processor.collector.combined_data.copy())
                        if not buffer_copy.empty:
                            buffer_copy['timestamp'] = pd.to_datetime(buffer_copy['timestamp'])
                            buffer_copy = buffer_copy.drop_duplicates(subset=['timestamp'], keep='last')
                            
                            # Calculate mid price
                            buffer_copy['mid_price'] = (pd.to_numeric(buffer_copy['bid_price'], errors='coerce') + 
                                            pd.to_numeric(buffer_copy['ask_price'], errors='coerce')) / 2
                            
                            # Try to put in queue
                            try:
                                data_queue.put(buffer_copy, block=False)
                                print(f"Manually added {len(buffer_copy)} records to queue")
                            except queue.Full:
                                pass
                
                # Now try to get data from queue
                new_data = data_queue.get(block=False)
                if isinstance(new_data, pd.DataFrame) and not new_data.empty:
                    print(f"Got DataFrame from queue with {len(new_data)} rows")
                    
                    # Process with model
                    spreads = gm_model.calculate_spreads_from_data(new_data)
                    
                    # Convert timestamps to strings for JSON serialization
                    timestamps = new_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').tolist()
                    
                    # Ensure all data is properly converted to lists
                    bid_prices = new_data["bid_price"].tolist() if "bid_price" in new_data.columns else []
                    ask_prices = new_data["ask_price"].tolist() if "ask_price" in new_data.columns else []
                    mid_prices = new_data["mid_price"].tolist() if "mid_price" in new_data.columns else None
                    trade_prices = new_data["trade_price"].tolist() if "trade_price" in new_data.columns else None
                    volumes = new_data["volume"].tolist() if "volume" in new_data.columns else None
                    
                    # Combine data and model outputs
                    result_data = {
                        "timestamp": timestamps,
                        "bid_price": bid_prices,
                        "ask_price": ask_prices,
                        "mid_price": mid_prices,
                        "trade_price": trade_prices,
                        "volume": volumes,
                        "delta_a": spreads["delta_a"],
                        "delta_b": spreads["delta_b"],
                        "gm_ask": spreads["gm_ask"],
                        "gm_bid": spreads["gm_bid"]
                    }
                    
                    # Validate the data - all arrays should be the same length
                    if len(timestamps) > 0:
                        length = len(timestamps)
                        if len(bid_prices) != length or len(ask_prices) != length:
                            print(f"Data length mismatch: timestamps={len(timestamps)}, bid={len(bid_prices)}, ask={len(ask_prices)}")
                            # Fix lengths if needed
                            result_data["bid_price"] = bid_prices[:length] if len(bid_prices) > length else bid_prices + [None] * (length - len(bid_prices))
                            result_data["ask_price"] = ask_prices[:length] if len(ask_prices) > length else ask_prices + [None] * (length - len(ask_prices))
                        
                    print(f"Updating dashboard with {len(timestamps)} points")
                    return result_data
            except queue.Empty:
                print("No new data in queue")
                pass
            
            return dash.no_update
        except Exception as e:
            print(f"Error in update_data_store: {str(e)}")
            import traceback
            traceback.print_exc()
            return dash.no_update
    
    @app.callback(
        Output("market-stats", "children"),
        Input("live-data-store", "data")
    )
    def update_market_stats(data):
        try:
            if not data or not data.get("timestamp", []):
                return "Waiting for data..."
            
            # Make sure we have data to show
            if not data["timestamp"]:
                return "No data points available"
                
            latest_idx = -1
            
            # Make sure the index is valid
            if abs(latest_idx) > len(data["timestamp"]):
                latest_idx = 0
                
            # Convert values to float and handle None values
            try:
                bid_price = float(data['bid_price'][latest_idx]) if data['bid_price'][latest_idx] is not None else 0.0
                ask_price = float(data['ask_price'][latest_idx]) if data['ask_price'][latest_idx] is not None else 0.0
                gm_bid = float(data['gm_bid'][latest_idx]) if data['gm_bid'][latest_idx] is not None else 0.0
                gm_ask = float(data['gm_ask'][latest_idx]) if data['gm_ask'][latest_idx] is not None else 0.0
                
                stats = [
                    html.P(f"Last update: {data['timestamp'][latest_idx]}"),
                    html.P([
                        "Current prices: ",
                        html.Span(f"Bid: {bid_price:.4f}", 
                                style={"color": "blue", "margin-right": "10px"}),
                        html.Span(f"Ask: {ask_price:.4f}", 
                                style={"color": "red"})
                    ]),
                    html.P(f"GM Model Bid-Ask: {gm_bid:.4f} - {gm_ask:.4f}"),
                    html.P(f"Spread: {(ask_price - bid_price):.4f}")
                ]
                return stats
            except (IndexError, TypeError) as e:
                return html.P(f"Error processing market data: {str(e)}")
                
        except Exception as e:
            return html.P(f"Error updating market stats: {str(e)}")

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
        
        # Fix mid price calculation to handle None values
        if data["mid_price"]:
            mid_prices = data["mid_price"]
        else:
            # Calculate mid prices with None handling
            mid_prices = []
            for a, b in zip(data["ask_price"], data["bid_price"]):
                if a is not None and b is not None:
                    mid_prices.append((float(a) + float(b)) / 2)
                else:
                    mid_prices.append(None)
        
        fig.add_trace(go.Scatter(
            x=data["timestamp"], 
            y=mid_prices,
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
    
    # Callback for debug info
    @app.callback(
        Output("debug-info", "children"),
        Input("interval-component", "n_intervals")
    )
    def update_debug_info(n):
        try:
            queue_size = data_queue.qsize() if hasattr(data_queue, 'qsize') else "N/A"
            websocket_status = "Unknown"
            connection_status = "Disconnected"
            
            if hasattr(data_processor, 'collector'):
                # Check if combined_data is being updated
                data_age = None
                if hasattr(data_processor.collector, '_last_data_time'):
                    data_age = datetime.now() - data_processor.collector._last_data_time
                    if data_age.total_seconds() < 5:
                        connection_status = "Connected"
                    elif data_age.total_seconds() < 30:
                        connection_status = "Intermittent"
                
                # Update the current time for next comparison
                data_processor.collector._last_data_time = datetime.now()
                
                # Update status based on data flow
                if len(data_processor.collector.combined_data) > 0:
                    websocket_status = f"Active ({len(data_processor.collector.combined_data)} records)"
                else:
                    websocket_status = "No data"
            
            # Status indicator style
            status_style = {
                "Connected": {"color": "green", "fontWeight": "bold"},
                "Intermittent": {"color": "orange", "fontWeight": "bold"},
                "Disconnected": {"color": "red", "fontWeight": "bold"}
            }
            
            # More detailed information
            return [
                html.P(f"Update #{n} | Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"),
                html.P([
                    "Connection status: ",
                    html.Span(connection_status, style=status_style.get(connection_status, {}))
                ]),
                html.P(f"Collector buffer: {len(data_processor.collector.combined_data)} records"),
                html.P(f"Processor buffer: {len(data_processor.buffer)} records"),
                html.P(f"Queue size: {queue_size}"),
                html.P(f"WebSocket status: {websocket_status}"),
                html.Button("Force Refresh Data", id="refresh-button", className="btn btn-primary mt-2")
            ]
        except Exception as e:
            return html.P(f"Error in debug info: {str(e)}")
    
    @app.callback(
        Output("flow-imbalance-chart", "figure"),
        Input("live-data-store", "data")
    )
    def update_flow_imbalance(data):
        try:
            if not data or not data["timestamp"]:
                return go.Figure()
                
            # Calculate order flow imbalance from bid/ask prices
            # This is a simplified measure - in a real system you would use order book data
            if len(data["ask_price"]) > 5:  # Need some data to calculate
                # Filter out None values before calculating mid prices
                valid_indices = []
                for i, (a, b) in enumerate(zip(data["ask_price"], data["bid_price"])):
                    if a is not None and b is not None:
                        valid_indices.append(i)
                
                if len(valid_indices) <= 5:  # Not enough valid data points
                    return go.Figure()
                    
                # Get valid timestamps and prices
                timestamps = [data["timestamp"][i] for i in valid_indices]
                valid_asks = [float(data["ask_price"][i]) for i in valid_indices]
                valid_bids = [float(data["bid_price"][i]) for i in valid_indices]
                
                # Now calculate mid prices safely
                mid_prices = [(a + b)/2 for a, b in zip(valid_asks, valid_bids)]
                
                # Calculate price changes
                price_changes = [0]
                for i in range(1, len(mid_prices)):
                    change = mid_prices[i] - mid_prices[i-1]
                    price_changes.append(change)
                
                # Smoothed measure
                window = min(10, len(price_changes))
                imbalance = pd.Series(price_changes).rolling(window=window, min_periods=1).mean()
                
                fig = go.Figure()
                
                # Create a list of colors based on the imbalance values
                colors = ['red' if x < 0 else 'green' for x in imbalance]
                
                fig.add_trace(go.Bar(
                    x=timestamps,
                    y=imbalance,
                    marker_color=colors
                ))
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Order Flow Imbalance",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                
                return fig
            else:
                return go.Figure()
        except Exception as e:
            print(f"Error in flow imbalance chart: {str(e)}")
            return go.Figure()
    
    # Callback for the refresh button
    @app.callback(
        Output("refresh-button", "n_clicks"),
        Input("refresh-button", "n_clicks"),
        prevent_initial_call=True
    )
    def force_refresh(n_clicks):
        if n_clicks:
            # Force a data refresh
            print("Manual refresh triggered")
            # This could potentially clear the queue and force new data collection
            while not data_queue.empty():
                try:
                    data_queue.get(block=False)
                except:
                    break
        return 0
    
    # Cleanup on server shutdown
    @app.server.teardown_appcontext
    def shutdown_data_thread(exception=None):
        stop_event.set()
        data_thread.join(timeout=1.0)
    
    return app