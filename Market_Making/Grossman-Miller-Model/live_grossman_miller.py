import json
import time
import pandas as pd
import numpy as np
import websocket
import threading
import os
import traceback
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from collections import deque
import plotly.express as px

# Set page config for Streamlit
st.set_page_config(
    page_title="Grossman-Miller Market Maker Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GrossmanMillerModel:
    def __init__(self, num_mm, gamma, sigma_squared):
        """
        Initializes the Grossman-Miller model.

        Args:
            num_mm (int): Number of market makers.
            gamma (float): Risk aversion parameter.
            sigma_squared (float): Variance of the asset price shock.
        """
        self.num_mm = num_mm
        self.gamma = gamma
        self.sigma_squared = sigma_squared
        self.mu = 0  # Expected asset value (can be dynamic in a more complex model)

    def calculate_price_t1(self, i):
        """
        Calculates the equilibrium price at t=1.

        Args:
            i (float): Liquidity trader's desired trade (positive for sell, negative for buy).

        Returns:
            float: Equilibrium price at t=1.
        """
        return self.mu - self.gamma * self.sigma_squared * (i / (self.num_mm + 1))

    def calculate_quantity_t1(self, i):
        """
        Calculates the quantity of asset held by each MM and LT1 at t=1.

        Args:
            i (float): Liquidity trader's desired trade.

        Returns:
            float: Quantity held by each agent at t=1.
        """
        return i / (self.num_mm + 1)

    def calculate_price_impact(self, i):
        """
        Calculates the price impact (lambda) and the actual trade quantity of LT1.

        Args:
            i (float): Liquidity trader's desired trade.

        Returns:
            tuple: A tuple containing lambda and the trade quantity of LT1.
        """
        trade_quantity_lt1 = i * self.num_mm / (self.num_mm + 1)
        price_impact = -(1 / self.num_mm) * self.gamma * self.sigma_squared
        return price_impact, trade_quantity_lt1

    def process_trade(self, i, current_price):
        """
        Process a single trade through the Grossman-Miller model.
        
        Args:
            i (float): Liquidity trader's desired trade.
            current_price (float): Current market price.
            
        Returns:
            dict: Results of the model's calculation for this trade.
        """
        self.mu = current_price  # Update expected value to current price
        
        price_t1 = self.calculate_price_t1(i)
        quantity_t1 = self.calculate_quantity_t1(i)
        price_impact, trade_quantity_lt1 = self.calculate_price_impact(i)
        
        return {
            "trade": i,
            "price_t1": price_t1,
            "quantity_t1": quantity_t1,
            "price_impact": price_impact,
            "trade_quantity_lt1": trade_quantity_lt1
        }


class LiveMarketDataCollector:
    def __init__(self, symbol, max_data_points=1000):
        """
        Initialize the live market data collector.
        
        Args:
            symbol (str): Trading symbol (e.g., "btcusdt")
            max_data_points (int): Maximum number of data points to keep in memory
        """
        self.symbol = symbol.lower()
        self.max_data_points = max_data_points
        self.data = deque(maxlen=max_data_points)
        self.raw_data = deque(maxlen=max_data_points)
        self.running = True
        self.last_processed_time = 0
        self.trades_ws = None
        self.depth_ws = None
        self.lock = threading.Lock()
        
        # Create placeholder for last prices
        self.last_bid = None
        self.last_ask = None
        self.last_trade = None
        
        # Model results
        self.model_results = deque(maxlen=max_data_points)
        self.cumulative_pnl = 0
        self.trade_count = 0
        
        # Start WebSocket connections
        self.start_websockets()
    
    def on_depth_message(self, ws, message):
        try:
            data = json.loads(message)
            timestamp = dt.datetime.fromtimestamp(data['E']/1000)
            
            # Extract best bid and ask
            if len(data.get('b', [])) > 0 and len(data.get('a', [])) > 0:
                bid_price = float(data['b'][0][0])
                ask_price = float(data['a'][0][0])
                
                with self.lock:
                    self.last_bid = bid_price
                    self.last_ask = ask_price
                    self.raw_data.append({
                        'timestamp': timestamp,
                        'bid_price': bid_price,
                        'ask_price': ask_price,
                        'type': 'depth'
                    })
        except Exception as e:
            st.error(f"Depth message error: {str(e)[:200]}")
    
    def on_trade_message(self, ws, message):
        try:
            data = json.loads(message)
            timestamp = dt.datetime.fromtimestamp(data['E']/1000)
            trade_price = float(data['p'])
            trade_qty = float(data['q'])
            trade_volume = trade_price * trade_qty
            is_buyer_maker = data['m']  # True if buyer is maker
            
            with self.lock:
                self.last_trade = trade_price
                self.raw_data.append({
                    'timestamp': timestamp,
                    'trade_price': trade_price,
                    'volume': trade_volume,
                    'is_buyer_maker': is_buyer_maker,
                    'type': 'trade'
                })
        except Exception as e:
            st.error(f"Trade message error: {str(e)[:200]}")
    
    def start_websockets(self):
        # Start WebSocket for depth (order book)
        def run_depth_ws():
            while self.running:
                try:
                    self.depth_ws = websocket.WebSocketApp(
                        f"wss://stream.binance.com:9443/ws/{self.symbol}@depth@100ms",
                        on_message=self.on_depth_message,
                        on_error=lambda ws, e: st.error(f"Depth WS error: {str(e)[:200]}"),
                        on_close=lambda ws: print(f"Depth WS closed")
                    )
                    self.depth_ws.run_forever(ping_interval=30, ping_timeout=10)
                    if not self.running:
                        break
                    time.sleep(1)  # Wait before reconnecting
                except Exception as e:
                    print(f"Depth WS error: {str(e)}")
                    time.sleep(5)
        
        # Start WebSocket for trades
        def run_trade_ws():
            while self.running:
                try:
                    self.trades_ws = websocket.WebSocketApp(
                        f"wss://stream.binance.com:9443/ws/{self.symbol}@trade",
                        on_message=self.on_trade_message,
                        on_error=lambda ws, e: st.error(f"Trade WS error: {str(e)[:200]}"),
                        on_close=lambda ws: print(f"Trade WS closed")
                    )
                    self.trades_ws.run_forever(ping_interval=30, ping_timeout=10)
                    if not self.running:
                        break
                    time.sleep(1)  # Wait before reconnecting
                except Exception as e:
                    print(f"Trade WS error: {str(e)}")
                    time.sleep(5)
        
        # Start the WebSocket threads
        depth_thread = threading.Thread(target=run_depth_ws)
        trade_thread = threading.Thread(target=run_trade_ws)
        
        depth_thread.daemon = True
        trade_thread.daemon = True
        
        depth_thread.start()
        trade_thread.start()
    
    def process_data(self, threshold_pct=0.75, model=None):
        """
        Process the raw data and apply the Grossman-Miller model.
        
        Args:
            threshold_pct (float): Percentile threshold for significant trades
            model (GrossmanMillerModel): Model to apply to the trades
            
        Returns:
            tuple: Processed data and model results
        """
        with self.lock:
            if not self.raw_data:
                return pd.DataFrame(), pd.DataFrame()
            
            # Convert raw data to DataFrame
            raw_df = pd.DataFrame(list(self.raw_data))
            
            # Skip if no new data since last processing
            if raw_df.empty or (self.last_processed_time and 
                               raw_df['timestamp'].max().timestamp() <= self.last_processed_time):
                return pd.DataFrame(), pd.DataFrame()
            
            self.last_processed_time = raw_df['timestamp'].max().timestamp()
            
            # Process the data to create a unified time series
            depth_data = raw_df[raw_df['type'] == 'depth'].copy()
            trade_data = raw_df[raw_df['type'] == 'trade'].copy()
            
            if depth_data.empty or trade_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Resample data to 100ms intervals
            if not depth_data.empty:
                depth_data = depth_data.set_index('timestamp')
                depth_data = depth_data.resample('100ms').agg({
                    'bid_price': 'last',
                    'ask_price': 'last'
                }).dropna()
            
            if not trade_data.empty:
                trade_data = trade_data.set_index('timestamp')
                trade_data = trade_data.resample('100ms').agg({
                    'trade_price': 'last',
                    'volume': 'sum',
                    'is_buyer_maker': 'last'
                }).dropna()
            
            # Merge the resampled data
            processed_data = pd.merge_asof(
                depth_data.reset_index(),
                trade_data.reset_index(),
                on='timestamp',
                direction='nearest'
            )
            
            # Calculate mid price
            processed_data['mid_price'] = (processed_data['bid_price'] + processed_data['ask_price']) / 2
            
            # Apply the model if provided
            model_results = pd.DataFrame()
            if model and not processed_data.empty:
                # Calculate threshold for significant trades
                volume_threshold = processed_data['volume'].quantile(threshold_pct)
                
                # Apply the model to each row
                for idx, row in processed_data.iterrows():
                    # Skip rows with missing data
                    if pd.isna(row['volume']) or pd.isna(row['mid_price']):
                        continue
                    
                    # Only process significant trades
                    trade_size = 0
                    if row['volume'] > volume_threshold:
                        # Scale trade size
                        price_level = row['mid_price']
                        divisor = 10 if price_level < 100 else 100
                        trade_size = (row['volume'] - volume_threshold) / divisor
                        
                        # Negative for buying, positive for selling
                        if row['is_buyer_maker']:
                            trade_size = -trade_size
                    
                    # Apply the model to get new metrics
                    model_output = model.process_trade(trade_size, row['mid_price'])
                    
                    # Add timestamp and market data
                    model_output['timestamp'] = row['timestamp']
                    model_output['mid_price'] = row['mid_price']
                    model_output['volume'] = row['volume']
                    
                    # Store results
                    self.model_results.append(model_output)
                    
                    # If this is a significant trade, count it
                    if trade_size != 0:
                        self.trade_count += 1
                
                # Convert model results to DataFrame for the current batch
                if self.model_results:
                    model_results = pd.DataFrame(list(self.model_results))
                    
                    # Calculate PnL
                    if len(model_results) > 1:
                        model_results['price_change'] = model_results['mid_price'].diff()
                        model_results['mm_pnl'] = -model_results['price_impact'] * model_results['trade_quantity_lt1'] * model_results['price_change']
                        
                        # Update cumulative PnL
                        self.cumulative_pnl += model_results['mm_pnl'].fillna(0).sum()
                    
            # Store processed data for later use
            for row in processed_data.to_dict('records'):
                self.data.append(row)
            
            return processed_data, model_results
    
    def get_latest_data(self):
        """Get the latest processed data as a DataFrame."""
        with self.lock:
            return pd.DataFrame(list(self.data))
    
    def get_latest_model_results(self):
        """Get the latest model results as a DataFrame."""
        with self.lock:
            return pd.DataFrame(list(self.model_results))
    
    def stop(self):
        """Stop all WebSocket connections."""
        self.running = False
        if self.depth_ws:
            self.depth_ws.close()
        if self.trades_ws:
            self.trades_ws.close()


def create_dashboard():
    st.title("Live Grossman-Miller Market Making Dashboard")
    
    # Sidebar for parameters
    st.sidebar.header("Model Parameters")
    
    symbol = st.sidebar.text_input("Trading Symbol", value="btcusdt").lower()
    num_market_makers = st.sidebar.slider("Number of Market Makers", min_value=1, max_value=50, value=10)
    gamma = st.sidebar.slider("Risk Aversion (gamma)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    sigma_squared = st.sidebar.slider("Price Variance (sigma²)", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    # Initialize session state
    if 'started' not in st.session_state:
        st.session_state.started = False
        st.session_state.data_collector = None
        st.session_state.model = None
    
    # Start/Stop button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Start" if not st.session_state.started else "Restart"):
            # Stop existing collector if running
            if st.session_state.data_collector:
                st.session_state.data_collector.stop()
            
            # Initialize model and data collector
            st.session_state.model = GrossmanMillerModel(num_market_makers, gamma, sigma_squared)
            st.session_state.data_collector = LiveMarketDataCollector(symbol)
            st.session_state.started = True
            st.rerun()
    
    with col2:
        if st.button("Stop") and st.session_state.started:
            if st.session_state.data_collector:
                st.session_state.data_collector.stop()
            st.session_state.started = False
            st.rerun()

    if not st.session_state.started:
        st.info("Set parameters and click 'Start' to begin the live dashboard.")
        return
    
    # Dashboard layout with tabs
    tab1, tab2, tab3 = st.tabs(["Live Performance", "Model Metrics", "Raw Data"])
    
    with tab1:
        # Create placeholders for charts and metrics
        metrics_placeholder = st.empty()
        price_chart_placeholder = st.empty()
        pnl_chart_placeholder = st.empty()
    
    with tab2:
        model_metrics_placeholder = st.empty()
        price_impact_placeholder = st.empty()
        trade_dist_placeholder = st.empty()
    
    with tab3:
        raw_data_placeholder = st.empty()
    
    # Function to update the dashboard
    def update_dashboard():
        try:
            if not st.session_state.started or not st.session_state.data_collector:
                return
            
            collector = st.session_state.data_collector
            model = st.session_state.model
            
            # Process new data and apply model
            new_data, model_results = collector.process_data(model=model)
            
            # Get all processed data
            all_data = collector.get_latest_data()
            all_model_results = collector.get_latest_model_results()
            
            if all_data.empty:
                return
            
            # Update Tab 1: Live Performance
            with metrics_placeholder.container():
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = collector.last_trade or 0
                    st.metric("Current Price", f"{current_price:.2f}")
                
                with col2:
                    bid_ask_spread = (collector.last_ask or 0) - (collector.last_bid or 0)
                    st.metric("Bid-Ask Spread", f"{bid_ask_spread:.4f}")
                
                with col3:
                    st.metric("Cumulative PnL", f"{collector.cumulative_pnl:.6f}")
                
                with col4:
                    st.metric("Trade Count", collector.trade_count)
            
            # For the price chart (around line 465)
            with price_chart_placeholder.container():
                if not all_data.empty and 'timestamp' in all_data.columns and 'mid_price' in all_data.columns:
                    # Create price chart with bid-ask spread
                    # Use separate subplots instead of secondary_y for better scaling
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Price", "Volume")
                    )
                    
                    # Get recent data for better visualization (last 100 points)
                    display_data = all_data.tail(100).copy()
                    
                    # Plot mid price
                    fig.add_trace(
                        go.Scatter(
                            x=display_data['timestamp'], 
                            y=display_data['mid_price'],
                            mode='lines',
                            name='Mid Price',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add bid-ask if available
                    if 'bid_price' in display_data.columns and 'ask_price' in display_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=display_data['timestamp'],
                                y=display_data['bid_price'],
                                mode='lines',
                                line=dict(width=1),
                                name='Bid Price'
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=display_data['timestamp'],
                                y=display_data['ask_price'],
                                mode='lines',
                                line=dict(width=1),
                                name='Ask Price',
                                fill='tonexty',
                                fillcolor='rgba(68, 68, 68, 0.2)'
                            ),
                            row=1, col=1
                        )
                    
                    # Add volume as separate subplot
                    if 'volume' in display_data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=display_data['timestamp'],
                                y=display_data['volume'],
                                name='Volume',
                                marker_color='rgba(68, 68, 68, 0.5)'
                            ),
                            row=2, col=1
                        )
                    
                    # Calculate y-axis range with padding for price
                    if len(display_data) > 0:
                        y_min = display_data['mid_price'].min() * 0.9995 if 'mid_price' in display_data else None
                        y_max = display_data['mid_price'].max() * 1.0005 if 'mid_price' in display_data else None
                        
                        if y_min and y_max:
                            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
                    
                    # Update layout for better readability
                    fig.update_layout(
                        title=f"{symbol.upper()} Price and Volume",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500,  # Increased height
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=True,
                        xaxis_rangeslider_visible=False,
                        plot_bgcolor='white',
                        hovermode='x unified'
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgrey',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='lightgrey'
                    )
                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgrey',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='lightgrey'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

         
            with pnl_chart_placeholder.container():
                if not all_model_results.empty and 'timestamp' in all_model_results.columns:
                    # Use only recent data points for better visualization
                    display_data = all_model_results.tail(100).copy()
                    
                    # Create separate charts for cumulative PnL and individual trades
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Cumulative PnL", "Individual Trade PnL")
                    )
                    
                    # Ensure mm_pnl exists
                    if 'mm_pnl' not in display_data.columns:
                        display_data['mm_pnl'] = 0
                    
                    # Calculate cumulative PnL
                    display_data['cumulative_pnl'] = display_data['mm_pnl'].fillna(0).cumsum()
                    
                    # Cumulative PnL line
                    fig.add_trace(
                        go.Scatter(
                            x=display_data['timestamp'],
                            y=display_data['cumulative_pnl'],
                            mode='lines',
                            name='Cumulative PnL',
                            line=dict(color='green', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Individual trade PnLs as markers
                    fig.add_trace(
                        go.Bar(
                            x=display_data['timestamp'],
                            y=display_data['mm_pnl'],
                            name='Trade PnL',
                            marker=dict(
                                color=display_data['mm_pnl'].apply(
                                    lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray'
                                )
                            )
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=450,  # Increased height
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=True,
                        hovermode='x unified',
                        plot_bgcolor='white'
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgrey'
                    )
                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgrey'
                    )
                    
            
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with price_impact_placeholder.container():
                if not all_model_results.empty and 'timestamp' in all_model_results.columns and 'price_impact' in all_model_results.columns:
                    # Make a copy to avoid modifying the original
                    plot_data = all_model_results.copy()
                    
                    # Check if mm_pnl exists, if not create it with zeros
                    if 'mm_pnl' not in plot_data.columns:
                        plot_data['mm_pnl'] = 0
                        color_var = None
                        hover_columns = ['timestamp', 'mid_price']
                    else:
                        color_var = 'mm_pnl'
                        hover_columns = ['timestamp', 'mid_price', 'mm_pnl']
                    
                    # Create price impact vs trade size chart with better visual elements
                    if color_var:
                        fig = px.scatter(
                            plot_data,
                            x='trade',
                            y='price_impact',
                            color=color_var,
                            color_continuous_scale='RdYlGn',
                            title="Price Impact vs Trade Size",
                            labels={'trade': 'Trade Size', 'price_impact': 'Price Impact'},
                            hover_data=hover_columns,
                            opacity=0.7,  # Add transparency for overlapping points
                            size_max=15,  # Limit maximum marker size
                        )
                    else:
                        fig = px.scatter(
                            plot_data,
                            x='trade',
                            y='price_impact',
                            title="Price Impact vs Trade Size",
                            labels={'trade': 'Trade Size', 'price_impact': 'Price Impact'},
                            hover_data=hover_columns,
                            opacity=0.7
                        )
                    
                    # Improve layout
                    fig.update_layout(
                        height=350,  # Increased height
                        margin=dict(l=10, r=10, t=40, b=10),
                        plot_bgcolor='white',
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgrey',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgrey',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        )
                    )
                    
                    # Add a reference line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                        
            # Trade distribution
            with trade_dist_placeholder.container():
                if not all_model_results.empty and 'trade' in all_model_results.columns:
                    # Only include non-zero trades
                    trade_data = all_model_results[all_model_results['trade'] != 0]
                    
                    if not trade_data.empty:
                        fig = px.histogram(
                            trade_data,
                            x='trade',
                            title="Trade Size Distribution",
                            labels={'trade': 'Trade Size'},
                            nbins=30
                        )
                        
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
            with raw_data_placeholder.container():
                # Display the most recent 100 rows
                if not all_data.empty:
                    display_cols = ['timestamp', 'bid_price', 'ask_price', 'trade_price', 
                                   'volume', 'mid_price']
                    display_cols = [col for col in display_cols if col in all_data.columns]
                    
                    st.dataframe(all_data[display_cols].tail(100))
        
        except Exception as e:
            st.error(f"Dashboard update error: {str(e)}")
            traceback.print_exc()
    
    # Update the dashboard
    update_dashboard()
    
    # Set up periodic refresh
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=1)
    
    
    if st.sidebar.checkbox("Auto-refresh", value=True):
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    create_dashboard()