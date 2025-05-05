import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import queue

class KyleDashboard:
    def __init__(self, update_interval=1000):
        """
        Initialize the dashboard.
        
        Args:
            update_interval (int): Update interval in milliseconds
        """
        self.app = dash.Dash(__name__, update_title=None)
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        self.model_metrics = {}
        self.recent_market_data = pd.DataFrame()
        self.model_history = pd.DataFrame()
        
        # Set up the app layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Frankline & Co. LP Modified Kyle Model : Trading with information disadvantage dashboard", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    html.H3("Market Data", style={'textAlign': 'center'}),
                    dcc.Graph(id='market-data-graph')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Order Flow", style={'textAlign': 'center'}),
                    dcc.Graph(id='order-flow-graph')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H3("Key Metrics", style={'textAlign': 'center'}),
                    html.Div(id='metrics-container', style={
                        'display': 'flex', 
                        'flexWrap': 'wrap',
                        'justifyContent': 'space-around'
                    })
                ], className='six columns'),
                
                html.Div([
                    html.H3("Information Revelation", style={'textAlign': 'center'}),
                    dcc.Graph(id='info-revelation-graph')
                ], className='six columns')
            ], className='row'),
            
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ], style={'padding': '20px', 'fontFamily': 'Arial'})
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [Output('market-data-graph', 'figure'),
             Output('order-flow-graph', 'figure'),
             Output('info-revelation-graph', 'figure'),
             Output('metrics-container', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(_):
            # Process all available updates in the queue
            while not self.data_queue.empty():
                try:
                    update = self.data_queue.get_nowait()
                    if 'market_data' in update:
                        self.recent_market_data = update['market_data']
                    if 'model_metrics' in update:
                        self.model_metrics = update['model_metrics']
                    if 'model_history' in update:
                        self.model_history = update['model_history']
                except queue.Empty:
                    break
            
            # Create market data figure
            market_fig = self._create_market_data_figure()
            
            # Create order flow figure
            order_flow_fig = self._create_order_flow_figure()
            
            # Create information revelation figure
            info_revelation_fig = self._create_info_revelation_figure()
            
            # Create metrics cards
            metrics_cards = self._create_metrics_cards()
            
            return market_fig, order_flow_fig, info_revelation_fig, metrics_cards
    
    def _create_market_data_figure(self):
        """Create the market data figure."""
        if self.recent_market_data.empty:
            return go.Figure()
        
        df = self.recent_market_data.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Add mid price
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['mid_price'],
            mode='lines',
            name='Mid Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add model price if available
        if 'price' in self.model_metrics:
            # Create a simple line at the current model price
            last_time = df['timestamp'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_time - timedelta(seconds=5), last_time],
                y=[self.model_metrics['price'], self.model_metrics['price']],
                mode='lines',
                name='Kyle Model Price',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Add true value if available
        if 'true_value' in self.model_metrics:
            # Create a simple line at the estimated true value
            last_time = df['timestamp'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_time - timedelta(seconds=5), last_time],
                y=[self.model_metrics['true_value'], self.model_metrics['true_value']],
                mode='lines',
                name='Estimated True Value',
                line=dict(color='green', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title='Market Prices vs. Model',
            xaxis_title='Time',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        return fig
    
    def _create_order_flow_figure(self):
        """Create the order flow figure."""
        if self.model_history.empty:
            return go.Figure()
        
        df = self.model_history.copy()
        
        fig = go.Figure()
        
        # Add total order flow
        fig.add_trace(go.Bar(
            x=df['timestamp'], 
            y=df['order_flow'],
            name='Total Order Flow',
            marker_color='blue'
        ))
        
        # Add insider and noise components if available
        if 'insider_order' in df.columns and 'noise_order' in df.columns:
            fig.add_trace(go.Bar(
                x=df['timestamp'], 
                y=df['insider_order'],
                name='Insider Orders',
                marker_color='green'
            ))
            
            fig.add_trace(go.Bar(
                x=df['timestamp'], 
                y=df['noise_order'],
                name='Noise Orders',
                marker_color='orange'
            ))
        
        fig.update_layout(
            title='Order Flow Components',
            xaxis_title='Time',
            yaxis_title='Order Size',
            barmode='group',
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        return fig
    
    def _create_info_revelation_figure(self):
        """Create the information revelation figure."""
        if self.model_history.empty:
            return go.Figure()
        
        df = self.model_history.copy()
        
        # Filter to include only the last 50 data points for clarity
        df = df.tail(50)
        
        fig = go.Figure()
        
        # Add information revelation
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['info_revelation'],
            mode='lines+markers',
            name='Information Revelation',
            line=dict(color='purple', width=2),
            marker=dict(size=8)
        ))
        
        # Add a reference line at 0.5 (Kyle model predicts 50% info revelation)
        fig.add_trace(go.Scatter(
            x=[df['timestamp'].min(), df['timestamp'].max()],
            y=[0.5, 0.5],
            mode='lines',
            name='Theoretical (50%)',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Information Revelation Metric',
            xaxis_title='Time',
            yaxis_title='Information Revealed (0-1)',
            yaxis=dict(range=[0, 1]),
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        return fig
    
    def _create_metrics_cards(self):
        """Create the metrics cards."""
        metrics = []
        card_style = {
            'width': '45%',
            'padding': '10px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'margin': '10px',
            'textAlign': 'center',
            'boxShadow': '2px 2px 2px lightgrey'
        }
        
        # Lambda (price impact)
        if 'lambda' in self.model_metrics:
            metrics.append(html.Div([
                html.H4("Price Impact (λ)"),
                html.H2(f"{self.model_metrics['lambda']:.6f}"),
                html.P("Higher values indicate greater price impact per unit of order flow")
            ], style=card_style))
        
        # Market depth
        if 'market_depth' in self.model_metrics:
            metrics.append(html.Div([
                html.H4("Market Depth (1/λ)"),
                html.H2(f"{self.model_metrics['market_depth']:.2f}"),
                html.P("Higher values indicate more liquid markets")
            ], style=card_style))
        
        # Information revelation
        if 'info_revelation' in self.model_metrics:
            metrics.append(html.Div([
                html.H4("Information Revelation"),
                html.H2(f"{self.model_metrics['info_revelation']:.2%}"),
                html.P("Percentage of insider information reflected in price")
            ], style=card_style))
        
        # Current price
        if 'price' in self.model_metrics:
            metrics.append(html.Div([
                html.H4("Model Price"),
                html.H2(f"{self.model_metrics['price']:.6f}"),
                html.P("Current price according to Kyle model")
            ], style=card_style))
        
        return metrics
    
    def update_data(self, market_data=None, model_metrics=None, model_history=None):
        """
        Update the dashboard data.
        
        Args:
            market_data (pd.DataFrame): Market data
            model_metrics (dict): Current model metrics
            model_history (pd.DataFrame): History of model updates
        """
        update = {}
        if market_data is not None:
            update['market_data'] = market_data
        if model_metrics is not None:
            update['model_metrics'] = model_metrics
        if model_history is not None:
            update['model_history'] = model_history
        
        if update:
            self.data_queue.put(update)
    
    def run_server(self, debug=False, port=8050):
        """
        Run the dashboard server.
        
        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port to run the server on
        """
        self.app.run(debug=debug, port=port)