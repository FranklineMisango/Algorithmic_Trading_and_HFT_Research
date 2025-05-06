from dash import Dash, dcc, html
import plotly.graph_objs as go
from src.models.gm_model_live import GlostenMilgromModelLive
from src.data.data_processor import MarketDataProcessor

class LiveDashboard:
    def __init__(self):
        self.app = Dash(__name__)
        self.data_processor = MarketDataProcessor()
        self.model = GlostenMilgromModelLive()
        
        self.app.layout = html.Div([
            html.H1("Glosten-Milgrom Live Dashboard"),
            dcc.Graph(id='live-graph'),
            dcc.Interval(
                id='graph-update',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])
        
        self.app.callback(
            dash.dependencies.Output('live-graph', 'figure'),
            [dash.dependencies.Input('graph-update', 'n_intervals')]
        )(self.update_graph)

    def update_graph(self, n):
        live_data = self.data_processor.get_latest_data()
        self.model.update_model(live_data)
        
        spreads = self.model.calculate_spreads()
        
        figure = {
            'data': [
                go.Scatter(
                    x=list(range(len(spreads))),
                    y=spreads,
                    mode='lines+markers'
                )
            ],
            'layout': go.Layout(
                title='Live Bid-Ask Spreads',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Spread'),
                showlegend=True
            )
        }
        return figure

    def run(self):
        self.app.run_server(debug=True)