import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly.io import write_html



# Generate the parameter t and define the 3D curve
t = np.linspace(0, 20, 100)
x = np.sin(t)
y = np.cos(t)
z = t

# Create the initial figure
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Curve'))
fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=10, color='red'), name='Moving Point'))
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Graph(id='graph', figure=fig, style={'height': '80vh', 'width': '80vw'}),
    html.Button('Play', id='play-button', n_clicks=0),
    html.Button('Pause', id='pause-button', n_clicks=0),
    dcc.Interval(id='interval', interval=100),
    dcc.Store(id='n_intervals', data=0),
    dcc.Store(id='is_playing', data=True)
])

# Callback to toggle play/pause
@app.callback(
    Output('is_playing', 'data'),
    [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')],
    [State('is_playing', 'data')]
)
def toggle_play_pause(play_clicks, pause_clicks, is_playing):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_playing
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'play-button':
        return True
    elif button_id == 'pause-button':
        return False

# Callback to update the moving point
@app.callback(
    [Output('graph', 'figure'), Output('n_intervals', 'data')],
    [Input('interval', 'n_intervals')],
    [State('graph', 'figure'), State('is_playing', 'data'), State('n_intervals', 'data')]
)
def update_point(_, figure, is_playing, n_intervals):
    if not is_playing:
        return figure, n_intervals

    if n_intervals < len(t):
        figure['data'][1].update(x=[x[n_intervals]], y=[y[n_intervals]], z=[z[n_intervals]])
        n_intervals += 1
    return figure, n_intervals

if __name__ == '__main__':
    app.run_server(debug=True)