import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample data for the graphs
planning_time = [0.2, 0.3, 0.1, 0.4, 0.15]
execution_time = [0.8, 0.7, 0.9, 0.6, 0.85]

shaking = [0.2, 0.25, 0.3, 0.35, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
steps = list(range(1, 11))

distance_to_obstacle = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

# Create the dashboard layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H1('Evaluation', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
        ], width=12),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H3('Planning and Execution Time', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='planning-execution-time',
                figure={
                    'data': [
                        go.Bar(x=list(range(1, 6)), y=planning_time, name='Planning Time'),
                        go.Bar(x=list(range(1, 6)), y=execution_time, name='Execution Time')
                    ],
                    'layout': go.Layout(barmode='stack', xaxis={'title': 'Episode'}, yaxis={'title': 'Time'})
                }
            ),
            dcc.Dropdown(
                id='planning-execution-time-dropdown',
                options=[
                    {'label': 'Placeholder 1', 'value': 'Placeholder 1'},
                    {'label': 'Placeholder 2', 'value': 'Placeholder 2'}
                ],
                value='Placeholder 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
        dbc.Col([
            html.H3('Shaking', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='shaking',
                figure={
                    'data': [
                        go.Scatter(x=steps, y=shaking, mode='lines+markers', name='Shaking')
                    ],
                    'layout': go.Layout(xaxis={'title': 'Steps'}, yaxis={'title': 'Shaking'})
                }
            ),
            dcc.Dropdown(
                id='shaking-dropdown',
                options=[
                    {'label': 'Placeholder 1', 'value': 'Placeholder 1'},
                    {'label': 'Placeholder 2', 'value': 'Placeholder 2'}
                ],
                value='Placeholder 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
                dbc.Col([
            html.H3('Distance to Obstacle', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='distance-to-obstacle',
                figure={
                    'data': [
                        go.Scatter(x=steps, y=distance_to_obstacle, mode='lines+markers', name='Distance')
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Steps'},
                        yaxis={'title': 'Distance'},
                        shapes=[{
                            'type': 'line',
                            'x0': steps[0],
                            'x1': steps[-1],
                            'y0': 0.5,
                            'y1': 0.5,
                            'yref': 'y',
                            'xref': 'x',
                            'line': {'color': 'red', 'width': 1, 'dash': 'dot'}
                        }]
                    )
                }
            ),
            dcc.Dropdown(
                id='distance-to-obstacle-dropdown',
                options=[
                    {'label': 'Placeholder 1', 'value': 'Placeholder 1'},
                    {'label': 'Placeholder 2', 'value': 'Placeholder 2'}
                ],
                value='Placeholder 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=True)