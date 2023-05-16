import dash
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import load_csv
from dash import dash_table


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# load the different csv_data 
# TODO: Right now its hardcoded, this needs to change
csv_directory = csv_directory = "/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV"

csv_filepaths = load_csv.get_csv_filepaths(csv_directory)


csv_PRM = load_csv.load_csv_data(csv_filepaths[0])
csv_RRT = load_csv.load_csv_data(csv_filepaths[1])
csv_DRL = load_csv.load_csv_data(csv_filepaths[2])



# Sample data for the graphs
planning_time = [0.2, 0.3, 0.1, 0.4, 0.15]
execution_time = [0.8, 0.7, 0.9, 0.6, 0.85]

#shaking = [0.2, 0.25, 0.3, 0.35, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
shaking_DRL = np.array([row["shaking_ur5_1"] for row in csv_DRL])
steps_DRL = list(range(1, 200))
steps_row = np.array([row[""] for row in csv_DRL])
steps_DRL = list(range(1,int(steps_row[-1])))

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
                        go.Scatter(x=steps_DRL, y=shaking_DRL, mode='lines+markers', name='DRL')
                    ],
                    'layout': go.Layout(xaxis={'title': 'Steps'}, yaxis={'title': 'Shaking'}, showlegend=True)
                }
            ),
            dcc.Dropdown(
                id='shaking-dropdown',
                options=[
                    {'label': 'DRL', 'value': 'DRL'},
                    
                ],
                value='DRL',
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
                        go.Scatter(x=steps_DRL, y=distance_to_obstacle, mode='lines+markers', name='Distance')
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Steps'},
                        yaxis={'title': 'Distance'},
                        shapes=[{
                            'type': 'line',
                            'x0': steps_DRL[0],
                            'x1': steps_DRL[-1],
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
       dbc.Col([
            html.H3('Radar Chart', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='radar-chart',
                figure={
                    'data': [
                        go.Scatterpolar(
                            r=[0.5, 0.3, 0.9, 0.5],
                            theta=['smoothness', 'collision','number of steps', 'execution time'],
                            fill='toself',
                            name='Radar Chart'
                        )
                    ],
                    'layout': go.Layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=False
                    )
                }
            ),
            dcc.Dropdown(
                id='radar-chart-dropdown',
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