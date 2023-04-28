import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc


#trajectory Points
#TODO: Replace with trajectores from csv
trajectory = np.array([
    [0.14964543, 0.51980054, 0.34284654],
    [0.15218042, 0.51606137, 0.34318519],
    [0.15552515, 0.52252108, 0.34451479],
    [0.16411068, 0.52975202, 0.34508801],
    [0.16643231, 0.53591859, 0.34596759],
    [0.16485478, 0.53941351, 0.3404704 ],
    [0.17190547, 0.54124993, 0.33658996],
    [0.17293172, 0.55146658, 0.33927181],
    [0.17038222, 0.55280972, 0.33981764],
    [0.17449835, 0.55258667, 0.34343454],
])


# Extract x, y, and z arrays from the trajectory
x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Create the initial figure with the 3D curve, waypoints, and the moving point
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Trajectory'))
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color='green'), name='Waypoints'))
fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=10, color='red'), name='Endeffector-Movement'))

fig.update_layout(scene=dict(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    aspectmode='auto',
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)  # Increase the values here to zoom out.
    )
))

# Initialize the Dash app with bootstrap to make the website responsive
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def update_table_data_and_highlight_active_waypoint(n_intervals):
    data = [
        {'waypoints': f'({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})', 'velocities': ''}
        for i in range(len(x))
    ]

    style_data_conditional = [
        {
            'if': {'row_index': n_intervals},
            'backgroundColor': 'rgb(255, 215, 0)',
            'color': 'black'
        }
    ]

    return data, style_data_conditional


# Define the app layout with the graph, buttons, headline, and dropdown menu
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Evaluation', style={'textAlign': 'center'}),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='graph', figure=fig, style={'height': '80vh', 'width': '100%'}),
        ], width=8, style={'padding-right': '0px'}),
        dbc.Col([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='evaluation-type',
                        options=[
                            {'label': 'Simulation', 'value': 'Simulation'},
                            {'label': 'Real', 'value': 'Real'}
                        ],
                        value='Simulation',
                        clearable=False,
                        style={'width': '100%', 'margin-bottom': '5px'}
                    ),
                    html.Div([
                        html.Button('Play', id='play-button', n_clicks=0),
                        html.Button('Pause', id='pause-button', n_clicks=0),
                        html.Button('Repeat', id='repeat-button', n_clicks=0),
                        dcc.Interval(id='interval', interval=500),  # speed in which the red point moves
                    ], style={'display': 'flex', 'justifyContent': 'center', 'margin-top': '0px'}),
                    html.Br(),
                    dcc.Dropdown(
                        id='waypoints-dropdown',
                        options=[
                            {'label': 'Hide Waypoints', 'value': 0},
                            {'label': 'Show Waypoints', 'value': 1}
                        ],
                        value=1,
                        clearable=False,
                        style={'width': '100%'}
                    ),
                    html.Br(),
                    dash_table.DataTable(
                        id='waypoints-table',
                        columns=[
                            {'name': 'Waypoints', 'id': 'waypoints'},
                            {'name': 'Velocities', 'id': 'velocities'}
                        ],
                        data=update_table_data_and_highlight_active_waypoint(0)[0],
                        style_data_conditional=update_table_data_and_highlight_active_waypoint(0)[1],
                        style_cell={'textAlign': 'center'},
                        style_header={'fontWeight': 'bold'}
                    ),
                ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'height': '80vh'})
            ], style={'padding': '0 15px'})
        ], width=4),
    ]),
    dcc.Store(id='n_intervals', data=0),
    dcc.Store(id='is_playing', data=True)
], fluid=True)


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

# Callback to update the moving point, waypoints visibility, and handle repeat button
# Callback to update the moving point, waypoints visibility, and handle repeat button
@app.callback(
    [Output('graph', 'figure'), Output('n_intervals', 'data'), Output('waypoints-table', 'data'), Output('waypoints-table', 'style_data_conditional')],
    [Input('interval', 'n_intervals'), Input('waypoints-dropdown', 'value'), Input('repeat-button', 'n_clicks')],
    [State('graph', 'figure'), State('is_playing', 'data'), State('n_intervals', 'data')]
)
def update_point_and_waypoints_visibility_and_repeat(_, dropdown_value, repeat_clicks, figure, is_playing, n_intervals):
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Handle repeat button click
        if input_id == 'repeat-button':
            n_intervals = 0
            figure['data'][2].update(x=[x[n_intervals]], y=[y[n_intervals]], z=[z[n_intervals]])

    # Update waypoints visibility
    figure['data'][1].update(visible=True if dropdown_value == 1 else False)

    # Update the moving point
    if not is_playing:
        data, style_data_conditional = update_table_data_and_highlight_active_waypoint(n_intervals)
        return figure, n_intervals, data, style_data_conditional

    if n_intervals < len(x):
        figure['data'][2].update(x=[x[n_intervals]], y=[y[n_intervals]], z=[z[n_intervals]])
        n_intervals += 1

    data, style_data_conditional = update_table_data_and_highlight_active_waypoint(n_intervals)
    return figure, n_intervals, data, style_data_conditional




if __name__ == '__main__':
    app.run_server(debug=True)