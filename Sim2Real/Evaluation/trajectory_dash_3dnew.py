import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import load_csv
from gen_obstacle import generate_obstacle
from dash.exceptions import PreventUpdate
import os
from dash_color_picker import ColorPicker
from dash.dependencies import Input, Output, State, ALL



#trajectory Points
#csv_data = load_csv.load_csv_data("/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV/episode_1.csv")
#csv_data2 = load_csv.load_csv_data("/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV/episode_6.csv")


csv_directory = "/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV"


csv_filepaths = load_csv.get_csv_filepaths(csv_directory)

csv_data = load_csv.load_csv_data(csv_filepaths[0])
csv_data2 = load_csv.load_csv_data(csv_filepaths[1])

csv_options = [{'label': f"{os.path.basename(file)} (+)", 'value': file} for file in csv_filepaths]


trajectory = np.array([row["position_ee_link_ur5_1"] for row in csv_data])
trajectory2 = np.array([row["position_ee_link_ur5_1"] for row in csv_data2])



# Extract x, y, and z arrays from the trajectory
x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
a,b,c = trajectory2[:,0], trajectory2[:,1], trajectory2[:,2]

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Trajectory'))
fig.add_trace(go.Scatter3d(x=a, y=b, z=c, mode='lines', name='Trajectory2'))

fig.update_layout(scene=dict(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    aspectmode='auto',
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)  # Increase the values here to zoom out.
    )
))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Evaluation', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
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
                        id='csv-dropdown',
                        options=csv_options,
                        value=[csv_filepaths[0]],  # Set the first CSV file as the default selected value
                        multi=True,  # Allow multiple selections
                        style={'width': '100%', 'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}
                    ),
                    dcc.Dropdown(
                        id='waypoints-dropdown',
                        options=[
                            {'label': 'Hide Waypoints', 'value': 0},
                            {'label': 'Show Waypoints', 'value': 1}
                        ],
                        value=0,
                        clearable=False,
                        style={'width': '100%', 'font-family': 'Arial, sans-serif', 'margin-bottom': '10px'}
                    ),
                ]),
            ], style={'flex': '0 0 auto'}),
            html.Div(
                [
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label(f"{os.path.basename(file)}", style={'font-family': 'Arial, sans-serif'}), width=6),
                           dbc.Col(html.Div(
                                ColorPicker(
                                    id={'type': 'color-picker', 'index': i},
                                    color='red',
                                ),
                                style={'position': 'relative', 'z-index': '1000'}
                            ), width=6),
                        ]),
                        html.Button("Toggle Table", id={'type': 'toggle-table-btn', 'index': i}, n_clicks=0),
                        dbc.Collapse(
                            dash_table.DataTable(
                                id={'type': 'waypoints-table', 'index': i},
                                columns = [
                                    {'name': 'Waypoints', 'id' : 'waypoints'},
                                    {'name' : 'Velocities', 'id' : 'velocities'}
                                ],
                                data=[
                                {
                                    'waypoints': f'({row["position_ee_link_ur5_1"][0]:.2f}, {row["position_ee_link_ur5_1"][1]:.2f}, {row["position_ee_link_ur5_1"][2]:.2f})',
                                    'velocities': f'({row["velocity_ee_link_ur5_1"][0]:.2f}, {row["velocity_ee_link_ur5_1"][1]:.2f}, {row["velocity_ee_link_ur5_1"][2]:.2f})'
                                } for row in load_csv.load_csv_data(file)
                                ],

                                fixed_rows={'headers': True, 'data': 0},
                                style_table={'overflowY': 'auto', 'maxWidth': '100%'},
                                style_data={
                                    'textAlign': 'center',
                                },
                            ),
                            id={'type': 'table-collapse', 'index': i},
                            is_open=False,
                        ),
                    ], style={'margin-bottom': '10px', 'border': '1px solid', 'padding': '5px'}) for i, file in enumerate(csv_filepaths)
                ],
                style={'max-height': '65vh', 'overflow-y': 'auto', 'flex': '1 1 auto'}
            )
        ], width=4, style={'padding': '0 15px', 'display': 'flex', 'flex-direction': 'column', 'height': '80vh'})
        ]),
], fluid=True)


# Add this callback below the app.layout
"""@app.callback(
    Output('graph', 'figure'),
    [
        Input('csv-dropdown', 'value'),
        Input('waypoints-dropdown', 'value'),
    ],
    [
        State({'type': 'color-picker', 'index': ALL}, 'color'),
    ],
)
"""

@app.callback(
    Output('graph', 'figure'),
    [
        Input('csv-dropdown', 'value'),
        Input('waypoints-dropdown', 'value'),
    ],
    [
        State({'type': 'color-picker', 'index': ALL}, 'color'),
    ],
)

def update_trajectories(selected_csv_files, show_waypoints, selected_colors):

    if not selected_csv_files:
        raise PreventUpdate

    fig = go.Figure()

    for i, (file, color) in enumerate(zip(selected_csv_files, selected_colors)):
            csv_data = load_csv.load_csv_data(file)
            trajectory = np.array([row["position_ee_link_ur5_1"] for row in csv_data])
            x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=f'Trajectory {i+1}', line=dict(color=selected_colors[i])))

            if show_waypoints:
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=f'Waypoints {i+1}', marker=dict(size=4, color=selected_colors[i])))

    fig.update_layout(scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode='auto',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)  # Increase the values here to zoom out.
        )
    ))

    return fig

#seperate callback only for the table to be able to hide/show it
for i in range(len(csv_filepaths)):
    @app.callback(
        Output({'type': 'table-collapse', 'index': i}, 'is_open'),
        Input({'type': 'toggle-table-btn', 'index': i}, 'n_clicks'),
        State({'type': 'table-collapse', 'index': i}, 'is_open')
    )
    def toggle_table(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open



if __name__ == '__main__':
    app.run_server(debug=True)