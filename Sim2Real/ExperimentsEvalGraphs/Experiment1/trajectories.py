import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
#from gen_obstacle import generate_obstacle
from dash.exceptions import PreventUpdate
import os
import load_csv
#from dash_color_picker import ColorPicker
from dash.dependencies import Input, Output, State, ALL



#trajectory Points
#csv_data = load_csv.load_csv_data("/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV/episode_1.csv")
#csv_data2 = load_csv.load_csv_data("/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV/episode_6.csv")

#TODO: evtl auch in der sim nur die tatsächlichen steps nehmen

csv_real = load_csv.load_csv_data("Sim2Real/EvSpecial/episode_real_2.csv")
csv_sim = load_csv.load_csv_data("Sim2Real/EvSpecial/episode_simulated_2.csv")
#csv_data2 = load_csv.load_csv_data(csv_filepaths[1])

trajectory_sim = np.array([row["position_ee_link_ur5_1"] for row in csv_sim])
trajectory_real = np.array([row["real_ee_position"]for row in csv_real])


x_real, y_real, z_real= trajectory_real[:, 0], trajectory_real[:, 1], trajectory_real[:, 2]
x_sim, y_sim, z_sim = trajectory_sim[:, 0], trajectory_sim[:, 1], trajectory_sim[:, 2]

# Create the initial figure with the 3D curve, waypoints, and the moving point
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x_real, y=y_real, z=z_real, mode='lines', name='Trajectory'))
fig.add_trace(go.Scatter3d(x=x_real, y=y_real, z=z_real, mode='markers', marker=dict(size=4, color='blue'), name='Waypoints'))

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


# Define the app layout with the graph, buttons, headline, and dropdown menu
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Sim and Real Comparison', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='graph', figure=fig, style={'height': '80vh', 'width': '100%'}),
        ], width=8, style={'padding-right': '0px'}),
    ]),
],fluid=True)





if __name__ == '__main__':
    app.run_server(debug=True)