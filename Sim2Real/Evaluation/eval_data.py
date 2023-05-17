import dash
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import load_csv
from dash import dash_table
from dash.dependencies import Input,Output

import random


#sim time = execution time

#TODO:
#Start und Goal markieren
#Computation time statt planning time, computation time farbe des planners
#shaking 
#gucken ob die werte für distance to obstacle stimmen 

#TODO: 
# Average per episode ()
# tabular farben statt basic farben ()
# Farben anpassen ()
# Goal und Ziel markieren () 
# Aufnahme ()


#DRL = Green,
#RRT = orange, 
#PRM = light purple
colors = ['#4E79A7', '#F28E2C', '#8E6BB4']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# load the different csv_data 
# TODO: Right now its hardcoded, this needs to change
csv_directory = csv_directory = "/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV"

csv_filepaths = load_csv.get_csv_filepaths(csv_directory)

csv_DRL = load_csv.load_csv_data(csv_filepaths[0])
csv_RRT = load_csv.load_csv_data(csv_filepaths[1])
csv_PRM = load_csv.load_csv_data(csv_filepaths[2])

#ignores last index of array as this is the one where the average is written down
def calc_average(array):
    if len(array) <= 1:
        return None

    sum_of_values = sum(array[:-1])
    count_of_values = len(array) - 1
    average = sum_of_values / count_of_values
    return average



def count_number_of_episodes(csv_data):
     # Array erstellen, in dem für jede indexstelle die anzahl der jeweiligen Zahlen i+1 gespeichert wird
    episodes = np.array([row["episodes"]for row in csv_data])
    max_number = int(max(episodes))
    number_array = [0] * max_number

    for episode in episodes: 
         number_array[int(episode) -1] += 1

    return number_array

def planning_execution_average(csv_data,mode):
    #Hole die row aus der jeweiligen csv data und berechne die Average time für jede episode
    #[26, 31, 21, 33, 22, 21, 33, 23, 21, 22]
    
    num_episodes = count_number_of_episodes(csv_data)
    
    sim_time = np.array([row["sim_time"]for row in csv_data])
    cpu_time_steps = np.array([row["cpu_time_steps"]for row in csv_data])
    cpu_time_full = np.array([row["cpu_time_full"]for row in csv_data])
    

    lower_bound = [None for _ in range(len(num_episodes))]
    upper_bound = [None for _ in range(len(num_episodes))]
    lower_bound[0] = 0
    upper_bound[0] = num_episodes[0] -1
    for i in range(1,len(num_episodes)):
        lower_bound[i] = upper_bound[i-1]
        upper_bound[i] = upper_bound[i-1] + num_episodes[i]
       
    
    computation_time_per_episode = [None for _ in range(len(num_episodes)+1)]
    exec_time_per_episode = [None for _ in range(len(num_episodes)+1)]
    # DRL
    if (mode == 1):
        #letzter Eintrag letzter eintrag cputime_steps
        computation_time_per_episode[0] = cpu_time_steps[upper_bound[0]]
        for i in range(len(num_episodes)):
            computation_time_per_episode[i] = cpu_time_steps[upper_bound[i]]
            exec_time_per_episode[i] = sim_time[upper_bound[i]]
    #RRT and PRM
    if (mode == 2):
        #Planner Time
        planner_time = np.array([row["planner_time_ur5_1"]for row in csv_data])
        for i in range(len(num_episodes)):
            computation_time_per_episode[i] = planner_time[upper_bound[i]]
            exec_time_per_episode[i] = sim_time[upper_bound[i]]

    # averages berechnen
    computation_time_per_episode[-1] = calc_average(computation_time_per_episode)
    exec_time_per_episode[-1] = calc_average(exec_time_per_episode)

    return computation_time_per_episode, exec_time_per_episode

def set_bounds(csv_data):
    num_episodes = count_number_of_episodes(csv_data)
    lower_bound = [None for _ in range(len(num_episodes))]
    upper_bound = [None for _ in range(len(num_episodes))]
    lower_bound[0] = 0
    upper_bound[0] = num_episodes[0] -1
    for i in range(1,len(num_episodes)):
        lower_bound[i] = upper_bound[i-1]
        upper_bound[i] = upper_bound[i-1] + num_episodes[i]
    
    return lower_bound, upper_bound

def distance_to_obstacles(csv_data):
    num_episodes = count_number_of_episodes(csv_data)
    lower_bound,upper_bound = set_bounds(csv_data)
    
    distance_row = np.array([row["ur5_1_closestObstDistance_robot"]for row in csv_data])
    distance = [[] for _ in range(len(num_episodes))]


    for i in range(len(num_episodes)):
        for j in range(lower_bound[i], upper_bound[i]):
            distance[i].append(distance_row[j])

    return distance

DRL_array = [
    [0.3, 0.5, 0.2, 0.8],
    [0.4, 0.6, 0.3, 0.9],
    [0.5, 0.7, 0.4, 0.8],
    [0.6, 0.7, 0.4, 0.7],
    [0.7, 0.8, 0.3, 0.6],
    [0.4, 0.5, 0.6, 0.9],
    [0.8, 0.9, 0.2, 0.5],
    [0.4, 0.6, 0.3, 0.9],
    [0.4, 0.6, 0.3, 0.9],
    [0.5, 0.7, 0.4, 0.8]  
]

def generate_random_array():
    return [[random.random() for _ in range(4)] for _ in range(10)]

RRT_array = generate_random_array()
PRM_array = generate_random_array()

plan_DRL, exec_DRL = planning_execution_average(csv_DRL,1)
plan_RRT, exec_RRT = planning_execution_average(csv_RRT,2)
plan_PRM, exec_PRM = planning_execution_average(csv_PRM,2)

# Planning execution time

planning = [plan_DRL[0], plan_RRT[0], plan_PRM[0]]
execution = [exec_DRL[0], exec_RRT[0], exec_PRM[0]]



#Number of Steps 
steps_DRL = count_number_of_episodes(csv_DRL)
steps_DRL.append(0.0)
steps_DRL[-1] = calc_average(steps_DRL)

steps_RRT = count_number_of_episodes(csv_RRT)
steps_RRT.append(0.0)
steps_RRT[-1] = calc_average(steps_RRT)

steps_PRM = count_number_of_episodes(csv_PRM)
steps_PRM.append(0.0)
steps_PRM[-1] = calc_average(steps_PRM)


#distance_to_obstacle
#distance_to_obstacle = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
distance_obst_DRL = distance_to_obstacles(csv_DRL)
distance_obst_RRT = distance_to_obstacles(csv_RRT)
distance_obst_PRM = distance_to_obstacles(csv_PRM)


#radar array
#radar_DRL = fill_radar_array(csv_DRL,1)
#radar_RRT = fill_radar_array (csv_RRT,2)
#radar_PRM = fill_radar_array (csv_PRM,2)

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
                        go.Bar(x=['DRL'], y=[planning[0]], name='Computation Time DRL', marker_color=colors[0]),
                        go.Bar(x=['RRT'], y=[planning[1]], name='Computation Time RRT', marker_color=colors[1]),
                        go.Bar(x=['PRM'], y=[planning[2]], name='Computation Time PRM', marker_color=colors[2]),
                        go.Bar(x=['DRL', 'RRT', 'PRM'], y=execution, name='Execution Time',marker_color='#5CB85C')
                    ],
                    'layout': go.Layout(barmode='stack', xaxis={'title': ''}, yaxis={'title': 'Time'})
                }
            ),
            dcc.Dropdown(
                id='planning-execution-time-dropdown',
                options=[{'label': f'Episode {i}', 'value': f'Episode {i}'} for i in range(1, 11)] + [{'label': 'Average', 'value': 'Episode 11'}],
                value='Episode 11',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
        dbc.Col([
           html.H3('Number of Steps', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='number-of-steps',
                figure={
                    'data': [
                        go.Bar(x=['DRL'], y=[steps_DRL[0]], name='DRL Number of Steps', marker_color=colors[0]),
                        go.Bar(x=['RRT'], y=[steps_RRT[1]], name='RRT Number of Steps', marker_color=colors[1]),
                        go.Bar(x=['PRM'], y=[steps_PRM[2]], name='PRM Number of Steps', marker_color=colors[2]),
                    ],
                    'layout': go.Layout(xaxis={'title': ''}, yaxis={'title': 'Number of Steps'})
                }
            ),
            dcc.Dropdown(
                id='number-of-steps-dropdown',
                options=[{'label': f'Episode {i}', 'value': f'Episode {i}'} for i in range(1, 11)] + [{'label': 'Average', 'value': 'Episode 11'}],
                value='Episode 11',
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
                        go.Scatter(x=list(range(1, 10000)), y=distance_obst_DRL[0], mode='lines+markers', name='DRL Distance', marker_color=colors[0]),
                        go.Scatter(x=list(range(1, 10000)), y=distance_obst_RRT[0], mode='lines+markers', name='RRT Distance',marker_color=colors[1]),
                        go.Scatter(x=list(range(1, 10000)), y=distance_obst_PRM[0], mode='lines+markers', name='PRM Distance',marker_color=colors[2])
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Steps'},
                        yaxis={'title': 'Distance'}
                    )
                }
            ),
            dcc.Dropdown(
                id='distance-to-obstacle-dropdown',
                options=[{'label': f'Episode {i}', 'value': f'Episode {i}'} for i in range(1, 11)] + [{'label': 'Average', 'value': 'Episode 11'}],
                value='Episode 1',
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
                r=DRL_array[0],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='DRL'
            ),
            go.Scatterpolar(
                r=RRT_array[0],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='RRT'
            ),
            go.Scatterpolar(
                r=PRM_array[0],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='PRM'
            )
        ],
        'layout': go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True
        )
    }
),
dcc.Dropdown(
    id='radar-chart-dropdown',
    options=[{'label': f'Episode {i}', 'value': f'Episode {i}'} for i in range(1, 11)] + [{'label': 'Average', 'value': 'Episode 11'}],
    value='Episode 1',
    clearable=False,
    style={'width': '100%', 'font-family': 'Arial, sans-serif'}
),
        ], width=4),
        
    ]),
])

#Graph 1: Planning and execution time
@app.callback(
    Output('planning-execution-time', 'figure'),
    [Input('planning-execution-time-dropdown', 'value')]
)
def update_planning_execution_chart(episode):
    episode_index = int(episode.split(" ")[-1]) - 1

    planning_values = [plan_DRL[episode_index], plan_RRT[episode_index], plan_PRM[episode_index]]
    execution_values = [exec_DRL[episode_index], exec_RRT[episode_index], exec_PRM[episode_index]]

    return {
        'data': [
            go.Bar(x=['DRL'], y=[planning_values[0]], name='Computation Time DRL', marker_color=colors[0]),
            go.Bar(x=['RRT'], y=[planning_values[1]], name='Computation Time RRT', marker_color=colors[1]),
            go.Bar(x=['PRM'], y=[planning_values[2]], name='Computation Time PRM', marker_color=colors[2]),
            go.Bar(x=['DRL', 'RRT', 'PRM'], y=execution_values, name='Execution Time',marker_color='#5CB85C')
        ],
        'layout': go.Layout(barmode='stack', xaxis={'title': ''}, yaxis={'title': 'Time'})
    }

#Graph 2: Number of steps
@app.callback(
    Output('number-of-steps', 'figure'),
    [Input('number-of-steps-dropdown', 'value')]
)
def update_number_of_steps_chart(episode):
    episode_index = int(episode.split(" ")[-1]) - 1

    steps_values = [steps_DRL[episode_index], steps_RRT[episode_index], steps_PRM[episode_index]]

    return {
        'data': [
                go.Bar(x=['DRL'], y=[steps_values[0]], name='DRL Number of Steps', marker_color=colors[0]),
                go.Bar(x=['RRT'], y=[steps_values[1]], name='RRT Number of Steps', marker_color=colors[1]),
                go.Bar(x=['PRM'], y=[steps_values[2]], name='PRM Number of Steps', marker_color=colors[2]),
        ],
        'layout': go.Layout(xaxis={'title': ''}, yaxis={'title': 'Number of Steps'})
    }

#Graph 3: Distance to obstacles
@app.callback(
    Output('distance-to-obstacle', 'figure'),
    [Input('distance-to-obstacle-dropdown', 'value')]
)
def update_distance_to_obstacle_chart(episode):
    episode_index = int(episode.split(" ")[-1]) - 1

    return {
        'data': [
            go.Scatter(x=list(range(1, 10000)), y=distance_obst_DRL[episode_index], mode='lines+markers', name='DRL Distance', marker_color=colors[0]),
            go.Scatter(x=list(range(1, 10000)), y=distance_obst_RRT[episode_index], mode='lines+markers', name='RRT Distance', marker_color=colors[1]),
            go.Scatter(x=list(range(1, 10000)), y=distance_obst_PRM[episode_index], mode='lines+markers', name='PRM Distance', marker_color=colors[2])
        ],
        'layout': go.Layout(
            xaxis={'title': 'Steps'},
            yaxis={'title': 'Distance'}
        )
    }


@app.callback(
    Output('radar-chart', 'figure'),
    [Input('radar-chart-dropdown', 'value')]
)
def update_radar_chart(episode):
    episode_index = int(episode.split(" ")[-1]) - 1

    return {
        'data': [
            go.Scatterpolar(
                r=DRL_array[episode_index],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='DRL'
            ),
            go.Scatterpolar(
                r=RRT_array[episode_index],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='RRT'
            ),
            go.Scatterpolar(
                r=PRM_array[episode_index],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time'],
                fill='toself',
                name='PRM'
            )
        ],
        'layout': go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)