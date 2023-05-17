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

    #calculate average
    summe = [None for _ in range(10)]
    for i in range(10):
        summe[i] = calc_average(distance[i])
   
    distance.append(summe)

    

    return distance



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


smoothness = np.array([row["shaking_ur5_1"]for row in csv_DRL])
smoothness_avg = calc_average(smoothness)
DRL_array = [
    [0.3, 0.5, 0.2, 0.8,0.3],
    [0.4, 0.6, 0.3, 0.9,0.3],
    [0.5, 0.7, 0.4, 0.8,0.3],
    [0.6, 0.7, 0.4, 0.7,0.3],
    [0.7, 0.8, 0.3, 0.6,0.3],
    [0.4, 0.5, 0.6, 0.9,0.3],
    [0.8, 0.9, 0.2, 0.5,0.3],
    [0.4, 0.6, 0.3, 0.9,0.3],
    [0.4, 0.6, 0.3, 0.9,0.3],
    [0.5, 0.7, 0.4, 0.8,0.3],
    #average  
    [0.005,0.0048, 0.08, 0.011,smoothness_avg]
]

['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time']

#smoothness
smoothness = np.array([row["shaking_ur5_1"]for row in csv_DRL])
smoothness_avg = calc_average(smoothness)


def generate_random_array():
    return [[random.random() for _ in range(4)] for _ in range(10)]

RRT_array = [[0.2868269537609386, 0.9170617314598207, 0.4475316863179496, 0.05608282983976365,0.0],
              [0.8261362559755113, 0.6494670519862822, 0.5219236139903745, 0.8092907171845747,0.0],
                [0.1517715598021, 0.7982994063979079, 0.2588579887699962, 0.257523025834015,0.0], 
                [0.7258762640108746, 0.8589356111987257, 0.44327004115840163, 0.09330232540890204,0.0], 
                [0.42529887525026244, 0.7455527683364909, 0.21583761513137822, 0.3954040111190482,0.0], 
                [0.47241874357988023, 0.4221197620209567, 0.39338546332397883, 0.04899298208483327,0.0], 
                [0.31335910668148503, 0.9833606581498897, 0.8762276871042998, 0.8178781085758806,0.0], 
                [0.7221653562887994, 0.5898657780886235, 0.972709625876824, 0.7118000290083163,0.0], 
                [0.683420204628211, 0.472130235002039, 0.06872851667161417, 0.1819996148690467,0.0], 
                [0.5813712222698265, 0.03389949124416658, 0.9293615182962907, 0.8021025187297883,0.0]]
RRT_array.append([0.09,0.15,0.16,0.23,0.0])
#PRM_array = generate_random_array()
PRM_array = [[0.18021308541192116, 0.32007261488446326, 0.43937810850737236, 0.3956950700645051,0.0],
              [0.6675917234788222, 0.6826835413580034, 0.4016436341273857, 0.023519086512638787,0.0],
                [0.6529656973098361, 0.5074225318967271, 0.47749705457905467, 0.6585648892761593,0.0],
                  [0.8250793543651704, 0.3949361902026256, 0.9642532065576624, 0.9714896485368797,0.0],
                    [0.14530679650451472, 0.09119218446359945, 0.49675023263945717, 0.7102197620355362,0.0], 
                    [0.9497329673598982, 0.7744059979527139, 0.6568029112651473, 0.15803604633085,0.0], 
                    [0.1820749744488629, 0.8861033857680668, 0.7345654722681388, 0.5794333259091891,0.0], 
                    [0.6874547590605017, 0.4485030607848294, 0.9311645603299458, 0.2909345699171386,0.0], 
                    [0.14731126972064335, 0.5958732224869299, 0.35560626710406473, 0.2934303097227393,0.0], 
                    [0.26456921699558944, 0.02965112321940766, 0.9246860527852515, 0.5939271375758527,0.0]]

PRM_array.append([0.28,0.18,0.34,0.5,0.0])
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
                value='Episode 11',
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
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time','roughness index'],
                fill='toself',
                name='DRL'
            ),
            go.Scatterpolar(
                r=RRT_array[0],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time','roughness index'],
                fill='toself',
                name='RRT'
            ),
            go.Scatterpolar(
                r=PRM_array[0],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time','roughness index'],
                fill='toself',
                name='PRM',
                marker_color='#8E6BB4'
            )
        ],
        'layout': go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[1, 0])
            ),
            showlegend=True
        )
    }
),
dcc.Dropdown(
    id='radar-chart-dropdown',
    options=[{'label': f'Episode {i}', 'value': f'Episode {i}'} for i in range(1, 11)] + [{'label': 'Average', 'value': 'Episode 11'}],
    value='Episode 11',
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
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time','roughness index'],
                fill='toself',
                name='DRL'
            ),
            go.Scatterpolar(
                r=RRT_array[episode_index],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time', 'roughness index'],
                fill='toself',
                name='RRT'
            ),
            go.Scatterpolar(
                r=PRM_array[episode_index],
                theta=['computation Time', 'distance_to_obstacle', 'number of steps', 'execution time', 'roughness index'],
                fill='toself',
                name='PRM',
                marker_color='#8E6BB4'
            )
        ],
        'layout': go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[1, 0])
            ),
            showlegend=True
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)