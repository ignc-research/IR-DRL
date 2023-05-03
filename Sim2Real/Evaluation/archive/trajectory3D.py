# Import necessary libraries
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Generate an array of 100 evenly spaced values between 0 and 20
t = np.linspace(0, 20, 100)

# Define the 3D curve using the parameter t
x = np.sin(t)
y = np.cos(t)
z = t

# Create a subplot with a 3D scatter plot
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Create a 3D line plot for the curve using x, y, and z coordinates
scatter3d = go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Curve')

# Add the curve to the subplot
fig.add_trace(scatter3d)

# Create a moving point on the curve with initial position at the beginning of the curve
moving_point = go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=10, color='red'), name='Moving Point')

# Add the moving point to the subplot
fig.add_trace(moving_point)

# Set the axis labels for the 3D plot
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)
pause_settings = dict(frame=dict(duration=1e10, redraw=True), fromcurrent=True)  # Extremely high duration to pretend there is a "pause"


# Define a function to update the moving point's position based on the index in the t array
def update_point_position(t_idx):
    fig.data[1].x = [x[t_idx]]
    fig.data[1].y = [y[t_idx]]
    fig.data[1].z = [z[t_idx]]
    fig.update()

# Set up the animation settings for the moving point
animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)

# Add a "Play" button to start the animation
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, animation_settings])])])

# Create a list of frames for the animation, updating the moving point's position for each frame
frames = [go.Frame(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines'),
                          go.Scatter3d(x=[x[k]], y=[y[k]], z=[z[k]], mode='markers', marker=dict(size=10, color='red'))])
          for k in range(1, len(t))]

# Assign the frames to the figure
fig.frames = frames
fig.write_html('traj3d.html',auto_open=False)
