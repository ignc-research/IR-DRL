import pybullet as p
import pybullet_data
import time
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#get data from csv
# Specify the csv file path
csv_path = '/home/moga/Desktop/IR-DRL/models/env_logs/episode_real_0.csv'

# Read the csv file
df = pd.read_csv(csv_path)

# Extract lists of numbers from the strings
df['real_joint_positions'] = df['real_joint_positions'].apply(
    lambda x: [float(num) for num in re.findall(r"[-+]?\d*\.\d+e?[+-]?\d*|[-+]?\d+", x)]
)

# Get all the values under the column "real_joint_positions"
real_joint_positions = df['real_joint_positions'].values

# Print the values
#for position in real_joint_positions:
 #   print(str([round(num, 8) for num in position]) + ',')


# Start the simulation
client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used to locate .urdf and other model files

# Load the UR5 model
robot = p.loadURDF("/home/moga/Desktop/IR-DRL/modular_drl_env/assets/robots/predefined/ur5/urdf/ur5.urdf",baseOrientation=p.getQuaternionFromEuler([0,0,-np.pi]),basePosition=[0,0,0.01])

# Set the gravity
p.setGravity(0, 0, -9.8, physicsClientId=client)

# Define joint angles
joint_angles = real_joint_positions
# Initialize an array to store end effector positions
end_effector_positions = []

joints = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
joint_ids = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

for angles in joint_angles:
    
    for idx,i in enumerate(joint_ids):
        p.resetJointState(robot, i, angles[idx])
   
    
    # Calculate the end effector position
    ls = p.getLinkState(robot, 7)  # Link 7 is usually the end effector
    end_effector_positions.append(ls[4])  # ls[4] contains the Cartesian coordinates
    
    # Add some delay for visual purposes
    time.sleep(0.05)

# Print the end effector positions
for i, pos in enumerate(end_effector_positions):
    print(f"End effector position for joint angles set {i+1}: {pos}")
#TODO: add end effector position to csv 

# End the simulation
p.disconnect()


# Your pybullet script
# ... your code above

# Convert the tuples in end_effector_positions to lists (vectors) and then to strings
end_effector_positions_str = ['[' + ' '.join(map(str, pos)) + ']' for pos in end_effector_positions]
# Convert end_effector_positions_str list to a DataFrame
df_end_effector = pd.DataFrame(end_effector_positions_str, columns=['real_ee_position'])

# Load the existing CSV data
df = pd.read_csv("/home/moga/Desktop/IR-DRL/models/env_logs/episode_real_0.csv")

# Check if the size of df_end_effector is the same as df. If not, there might be some issue.
if len(df) != len(df_end_effector):
    print("Warning: the size of the end effector positions data and the original DataFrame is not equal!")

# Concatenate the end_effector_positions to the right of df (i.e., column-wise)
df = pd.concat([df, df_end_effector], axis=1)

# Save the DataFrame to CSV, overwrite the original file
df.to_csv("/home/moga/Desktop/IR-DRL/models/env_logs/episode_real_0.csv", index=False)




def plot_3d_points(points):
    # Extract x, y, and z coordinates from the points list
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as scatter plot
    ax.scatter(x, y, z, c='r', marker='o')

    # Connect the points with lines
    ax.plot(x, y, z, c='b')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()

"""# Example usage
points = [
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15]"""
#]

plot_3d_points(end_effector_positions)