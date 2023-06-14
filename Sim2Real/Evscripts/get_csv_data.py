import pandas as pd
import re

# Specify the csv file path
csv_path = '/home/moga/Desktop/IR-DRL/Sim2Real/EvSpecial/episode_real_2.csv'

# Read the csv file
df = pd.read_csv(csv_path)

# Extract lists of numbers from the strings
df['real_joint_positions'] = df['real_joint_positions'].apply(
    lambda x: [float(num) for num in re.findall(r"[-+]?\d*\.\d+e?[+-]?\d*|[-+]?\d+", x)]
)

# Get all the values under the column "real_joint_positions"
real_joint_positions = df['real_joint_positions'].values

# Print the values
for position in real_joint_positions:
    print(str([round(num, 8) for num in position]) + ',')
