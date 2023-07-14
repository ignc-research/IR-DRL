# This logs the important data from the Sim2Real Evaluations

#Difference to Simulation logging, different steps, different joint velocities
#TODO: first get the information from the simulation csv, then overwrite the necessary parts, afterwards create a new csv

import pandas as pd

#global variables
df = None


def load_sim_csv(df):
    # Read CSV file into DataFrame
    df = pd.read_csv('/home/moga/Desktop/IR-DRL/models/env_logs/DRL_new_episode.csv')
    return df

   

if __name__ == '__main__':
    # Print the entire DataFrame
    df = load_sim_csv(df)
    print(df)

    # Access a specific row
    row_index = 1  # change to your desired row index
    print(df.iloc[row_index])

    # Access a specific column
    column_name = 'episodes'  # change to your desired column name
    print(df[column_name])

    # Access a specific cell (row, column)
    print(df.at[row_index, column_name])