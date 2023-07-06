import pandas as pd
import csv
import os

def load_csv_data(file_path, columns=None):
    def string_to_list(string):
        return [float(x) for x in string.strip('[]').split()]

    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_data = {}
            for column in row:
                if column != 'obstacles':
                    if columns is None or column in columns:
                        if '[' in row[column] and ']' in row[column]:
                            row_data[column] = string_to_list(row[column])
                        else:
                            try:
                                row_data[column] = float(row[column])
                            except ValueError:
                                row_data[column] = row[column]
                data.append(row_data)
    return data



def get_csv_filepaths(directory):
    filepaths = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            filepaths.append(filepath)
    return filepaths