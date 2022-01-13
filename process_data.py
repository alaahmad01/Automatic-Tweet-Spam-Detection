import pandas as pd

def read_file(filename):
    return pd.read_csv(filename)

def process_data(df):
    l = []
    for row_index in range(df.size):
        row = df.iloc[row_index]
