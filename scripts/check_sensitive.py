import pandas as pd
import argparse
import os

# Get input folder to search
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder to search for sensitive data")
args = parser.parse_args()
#
terms = [""]

def check_sensitive(_terms):
# Process every feather file in the folder
for file in os.listdir(args.folder):
    if file.endswith(".feather"):
        # Open the feather file using Pandas
        df = pd.read_feather(args.folder + file, nrows=1000)

        # Perform operations on the dataframe
        cols = df.columns.values

        # Print the first few rows of the dataframe
        print(df.head())

# Print the first few rows of the dataframe
print(df.head())

if __name__ == "__main__":
    check_sensitive(terms)