import pandas as pd

folder = './data'
# Read the CSV file
df = pd.read_csv(f'{folder}/HGMD_pubmed.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Find the locations of missing values
missing_locations = df[df.isnull().any(axis=1)].index

print("Missing values:")
print(missing_values)
print("Locations of missing values:")
print(missing_locations)
