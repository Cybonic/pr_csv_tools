import pandas as pd

# Define the multi-row header
header = pd.MultiIndex.from_tuples([
    ('1', '1m'), ('1', '10m'),('1', 'row'),
    ('1%', '1m'), ('1%', '10m'),('1%', 'row')
])

# Create a DataFrame with the multi-row header
df = pd.DataFrame([[1, 2, 3, 4,5,6], [7, 8, 9, 10,11,12]], columns=header)

print(df)