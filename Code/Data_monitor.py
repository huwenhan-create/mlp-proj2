import pandas as pd

# Missing value
data = pd.read_csv('Code\Data\hotel_pre.csv')

# target 
target = 'children'
missing = data.loc[:, target]
print("test")
