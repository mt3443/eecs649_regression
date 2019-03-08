import pandas as pd
import numpy as np

# seed RNG for reproducable results
np.random.seed(1)

# get original data set
df = pd.read_csv('creditcard.csv')

#drop fraudulent transactions
df = df.drop(df[df.Class == 1].index)

# split original data set into training and testing data sets
mask = np.random.rand(len(df)) < 0.7
train = df[mask]
test = df[~mask]

# save to new csv files
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)