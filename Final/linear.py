import numpy as np
import pandas as pd

df = pd.read_csv("train_not.csv")
df.drop('Id',axis = 1)

X_train = df.loc[:,df.columns!='SalePrice']
Y_train = df['SalePrice']

X_train = X_train.values
Y_train = Y_train.values

initial_W = 