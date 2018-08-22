import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.linear_model import LinearRegression
import missingno as msno

def load_data():
	""" Creating the DataFrames """
	df = pd.read_csv("train.csv")
	df_test = pd.read_csv("test.csv")
	df = pd.DataFrame(df)
	df_test = pd.DataFrame(df_test)
	return df,df_test

def see_null(df):
	"""Nullity check """
	if(df.isna().any().sum()):
		print("Yes you have NULL values")

def drop_columns(df,drop_cols):
	""" Cols to be Dropped """
	# drop_cols is a list
	df = df.drop(drop_cols)
	return df

def string_to_val(df,list_cols):
	""" Converts string to val """
	nan_col =[]
	for col in list_cols:
		check_nan_col = df[col].isnull().sum()

		if (check_nan_col) == 1:
			nan_col.append(col)
		
		loop_counter = 0
		unique_str = df[col].unique()
		for str_val in unique_str:
			df[col] = df[col].replace(str_val,loop_counter)
			loop_counter+=1

	return df,nan_col



def str_null(df,nan_col):
	""" NULL values not converted to vals """
	for col in nan_col:
		loop_counter = 0
		unique_str = df[col].unique()
		for str_val in unique_str:
			if pd.isnull(str_val):
				continue
			df[col] = df[col].replace(str_val,loop_counter)
			loop_counter+=1

	return df


def fill_null(df):
	""" Fill NUll values """
	df = df.fillna(df.mean())
	return df

def plot_graph(df,train_id,s):
	""" Plots graph """
	plt.scatter(train_id,df[s])
	plt.xlabel('Id')
	plt.ylabel(s)
	plt.title('Training Set')
	plt.show()


def convert_to_matrix(df,test=False):
	""" X_train and X_test """
	# Creating the Training and Test Set
	X = df.loc[:,df.columns!='SalePrice'] #Locates and Allocate all cols except last one
	
	# Convert to Numpy Array
	X = X.values 
	
	if(test == False):
		Y = df['SalePrice']
		Y = Y.values
		return X,Y
	else:
		return X

	
def Model(X_train,Y_train,X_test):
	""" Model """
	lm = LinearRegression()
	model = lm.fit(X_train,Y_train)
	Y_test = model.predict(X_test)
	Y_test = Y_test.flatten()
	return Y_test


def convert_to_csv(df,Y_test,test = False):
	"""Stores in CSV """

	
	if(test == False):
		train = df.to_csv('train_mod.csv',index = False)
	else:
		passenger_id = df['Id']
		passenger_id = passenger_id.values
		df.drop('Id',axis=1)
		test = pd.DataFrame({'Id':passenger_id,'SalePrice':Y_test})
		test.to_csv('test_mod.csv',index = False)
## sid work
def line(x , credit):
	line = 50 
	credit = credit - line 
	return h_train[x].plot(kind = 'line')

def hist(x , credit):
	hist = 50 
	credit = credit - hist 
	return h_train[x].plot(kind = 'hist')

def bar(x , credit):
	bar = 50 
	credit = credit - bar 
	return h_train[x].plot(kind = 'bar')

 # for missing number analysis

 def matrix(x , credit):
 	matrix = 100
 	credit = credit - matrix
 	return msno.matrix(h_train.sample(250))

 def heatmap(x , credit):
 	heatmap = 100
 	credit = credit - heatmap
 	return msno.heatmap(h_train.sample(250))	

 def dendrogram(x , credit):
 	dendrogram = 100
 	credit = credit - dendrogram
 	return msno.dendrogram(h_train.sample(250))


# using matplotlib.pyplot

 	

# Normalization Functions

def Normalization():
	Normalization = 100
	credit = credit - Normalization
	return h_train[x]=(h_train[x]-h_train[x].mean())/h_train[x].std()

		
