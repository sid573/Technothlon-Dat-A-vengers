import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import missingno as msno


########### LOADING TRAIN DATA #####################

def load_data_init_train():
	""" Creating the DataFrames """
	df = pd.read_csv("train_mod.csv")
	df = pd.DataFrame(df)
	return df[0:300]

def load_data_in_train(x1,x2,credits):
	""" Creating Dataframes with credits change """
	df = pd.read_csv("train_mod.csv")
	df = pd.DataFrame(df)
	if(x2 >= 1001):
		print("Dont cross your limits!!")
		return None
	cdf = df[x1:x2]
	credits -= ((x2 - x1) * 2)
	return cdf,credits

############## LOADING TRAIN DONE ######################

################### LOADING TEST #######################

def load_data_test():
	""" Test Loading Initially """
	df = pd.read_csv("train_mod.csv")
	df = pd.DataFrame(df)
	return df[1000:]

################# LODING TEST DONE #######################

############# NULLITY CHECK ######################

def see_null_each(df,credits):
	"""Shows Cols with Null"""
	print(df.isna().any())
	credits -= 100
	return credits

def null_sum(df,col_name,credits):
	""" Number of Null in Cols """
	print(str(df[col_name].isnull().sum()))
	credits -= 50
	return credits


def null_any(df,credits):
	""" Number of Columns having NULL """
	print(df.isna().any().sum())
	credits -= 300
	return credits

############ NULLITY CHECK DONE ####################

############ NORMALISATION ####################

def better_normalization(df,col_name,credits):
	Normalization = 100
	credits = credits - Normalization
	df[col_name] = ((df[col_name] - df[col_name].mean())/df[col_name].std())
	return df,credits

def mean_normalization(df,col_name,credits):
	credits = credits - 50
	df[col_name] = (df[col_name] - df[col_name].mean())
	return df,credits

def std_normalization(df,col_name,credits):
	credits = credits - 20
	df[col_name] = df[col_name] / df[col_name].std()
	return df,credits 

############ NORMALISATION DONE ###################

############# VISUALIZATION OF TRAINING SET ###############

def line(df,credits,col_name):
	""" Line Graph per column"""
	line = 50 
	credits = credits - line 
	return df[col_name].plot(kind = 'line'),credits

def histogram(df,credits):
	""" Histogram for Whole Data """
	hist = 500 
	credits = credits - hist 
	return df.plot(kind = 'hist'),credits

############ VISUALIZATION DONE #######################

############ MISSING NUMBER VISUALIZATION #################

def matrix(df,credits):
 	matrix = 500
 	credits = credits - matrix
 	return msno.matrix(df.sample(250)),credits

def heatmap(df,credits):
 	heatmap = 400
 	credits = credits - heatmap
 	return msno.heatmap(df.sample(250)),credits	

def dendrogram(df,credits):
 	dendrogram = 400
 	credits = credits - dendrogram
 	return msno.dendrogram(df.sample(250)),credits

def bar(df,credits):
	bar = 700
	credits = credits - bar
	return msno.bar(df.sample(250)),credits

################## MISSING DONE #########################

##################### DROPPING COLUMNS ######################

def drop_columns(df,credits,col_name):
	""" Cols to be Dropped """
	df = df.drop(col_name,axis = 1)
	credits -= 50
	return df,credits

def drop_rows(df,credits,row_index):
	""" Row to be Dropped """
	df = df.drop(row_index,axis = 0)
	credits -= 100
	return df,credits

###################### DROPPING DONE  ########################

###################### ALREADY IMPLEMENTED #############################

def string_to_val(df,list_cols = ['Alley','Street','Utilities','LandSlope','Condition2','RoofMatl','BsmtQual','BsmtCond','BsmtFinSF2','Heating','GarageYrBlt','GarageFinish','GarageQual','PoolArea','PoolQC','MiscFeature','YrSold','SaleType','MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageCond','PavedDrive','Fence','SaleCondition','RoofStyle','MasVnrType']):
	""" Converts string to val """
	nan_col =[]
	for col in list_cols:
		check_nan_col = df[col].isnull().sum()

		if (check_nan_col) == 1:
			nan_col.append(col)
			continue
		
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

############################## ALREADY IMPLEMENTED DONE ##########################

############################## FILL NULL VALUES  ##################################

def mean_null(df,credits,col_name):
	""" Fill NUll values with mean"""
	df[col_name] = df[col_name].fillna(df[col_name].mean())
	credits -= 400
	return df,credits

def zero_null(df,credits,col_name):
	""" Fill Null with 0's """
	df[col_name] = df[col_name].fillna(0)
	credits -= 100
	return df,credits

def std_null(df,credits,col_name):
	""" Fill NUll values with standard deviation """
	df[col_name] = df[col_name].fillna(df[col_name].std())
	credits -= 250
	return df,credits

############################## FILL NULL VALUES DONE  ##############################

########################## MATRIX ENCODING ###############################

def convert_to_matrix(df,test=False):
	""" X_train and X_test """
	# Creating the Training and Test Set
	X = df.loc[:,df.columns!=['SalePrice','Id']] #Locates and Allocate all cols except last one
	
	# Convert to Numpy Array
	X = X.values 
	
	if(test == False):
		Y = df['SalePrice']
		Y = Y.values
		return X,Y
	else:
		return X

########################### MATRIX ENCODING DONE ###########################

################## Model ############################
	
def Model_Linear(X_train,Y_train,X_test):
	lm = LinearRegression()
	model = lm.fit(X_train,Y_train)
	Y_test = model.predict(X_test)
	Y_test = Y_test.flatten()
	plt.plot(model.history['acc'])
	plt.plot(model.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	return Y_test

def Model_Rigid(X_train,Y_train,X_test):
	reg = linear_model.Ridge (alpha = .5)
	model = reg.fit(X_train,Y_train)
	Y_test = model.predict(X_test)
	Y_test = Y_test.flatten()
	return Y_test

def Model_Lasso(X_train,Y_train,X_test):
	reg = linear_model.Lasso(alpha = 0.1)
	model = reg.fit(X_train,Y_train)
	Y_test = model.predict(X_test)
	Y_test = Y_test.flatten()
	return Y_test

##################### Model Done ######################

########################### CSV FILE ################################
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

####################### CSV FILE DONE #####################################	
