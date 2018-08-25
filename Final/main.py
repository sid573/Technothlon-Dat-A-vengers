import numpy as np
import pandas as pd
import house_mod as hd
import matplotlib.pyplot as plt
from sklearn import metrics
credits = 100000

# Initially 300 Data
df = hd.load_data_init_train()
df_test = hd.load_data_test()
orig_df = pd.read_csv("train_game.csv")
print(df.shape)

def append_data(x1 , x2 , credits):
	# Appending the Data
	extra_data,credits = hd.load_data_in_train(x1,x2,credits)
	global df
	df = df.append(extra_data)
	return df,credits

def check_null(df,input_val,credits,col_name = None):
	""" Which Null to call """
	if(input_val == 1):
		credits = hd.see_null_each(df,credits)
	elif(input_val == 2):
		credits = hd.null_sum(df,col_name,credits)
	else:
		credits = hd.null_any(df,credits)
	return credits

def normalization(df,input_val,credits,col_name):
	""" Which Normalization to call """
	if(input_val == 1):
		df,credits = hd.better_normalization(df,col_name,credits)
	elif(input_val == 2):
		df,credits = hd.mean_normalization(df,col_name,credits)
	else:
		df,credits = hd.std_normalization(df,col_name,credits)
	return df,credits

def draw_graph(df,credits,input_val,col_name):
	""" Which Graph """
	if(input_val == 1):
		pl,credits = hd.line(df,credits,col_name)
	else:
		pl,credits = hd.histogram(df,credits,col_name)

	plt.show()
	return credits


def null_graph(df,credits,input_val):
	""" Which Missing No Grpah to Call """
	# Graphs are Sexy but are not very Sizzling Hot #
	if(input_val == 1):
		pl,credits = hd.matrix(df,credits)
	elif(input_val == 2):
		pl,credits = hd.heatmap(df,credits)
	elif(input_val == 3):
		pl,credits = hd.dendrogram(df,credits)
	else:
		pl,credits = hd.bar(df,credits)
	
	plt.show()
	return credits

def fill_null(df,credits,input_val,col_name):
	""" Which null filler to call """
	if(input_val == 1):
		df,credits = hd.mean_null(df,credits,col_name)
	elif(input_val == 2):
		df,credits = hd.zero_null(df,credits,col_name)
	else:
		df,credits = hd.std_null(df,credits,col_name)

	return df,credits

def drop(df,credits,input_val):
	""" What to drop? """
	if(input_val == 1):
		col_name = input("Enter Column Name\n")
		df,credits = hd.drop_columns(df,credits,col_name)
		orig_df,_ = hd.drop_columns(orig_df,0,col_name)
	else:
		row_index = int(input("Enter Row Name\n"))
		df,credits = hd.drop_rows(df,credits,row_index)
		orig_df,_ = hd.drop_rows(orig_df,0,row_index)

	return df,credits

def model_type(model_name,X_train,Y_train,X_test,credits,model_counter):
	""" Which Model to call """

	model_counter+=1
	if(model_counter < 2):
		Y_test = hd.Model_Linear(X_train,Y_train,X_test)
	else:
		if(model_name == 'linear'):
			credits -= 3000
			Y_test,train_y = hd.Model_Linear(X_train,Y_train,X_test)
		elif(model_name == 'ridge'):
			credits -= 5000
			Y_test,train_y = hd.Model_Ridge(X_train,Y_train,X_test)
		elif(model_name == 'lasso'):
			credits -= 5000
			Y_test,train_y = hd.Model_Lasso(X_train,Y_train,X_test)

	return Y_test,train_y,credits

def accuracy(Y_test,true_pred):
	""" Accuracy """
	val = metrics.explained_variance_score(Y_test,true_pred)
	return val * 100


def show_data(df,credits,input_val):
	""" Shows Parts of Data """
	if(input_val == 1):
		col_name = input("Enter Column Name\n")
		print(df[col_name])
		credits -= 1000
	elif(input_val == 2):
		row_index = int(input("Enter Row Index\n"))
		print(df.iloc[row_index])
		credits -= 100
	elif(input_val == 3):
		range_of_vals = int(input("Enter The Column Range\n"))
		credits -= range_of_vals * 10
		print(df.head(range_of_vals))

	return credits


def prize(test_acc,train_acc,credits):
	""" Credits given to them for getting good train and test accuracy """
	######### Give Credits depending on the Accuracies ########
	######### Credits += (Whatever will be the best)

	return credits