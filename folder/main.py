import numpy as np
import pandas as pd
import house_mod as hd
import matplotlib.pyplot as plt

credits = 100000
model_counter = 0

# Initially 300 Data
df = hd.load_data_init_train()
df_test = hd.load_data_test()
print(df.shape)

def append_data():
	# Appending the Data
	extra_data,credits = hd.load_data_in(300,400,credits)
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

def normalization(df,input_val,credits,col_name	= None):
	""" Which Normalization to call """
	if(input_val == 1):
		df,credits = hd.better_normalization(df,col_name,credits)
	elif(input_val == 2):
		df,credits = hd.mean_normalization(df,col_name,credits)
	else:
		df,credits = hd.std_normalization(df,col_name,credits)
	return df,credits

def draw_graph(df,credits,input_val,col_name = None):
	""" Which Graph """
	if(input_val == 1):
		pl,credits = hd.line(df,credits,col_name)
	else:
		pl,credits = hd.histogram(df,credits)

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

def drop(df,credits,input_val,col_name = None,row_index = None):
	""" What to drop? """
	if(input_val == 1):
		df,credits = hd.drop_columns(df,credits,col_name)
	else:
		df,credits = hd.drop_rows(df,credits,row_index)

	return df,credits

def model_type(model_name,X_train,Y_train,X_test,credits):
	""" Which Model to call """
	model_counter+=1
	if(model_counter < 2):
		Y_test = hd.Model(X_train,Y_train,X_test)
	else:
		if(model_name == 'linear'):
			credits -= 3000
			Y_test = hd.Model(X_train,Y_train,X_test)

	return Y_test,credits

		

