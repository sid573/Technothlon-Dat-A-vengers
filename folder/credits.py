import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import os
import sys




h_train = pd.read_csv('train.csv')

credit = 1000

# for data analysis of given data set using pandas

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

	