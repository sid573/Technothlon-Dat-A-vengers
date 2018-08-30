from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import TableSet, Credits
from pandas.compat import StringIO
import json 
##############################################
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import missingno as msno
import io
import base64
###############################################



def load_data_init_train():
	""" Creating the DataFrames """
	df = pd.read_csv("train_game.csv")
	df = pd.DataFrame(df)
	return df

def load_data_test():
	""" Test Loading Initially """
	df = pd.read_csv("train_game.csv")
	df = pd.DataFrame(df)
	return df[1000:]

def to_pd(df):
	return pd.read_csv(StringIO(df), sep='\s+')

###################################### Extra Data #############################
def load_data_in_train(x1,x2,credits):
	""" Creating Dataframes with credits change """
	df = pd.read_csv("train_game.csv")
	df = pd.DataFrame(df)
	if(x2 >= 1001):
		#print("Dont cross your limits!!")
		return None
	cdf = df[x1:x2]
	credits -= ((x2 - x1) * 5)
	return cdf,credits

def append_data(df, x1 , x2 , credits):
	# Appending the Data
	extra_data,credits = load_data_in_train(x1,x2,credits)
	df = df.append(extra_data)
	return df,credits
################################ E D Done ########################################

################################# Null Checking ########################################


def see_null_each(df,credits):
	"""Nullity check """

	credits -= 1500
	return credits

def null_sum(df,col_name,credits):
	""" Number of Null in Cols """
	credits -= 300
	return credits


def null_any(df,credits):
	""" Number of Columns having NULL """
	credits -= 800
	return credits


def check_null(df,input_val,credits,col_name = None ):
	""" Which Null to call """
	if(input_val == "columns_null"):
		credits = see_null_each(df,credits)
	elif(input_val == "total_null"):
		credits = null_sum(df,col_name,credits)
	else:
		credits = null_any(df,credits)
	return credits


##################################################### n c Done ###########################################
############################################ normalization ################################################

def better_normalization(df,col_name,credits):
	Normalization = 400
	credits = credits - Normalization
	df[col_name] = ((df[col_name] - df[col_name].mean())/df[col_name].std())
	return df,credits

def mean_normalization(df,col_name,credits):
	credits = credits - 400
	df[col_name] = (df[col_name] - df[col_name].mean())
	return df,credits

def std_normalization(df,col_name,credits):
	credits = credits - 400
	df[col_name] = df[col_name] / df[col_name].std()
	return df,credits 

 
def normalization(df,input_val,credits,col_name):
	""" Which Normalization to call """
	if(input_val == 1):
		df,credits = better_normalization(df,col_name,credits)
	elif(input_val == 2):
		df,credits = mean_normalization(df,col_name,credits)
	else:
		df,credits = std_normalization(df,col_name,credits)
	return df,credits
############################# nor done #######################################################################

########################################### graph data ##########################################################

def line(df,credits,col_name):
	""" Line Graph per column"""
	line = 50 
	credits = credits - line 
	return df[col_name].plot(x = df.shape[0] , y = df.shape[1] , kind = 'line'),credits

def histogram(df,credits , col_name):
	""" Histogram for Whole Data """
	hist = 500 
	credits = credits - hist 
	return df[col_name].plot(x = df.shape[1] , y = df.shape[0] ,kind = 'hist'),credits

def draw_graph(df,credits,input_val,col_name ):
	""" Which Graph """
	if(input_val == 1):
		pl,credits = line(df,credits,col_name)
	else:
		pl,credits = histogram(df,credits , col_name)

	buf = io.BytesIO()
	plt.savefig(buf, format='jpg')
	image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
	buf.close()
	return image_base64, credits

#############################done graph #######################################################################

################################################# null graph ####################################################
def matrix2(df,credits):
 	matrix3 = 500
 	credits = credits - matrix3
 	return msno.matrix(df.sample(df.shape[0])),credits

def heatmap2(df,credits):
 	heatmap3 = 400
 	credits = credits - heatmap3
 	return msno.heatmap(df.sample(df.shape[0])),credits	

def dendrogram2(df,credits):
 	dendrogram3 = 400
 	credits = credits - dendrogram3
 	return msno.dendrogram(df.sample(df.shape[0])),credits

def bar2(df,credits):
	bar3 = 700
	credits = credits - bar3
	return msno.bar(df.sample(df.shape[0])),credits


def null_graph(df,credits,input_val):
	""" Which Missing No Grpah to Call """
	# Graphs are Sexy but are not very Sizzling Hot #

	if(input_val == 1):
		pl,credits = matrix2(df,credits)
	elif(input_val == 2):
		pl,credits = heatmap2(df,credits)
	elif(input_val == 3):
		pl,credits = dendrogram2(df,credits)
	else:
		pl,credits = bar2(df,credits)
		
	buf = io.BytesIO()
	plt.savefig(buf, format='jpg')
	image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
	buf.close()
	return image_base64, credits



############################################ ng Done ############################################################## 
############################################# Fill null ###########################################################
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

def fill_null(df,credits,input_val,col_name):
	""" Which null filler to call """
	if(input_val == 1):
		df,credits = mean_null(df,credits,col_name)
	elif(input_val == 2):
		df,credits = zero_null(df,credits,col_name)
	else:
		df,credits = std_null(df,credits,col_name)

	return df,credits

############################################## f n done #########################################################
################################################ drop column ######################################################
def drop_columns(df,credits,col_name):
	""" Cols to be Dropped """
	df = df.drop(col_name,axis = 1)
	credits -= 50
	return df,credits

def drop_col(df,credits,col_name):
	
	df,credits = drop_columns(df,credits,col_name)
	return df,credits
	

######################################## d c done #################################################################

############################################### Drop Row ##########################################################
def drop_rows(df,credits,row_index):
	""" Row to be Dropped """
	df = df.drop(row_index,axis = 0)
	credits -= 100
	return df,credits
def drop_r(df,credits,row_index ):
	df,credits = drop_rows(df,credits,row_index)
	orig_df,_ = drop_rows(orig_df,0,row_index)

################################################ d r Done ########################################################

################################################## NUMPY MATRIX ################################################
def convert_to_matrix(df,test=False):
	""" X_train and X_test """
	# Creating the Training and Test Set
	temp_df = pd.DataFrame(df)
	if(df.isnull().values.any()):
		df = df.fillna(-1)

	X = df.loc[:,df.columns!='SalePrice'] #Locates and Allocate all cols except last one
	
	# Convert to Numpy Array
	X = X.values 
	df = temp_df
	
	if(test == False):
		Y = df['SalePrice']
		Y = Y.values
		return X,Y
	else:
		return X

################################################### N  M done ##################################################

############################################### LINEAR MODEL #####################################################
def Model_Linear(X_train,Y_train,X_test):
	lm = LinearRegression()
	model = lm.fit(X_train,Y_train)
	Y_test = model.predict(X_test)
	Y_test = Y_test.flatten()
	train_y = model.predict(X_train)
	train_y = train_y.flatten()
	return Y_test,train_y

def model_type(X_train,Y_train,X_test,credits,free_service):
	""" Which Model to call """

	if(free_service < 3):
		Y_test,train_y = Model_Linear(X_train,Y_train,X_test)
	else:
		credits -= 3000
		Y_test,train_y = Model_Linear(X_train,Y_train,X_test)

	free_service = free_service +1
	return Y_test,train_y,credits, free_service

def accuracy(Y_test,true_pred):
	""" Accuracy """
	val = metrics.explained_variance_score(Y_test,true_pred)
	return val * 100
######################################################LM done ###################################################


def Start(request):
	if request.user.is_authenticated:
		credit = Credits.objects.filter(user=request.user.id).first()
		columns = ["MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition","SalePrice"]
		return render(request,'main/index.html',{'credits':credit.credits,'cols':columns})
	else:
		return HttpResponse("Not Logged IN")


def Index(request):

	if request.user.is_authenticated:
		data = TableSet.objects.filter(user=request.user.id).first()
		if data is None:
			df = load_data_init_train()
			cre = Credits()
			cre.user = request.user
			cre.save()
			dat = TableSet()
			dat.user = request.user
			dat.data = df.to_string()
			dat.save()
			return HttpResponseRedirect('/start')
		else:
			return HttpResponseRedirect('/start')
	else:
		return HttpResponseRedirect('/accounts/login/')

def View_1(request):
	if request.method=="POST":
		low = request.POST['low']
		high = request.POST['high']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		df,credits = append_data(df,int(low),int(high),credits)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))

def View_2(request):
	if request.method=="POST":
		store = ""
		input_val = request.POST['input_val']
		col = request.POST['col']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		df = to_pd(ts.data)
		credits = cr.credits

		if input_val =="columns_null":
			credits = check_null(df,input_val,credits)
			pd.set_option('max_rows', 81)
			store = df.isna().any().to_dict()
		elif input_val =="total_null":
			credits = check_null(df,input_val,credits,col)
			store = str(df[col].isnull().sum())
		elif input_val =="t_c_null":
			credits = check_null(df,input_val,credits)
			store = int(df.isna().any().sum())
		
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		dic['data'] =  store
		return JsonResponse(dic)



def View_3(request):
	if request.method=="POST":
		input_val = int(request.POST['input_val'])
		col = request.POST['col']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		df,credits = normalization(df,input_val,credits,col )
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))



def View_4(request):
	if request.method=="POST":
		input_val = int( request.POST['input_val'])
		col = request.POST['col']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		graph, credits = draw_graph(df,credits,input_val,col)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		dic['graph'] = graph
		return HttpResponse(json.dumps(dic))

def View_5(request):
	if request.method=="POST":
		input_val = int(request.POST['input_val'])
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		graph , credits = null_graph(df,credits,input_val)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		dic['graph'] = graph
		return HttpResponse(json.dumps(dic))


def View_6(request):
	if request.method=="POST":
		input_val = int(request.POST['input_val'])
		col = request.POST['col']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		df,credits = fill_null(df,credits,input_val,col)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))

def View_7(request):
	if request.method=="POST":
		col = request.POST['col']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		df,credits = drop_col(df,credits,col)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))

def View_8(request):
	if request.method=="POST":
		input_val = request.POST['input_val']
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		df,credits = drop_r(df,credits,input_val)
		ts.data = df.to_string()
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))

def View_9(request):
	if request.method=="POST":
		input_val = int(request.POST['input_val'])
		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		df = to_pd(ts.data)
		X_train,Y_train = convert_to_matrix(df[0:input_val],test = False)
		X_test = convert_to_matrix(df[1000:],test = True)
		true_pred = df['SalePrice']
		true_pred = pd.DataFrame(true_pred)
		true_pred = true_pred[1000:]
		true_pred = true_pred.values
		free_service = ts.free_service
		Y_test,train_y,credits, free_service = model_type(X_train,Y_train,X_test,credits,free_service)
		test_acc = accuracy(Y_test,true_pred)
		train_acc = accuracy(train_y,Y_train)
		ts.data = df.to_string()
		ts.free_service = free_service
		cr.credits = credits
		ts.save()
		cr.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		dic['acc_test'] = test_acc
		dic['acc_train'] = train_acc
		dic['c']= free_service
		return HttpResponse(json.dumps(dic))

def c_p(request):
	if request.method=="POST":
		
		cn = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').count()
		while cn>=2:
			ts = TableSet.objects.filter(user=request.user.id).order_by('checkpoint').first().delete()
			cn = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').count()

		ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
		cr = Credits.objects.filter(user=request.user.id).first()
		credits = cr.credits
		credits = credits - 500
		cr.credits = credits
		ts.save()
		cr.save()
		tsn = TableSet()
		tsn.data = ts.data
		tsn.user_id = request.user.id
		tsn.checkpoint = ts.checkpoint +1
		tsn.save()
		dic = {}
		dic['credit'] = credits
		dic['message'] = "Successful."
		return HttpResponse(json.dumps(dic))

def c_p_revert(request):
	if request.method=="POST":
		
		cn = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').count()
		if cn>1:		
			ts = TableSet.objects.filter(user=request.user.id).order_by('-checkpoint').first()
			cr = Credits.objects.filter(user=request.user.id).first()
			credits = cr.credits
			credits = credits - 500
			cr.credits = credits
			ts.delete()
			cr.save()
			dic = {}
			dic['credit'] = credits
			dic['message'] = "Successful."
			return HttpResponse(json.dumps(dic))
		else:		
			dic = {}
			cr = Credits.objects.filter(user=request.user.id).first()
			credits = cr.credits
			dic['credit'] = credits
			dic['message'] = "No Previous Checkpoint."
			return HttpResponse(json.dumps(dic))


