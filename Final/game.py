import numpy as np
import pandas as pd
import house_mod as hd
import main as mn

credits = 100000
print("The dataset with 300 entries will be loaded")
# Initially 300 Data
df = hd.load_data_init_train()
df_test = hd.load_data_test()
print(df.shape)
print(str(df_test.shape))
true_pred = pd.DataFrame({'SalePrice' :df_test['SalePrice']})
true_pred = true_pred.values
true_pred = true_pred.flatten()
print(str(true_pred.shape))

model_counter = 0
print("Dataset is Loaded")
input("Press to continue")
print()

print("Lets Begin !!!")
input("Press Key to continue")
print()

print("The Initial two Predicting Models you get are Free. After that if you use it it needs payment")
print()
print("Every Data Preprocessing Step amounts to something....So be wise to use your credits")
print()

# EXCEPTION HANDLING NOT DONE YET
while(True):

	print("Your Available Credits are " + str(credits))
	print("Use it Wisely....")
	print()
	print("Here's the List of Options Available for you")
	print()
	
	print("WARNING!!!! WARNING!!!! WARNING!!!!")
	print("Whenever you make a change to the DataFrame Never Forget to convert it to a Numpy Array")
	print()
	print("Also Remember Whatever Change you make in Train it happens to test as well")
	print()
	print("MENU")
	print()
	print("1 - Load Extra Data ( Cost of This Move : Range of Values * 2 ")
	print("2 - Null Values in your Datasets ")
	print("3 - Normalization ")
	print("4 - Draw Basic Graphs")
	print("5 - Draw Missing Number Graphs")
	print("6 - Fill Null Values")
	print("7 - Drop Values")
	print("8 - Choose a Model to Train")
	print("9 - Make a Numpy Array of the DataFrame (Use it After you make any change to DataFrame and before training for the model)")
	print("10 - Predict the Values (Do it when you have Trained your Model)")
	print("11 - Shows the Dataset (specific elements)")
	print()

	str_val = int(input("Enter your Choice\n"))
	
	if(str_val == 1):
		low = int(input("Enter Lower Range which should be obviously greater then 300 or any previous value you have entered"))
		high = int(input("Enter Higher Range which you need....Remember it Costs Credits"))
		df,credits = mn.append_data(low,high,credits)
		print("Added" + str(high - low) + "Datas")
		print("Shape of Train DataFrame = " + str(df.shape))
		print()

	elif(str_val == 2):
		print("Options Available")
		print()
		print("1 - Shows Cols with Null")
		print("2 - Number of Null in Cols")
		print("3 - Number of Columns having NULL")
		print()
		input_val = int(input("Enter the Value\n"))
		if(input_val == 2):
			col = input("Enter the Column Name\n")
			credits = mn.check_null(df,input_val,credits,col_name = col)
		else:
			credits = mn.check_null(df,input_val,credits)

		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 3):
		print("Options Available")
		print()
		print("1 - Mean and Standard Deviation")
		print("2 - Mean")
		print("3 - Standard Deviation")
		print()
		input_val = int(input("Enter the Value\n"))
		col = input("Enter the Column Name\n")
		df,credits = mn.normalization(df,input_val,credits,col_name = col)

		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 4):
		print("Options Available")
		print()
		print("1 - Line")
		print("2 - Histogram")
		print()
		input_val = int(input("Enter the Value\n"))
		col = ""
		
		col = input("Enter Column Name\n")

		credits = mn.draw_graph(df,credits,input_val,col_name = col)

		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 5):
		print("Options Available")
		print()
		print("1 - Matrix")
		print("2 - HeatMap")
		print("3 - Dendrogram")
		print("4 - Bar")
		print()
		input_val = int(input("Enter the Value\n"))
		credits = mn.null_graph(df,credits,input_val)

		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 6):
		print("Options Available")
		print()
		print("1 - Mean")
		print("2 - Zero")
		print("3 - Standard Deviation")
		print()
		input_val = int(input("Enter the Value\n"))
		col = input("Enter Column Name\n")
		df,credits = mn.fill_null(df,credits,input_val,col_name = col)

		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 7):
		print("Options Available")
		print()
		print("1 - Drop Cols")
		print("2 - Drop Rows")
		print()
		input_val = int(input("Enter the Value\n"))
		df,credits = mn.drop(df,credits,input_val)
		
		print("Your Available Credits are " + str(credits))
		print()

	elif(str_val == 8):
		print("You will train your model now, Did you change your DataFrame to a Numpy Array!!")
		print()
		print("Do it by pressing 9")
		check_model_str = input("Are we good to go?\n")

		if(check_model_str == 'n' or check_model_str == 'N'):
			print("We are returning back then!! Next time be Careful")
			continue

		model_counter+=1;
		print("Options Available")
		print()
		print("linear - Linear Regression")
		print("ridge - Ridge Regression")
		print("lasso - Lasso Regression")
		print()
		input_val = (input("Enter the Value\n"))
		Y_test,train_y,credits = mn.model_type(input_val,X_train,Y_train,X_test,credits,model_counter)
		print(str(Y_test.shape))
		print(str(train_y.shape))
		print("Your Available Credits are " + str(credits))
		print()


	elif(str_val == 9):
		X_train,Y_train = hd.convert_to_matrix(df,test = False)
		X_test = hd.convert_to_matrix(df_test,test = True)
		print("Shape of X_train, Y_train and X_test are " + str(X_train.shape) + str(Y_train.shape) + str(X_test.shape))
		print()

	elif(str_val == 10):
		print("Are u Sure you have trained the Model and got your Predictions")
		check_pred_str = input("So you want to continue\n")
		
		if(check_pred_str == 'n' or check_pred_str == 'N'):
			print("Revert back then")
			continue

		test_acc = mn.accuracy(Y_test,true_pred)
		train_acc = mn.accuracy(train_y,Y_train)

		credits = mn.prize(test_acc,train_acc,credits)

		print("Your Test Prediction Value is " + str(test_acc))
		print("Your Train accuracy is " + str(train_acc))
		print()
		print("Good! You got some Credits")
		print("Available Credits are " + str(credits))
		print()

	elif(str_val == 11):
		print("Options Available")
		print()
		print("1 - Shows a Column")
		print("2 - Shows a Row")
		print("3 - Shows the Dataset (The Range you select will decrease credits)")
		print()
		head_str = input("Enter your Choice\n")
		credits = mn.show_data(df,credits,head_str)
		print()

	# NEED TO ADD PREDICT FUCNTION AND ORIGINAL DATASET
	print("Well! Do you want to Continue")
	continue_str = input("Enter [y/n]\n")
	if(continue_str == 'y'):
		input_val = None
		continue
	else:
		print("So here we are at the end of the Game")
		print("Enjoy other events for the day and HAVE FUN!!!!")
		break;

	


print("Good Bye!!!")
input("Press B to bid us a goodbye!!!")
