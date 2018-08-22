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

while(True):
	print("Your Available Credits are " + credits)
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
	print("9 - Make a Numpy Array of the DataFrame (Use it After you make any change to DataFrame)")
	print()

	str_val = int(input("Enter your Choice\n"))
	
	if(str_val == 1):
		low = int(input("Enter Lower Range which should be obviously greater then 300 or any previous value you have entered"))
		high = int(input("Enter Higher Range which you need....Remember it Costs Credits"))
		df,credits = mn.append_data(low,high,credits)
		print("Added" + (high - low) + "Datas")
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
			credits = check_null(df,input_val,credits,col_name = col)
		else:
			credits = check_null(df,input_val,credits)

		print("Available Credits are " + credits)
		print()

	elif(str_val == 3):
			

	

