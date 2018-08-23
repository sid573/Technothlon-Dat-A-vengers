import numpy as np
import pandas as pd
import house_mod as hd
df = pd.read_csv("train_game.csv")

df = pd.DataFrame(df)


df,nan_col = hd.string_to_val(df,list_cols = ['Alley','Street','Utilities','LandSlope','Condition2','RoofMatl','BsmtQual','BsmtCond','BsmtFinSF2','Heating','GarageYrBlt','GarageFinish','GarageQual','PoolArea','PoolQC','MiscFeature','YrSold','SaleType','MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageCond','PavedDrive','Fence','SaleCondition','RoofStyle','MasVnrType'])
df = hd.str_null(df,nan_col)

df.to_csv("train_game7.csv",index=False)