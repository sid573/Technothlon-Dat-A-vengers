import numpy as np
import pandas as pd
import house_mod as hd

df,df_test = hd.load_data();
print(df.shape);
print(df_test.shape)

hd.see_null(df)
hd.see_null(df_test)

nan_col = []
nan_col_test = []
df,nan_col = hd.string_to_val(df,list_cols=['Alley','Street','Utilities','LandSlope','Condition2','RoofMatl','BsmtQual','BsmtCond','BsmtFinSF2','Heating','GarageYrBlt','GarageFinish','GarageQual','PoolArea','PoolQC','MiscFeature','YrSold','SaleType','MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageCond','PavedDrive','Fence','SaleCondition','RoofStyle','MasVnrType'])
df_test,nan_col_test = hd.string_to_val(df,list_cols=['Alley','Street','Utilities','LandSlope','Condition2','RoofMatl','BsmtQual','BsmtCond','BsmtFinSF2','Heating','GarageYrBlt','GarageFinish','GarageQual','PoolArea','PoolQC','MiscFeature','YrSold','SaleType','MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageCond','PavedDrive','Fence','SaleCondition','RoofStyle','MasVnrType'])

df_test = hd.str_null(df_test,nan_col_test)
df = hd.str_null(df,nan_col)

print(df.head(5))

df = hd.fill_null(df)
df_test = hd.fill_null(df_test)

X_train,Y_train = hd.convert_to_matrix(df,test = False)
X_test = hd.convert_to_matrix(df_test,test = True)

print(str(X_test.shape))
print(str(X_train.shape))
print(str(Y_train.shape))

Y_test = hd.Model(X_train,Y_train,X_test)
print(str(Y_test.shape))

hd.convert_to_csv(df,Y_test,test = False)
hd.convert_to_csv(df_test,Y_test,test = True)

