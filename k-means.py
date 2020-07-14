#=========-=============================IMPORTING NEEDED LINRARIES=======================================
import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
#%matplotlib inline #needed in jupyter notebooks


#============================================DATASET=====================================

#===================================Reading the data from the dataset in=================================
cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.head()
print(df)


#=======================================DATA PREPROCESSING======================================

#========================================Preprocessing======================================
df = cust_df.drop('Address', axis=1)
df = df.head()
print(df)

#=================================Normalizing over standard deviation==============================

X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)