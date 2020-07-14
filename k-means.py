#=========-=============================IMPORTING NEEDED LINRARIES=======================================
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import pandas as pd
#%matplotlib inline #needed in jupyter notebooks


#============================================DATASET=====================================

#===================================Reading the data from the dataset in=================================
cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.head()
print(df)