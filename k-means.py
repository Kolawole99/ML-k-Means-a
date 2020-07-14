#=========-=============================IMPORTING NEEDED LINRARIES=======================================
import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 
#%matplotlib inline #needed in jupyter notebooks



#============================================DATASET=====================================

#===================================Reading the data from the dataset in=================================
cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.head()
print(df)



#=======================================DATA PREPROCESSING======================================

#========================================Preprocessing======================================
df = cust_df.drop('Address', axis=1)
df_view = df.head()
print(df_view)

#=================================Normalizing over standard deviation==============================
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)



#=============================================MODELLING=============================================

#==============================We use the k-means to model the dataframe==============================
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#=======================================INSIGHTS INTO THE DATAFRAME=======================================

#================================Assign labels to each row in the dataframe==========================
df["Clus_km"] = labels
label_view = df.head(5)
print(label_view)

#==============================Checking for the centroid values by average============================
centroid = df.groupby('Clus_km').mean()
print(centroid)

#=========================Look at the distribution of customers by age and Income=====================
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

#==============================Plotting the image of the scatter plot=============================
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()
