#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library
#print(df)

#assign your training data to X_training feature matrix
X_training = df
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
s_coefficients = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
    s_coefficients.append(silhouette_coefficient)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
k_values = [k for k in range(2, 21)]
plt.plot(k_values, silhouette_coefficient)
plt.show()
#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
test_data = pd.read_csv('testing_data.csv', header=None)
#print(test_data)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
data_labels = np.array('testing_data'.values).reshape(1, -1)[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(data_labels, kmeans.data_labels_).__str__())
#--> add your Python code here
homogenity_s_coefficient = (max(s_coefficients))
k_val = k_values[s_coefficients.index(homogenity_s_coefficient)]