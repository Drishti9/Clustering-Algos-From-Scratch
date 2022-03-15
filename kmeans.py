import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def compute_euclidean_dist(data_index, inputs, centroids):
    #computes euclidean distance of a data point from all centroids

    dist=np.zeros(k)
    for j in range(k):
        sum=0
        for i in range(len(inputs[0])):
            sum+=(inputs[data_index, i]-centroids[j,i])**2
        dist[j]=math.sqrt(sum)
    return dist

def associate_to_cluster(inputs,centroids):
    #To assign each data point to an existent cluster using euclidean similarity measure

    cluster= np.empty([len(inputs), 1])
    for i in range(len(inputs)):
        dist=compute_euclidean_dist(i, inputs,centroids) #computes distance from all centroids
        temp=np.where(dist==np.amin(dist)) #for centroid with minimum distance
        cluster[i]=temp[0]
    return cluster

def compute_cluster_centroid(inputs, cluster):
    #To compute cluster centroids

    centroids = np.empty([k, len(inputs[0])])

    for i in range(k): #iterating clusters
        #selecting elements belonging to current cluster i
        cluster_indexes=np.where(cluster==i)
        cluster_indexes=cluster_indexes[0]

        #centroid calculation
        for p in range(len(inputs[0])): #iterating attributes
            sum=0
            for j in cluster_indexes: #iterating indexes belonging to current cluster
                sum+=inputs[j,p]
            centroids[i,p]=sum/len(cluster_indexes)
    return centroids

def no_change(x, y):
    #to check for termination condition of K-means where centroids in subsequent iterations remain same
    comparison = x == y
    return comparison.all()

def kmeans(inputs, k):
    inputs_np=inputs.to_numpy()

    #initializing centroids
    prev_centroids = np.zeros([k, len(inputs.columns)])
    for i in range(k):
        centroids=inputs.iloc[0:k,0:].to_numpy()

    while(no_change(prev_centroids,centroids)==False):
        cluster=associate_to_cluster(inputs_np, centroids)
        prev_centroids=centroids
        centroids=compute_cluster_centroid(inputs_np, cluster)

    return centroids, cluster

#Reading dataset
print("Dataset: ")
df=pd.read_csv("iris_data.csv")
print(df.head())

print("\nInputs considered for clustering: ")
inputs=df.iloc[:,:2]
print(inputs.head())

#Standardizing Inputs
scaler=StandardScaler()
scaled_inputs=scaler.fit_transform(inputs)

#User input for number of clusters
k=int(input("Enter number of clusters: "))
centroids, cluster= kmeans(inputs, k)

#PLotting clusters
plt.figure(1)
plt.title("K-Means Clustering")
plt.scatter(inputs.iloc[:, 0], inputs.iloc[:, 1], c=cluster)
plt.xlabel(inputs.columns[0])
plt.ylabel(inputs.columns[1])
plt.show()

