import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def compute_euclidean_distance(inputs, x , y):
    #computes euclidean distance between to data tuples x n y

    sum=0
    for i in range(len(inputs[0])):
        sum+=(inputs[x,i]-inputs[y,i])**2
    return math.sqrt(sum)


def compute_distance_matrix(inputs):
    #computes similarity matrix based on euclidean similarity measure between all data tuples

    dist_matrix=np.empty([len(inputs),len(inputs)])
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            if i==j:
                dist_matrix[i,j]=float('inf') #Distance between a tuple and itself is set to infinite
            else:
                dist_matrix[i,j]=compute_euclidean_distance(inputs, i , j)
    return dist_matrix

def merge_minimum_distance(dist_matrix, clusters):
    #merges clusters with least distance between any of its data tuples in the distance/similarity matrix

    min=float('inf')
    x,y=np.where(dist_matrix==np.min(dist_matrix))#tuples with minimum distance between them

    for i in range(len(x)):#iterating through tuples with minimum distance between them
        new = [] #for merged cluster
        for j in clusters:
            if(x[i] in j or y[i] in j):#selecting cluster with required tuple
                new=new+j
                clusters.remove(j)#removing individual cluster
        clusters.append(new)#appending merged cluster
        dist_matrix[x[i],y[i]]=float('inf') #distance between these tuples now set to infinity
    return clusters


def single_link_hierarchical(inputs, k):
    #returns a nested list of clusters


    #initializing with each tuple as an individual cluster
    clusters = []
    for i in range(len(inputs)):
        clusters.append([i])

    dist_matrix=compute_distance_matrix(inputs)
    while(len(clusters)>k):
        clusters=merge_minimum_distance(dist_matrix, clusters)
    return clusters



#Reading dataset
df=pd.read_csv("iris_data.csv")
print(df.head())

inputs=df.iloc[:,:2]
print(inputs.head())

scaler=StandardScaler()
scaled_inputs=scaler.fit_transform(inputs)

k=int(input("Enter number of clusters: "))
clusters= single_link_hierarchical(scaled_inputs, k) #returns a nested list of clusters

plt.figure(1)
plt.title("Heirarchial Clustering")
for i in range(len(clusters)): #Plotting each cluster
    plt.scatter(inputs.iloc[clusters[i], 0], inputs.iloc[clusters[i],1])
plt.xlabel(inputs.columns[0])
plt.ylabel(inputs.columns[1])
plt.show()