import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans  # to perform kmeans analysis
from sklearn.preprocessing import StandardScaler  # to standardize data using z-score
from sklearn import decomposition  # the method for PCA is included in sklearn.decomposition library
from sklearn.metrics import silhouette_score # to calculate silhouetter score from K Means

# List of functions to do segmentation
# All the segmentation methods here are using KMeans algorithm
# Make some adjustment if you want to use another segmentation method


'''
opt_visualize() is a function to do segmentation analysis
df          : pass the data frame that has been normalized
features    : pass the columns on the dataframe that will be the features
clusters    : how many clusters do you want to see (from 2 until n_cluster)
markers     : markers (define the markers, n_markers should be the same with clusters)
colors      : colors (define the colors, n_colors should be the same with clusters)
'''

def opt_visualize(df, features, clusters, markers, colors):
    
    # PCA analysis first

    pca = decomposition.PCA(n_components=2)
    pca_res = pca.fit_transform(df[features])

    df['pc1'] = pca_res[:, 0]
    df['pc2'] = pca_res[:, 1]
    
    plt.figure(figsize=(20, 6), layout="constrained")
    
    i = 1
    
    for i in range(1, clusters):
        n_clust = i + 1
        model = KMeans(n_clusters=n_clust, random_state=42)  # define the model for k means
        
        model.fit(df[features])  # fitting the model to the data
        df['cluster_label'] = model.predict(df[features])  # predicting the cluster

        n_column = np.ceil(clusters / 2)
        n_column = int(n_column)
        
        ax = plt.subplot(2, n_column, i)
        
        for clust, color in zip(range(n_clust), colors):  # make loops to make the principal components plot
            temp = df[df['cluster_label'] == clust]  # filter the data to only for one specific cluster
            plt.scatter(temp['pc1'], temp['pc2'], marker=markers[clust], label='Cluster' + str(clust), color=color)
            plt.title('K Means with ' + str(n_clust) + ' Clusters')
            
    sns.despine()
    plt.show()
    
    
'''
opt_elbow() is a function to choose the best number for segments using elbow method
df          : dataframe containing your data
features    : variables used to make the segmentation
clusters    : how many clusters you want to plot (n-1)
'''

def opt_elbow(df, features, clusters):
    sse_s = []
    df_sse_s = pd.DataFrame()
    
    for n_clust in range(2, clusters):    
        model = KMeans(n_clusters=n_clust, random_state=42) 
        model.fit(df[features])
        sse = model.inertia_
        sse_s.append(sse)
        
        row_df = n_clust - 2
        df_sse_s.loc[row_df, 'n_cluster'] = n_clust
        df_sse_s.loc[row_df, 'sse'] = sse
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, clusters), sse_s)
    plt.title('Number of K Clusters to its SSE')
    plt.xticks(range(2, clusters))
    sns.despine()
        
    plt.show
    

'''
opt_silhouette() is a function to choose the best number for segments by looking at silhouette score of each number of clusters
df          : dataframe containing your data
features    : variables used to make the segmentation
clusters    : how many clusters you want to plot (n-1)
'''

def opt_silhouette(df, features, clusters):
    
    for n_clust in range(2, clusters):
        model = KMeans(n_clusters=n_clust, random_state=42)
        model.fit(df[features])
        silhouette_avg = silhouette_score(df[features], model.labels_)
        print(f'Silhouette score for {n_clust} Clusters:', round(silhouette_avg, 4))    