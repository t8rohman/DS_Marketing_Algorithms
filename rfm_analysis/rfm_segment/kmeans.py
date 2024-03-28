import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import decomposition


class OptimizeClusters():
    '''
    Class for RFM-based KMeans clustering with methods for optimizing the number of clusters using elbow method and silhouette analysis.

    Parameters:
    - df : DataFrame
        Input DataFrame containing RFM scores.

    Methods:
    - method_elbow(K_max) : void
        Optimize the number of clusters using elbow method.

    - method_silhouette(K_max) : void
        Optimize the number of clusters using silhouette analysis.
    '''

    def __init__(self, df):
        self.df = df


    def method_elbow(self, K_max):
        '''
        Optimize the number of clusters using elbow method.

        Parameters:
        - K_max : int
            Maximum number of clusters to test.
        '''

        df = self.df

        distortions = []
        inertias = []

        # mapping 1 stores the sum of minimum distances from each sample to the cluster center for each value of k. 
        # the key is the value of k, and the value is the corresponding sum of minimum distances.
        mapping1 = {}

        # mapping 2 stores the inertia of the KMeans model for each value of k. 
        # the key is the value of k, and the value is the corresponding inertia.
        mapping2 = {}

        # range of K we want to test using elbow method
        K = range(2, K_max + 1)

        for k in K:
            model = KMeans(n_clusters=k).fit(df)

            distortions.append(sum(np.min(cdist(df, model.cluster_centers_, 'euclidean'),axis=1)) / df.shape[0]) 
            inertias.append(model.inertia_) 
        
            mapping1[k] = sum(np.min(cdist(df, model.cluster_centers_, 'euclidean'),axis=1)) / df.shape[0] 
            mapping2[k] = model.inertia_

        self.K = K
        self.distortions = distortions
        self.inertias = inertias

        inertias = self.inertias
        K = self.K
        
        plt.figure(figsize=(10,6))

        plt.plot(K, inertias, '-bx')
        plt.xlabel("K's value")
        plt.ylabel("Inertias")
        plt.title('Elbow Plot Using Inertias')

        plt.show()
    

    def method_silhouette(self, K_max):
        '''
        Optimize the number of clusters using silhouette analysis.

        Parameters:
        - K_max : int
            Maximum number of clusters to test.
        '''

        df = self.df

        K = range(2, K_max + 1)

        for n_clusters in K:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(df)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(df, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(df, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

            # use PCA to reduce the dimensionality from 3 to only 2 variables
            # for us to be able to visualize the data points on a 2D plot
            n = 2
            pca = decomposition.PCA(n_components=n)
            pca_res = pca.fit_transform(df)

            df['pc1'] = pca_res[:, 0]
            df['pc2'] = pca_res[:, 1]

            ax2.scatter(
                pca_res[:, 0], pca_res[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()


class FitEvaluate():
    '''
    A class for fitting and evaluating clustering models.

    Attributes:
        df_rfm_ori (DataFrame): The original DataFrame.
        df_rfm_scaled (DataFrame): The scaled and preprocessed DataFrame.
    '''

    def __init__(self, df_rfm_ori, df_rfm_ori_wo_id, df_rfm_scaled, K_max=None):
        '''
        Initialize the FitEvaluate object.

        Args:
            df_rfm_ori (DataFrame): The original DataFrame with it's customer ID.
            df_rfm_ori_wo_id (DataFrame): The original DataFrame without it's customer ID.
            df_rfm_scaled (DataFrame): The scaled DataFrame.
        '''
        self.df_rfm_ori = df_rfm_ori
        self.df_rfm_ori_wo_id = df_rfm_ori_wo_id
        self.df_rfm_scaled = df_rfm_scaled
        self.K_max = K_max


    @staticmethod
    def kmeans_plot_df_ind(df_rfm_scaled, df_rfm_ori_wo_id, n_clusters, return_df=False):
        '''
        Fit a KMeans model, assign cluster labels, and plot the t-SNE visualization.

        Args:
            df_rfm_scaled (DataFrame): The scaled DataFrame.
            df_rfm_ori (DataFrame): The original DataFrame.
            n_clusters (int): The number of clusters.
            return_df (bool): Whether to return the DataFrame with cluster labels.

        Returns:
            DataFrame or None: The DataFrame with cluster labels if return_df is True, otherwise None.
        '''
        # fit the kmeans model
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df_rfm_scaled)

        cluster_labels = kmeans.labels_

        # asssing cluster label
        # and calculate the tsne based on this df
        df_new = df_rfm_ori_wo_id.assign(Cluster=cluster_labels)

        if return_df:
            return df_new
        else:
            tsne = TSNE()
            transformed = tsne.fit_transform(df_new)

            # create t-sne plot
            plt.title('Flattened Graph of {} Clusters'.format(n_clusters))
            sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
            return plt.show()
    

    def kmeans_plot_all(self):
        '''
        Fit a KMeans model, assign cluster labels, and plot the t-SNE visualization for all clusters.
        '''
        df_rfm_ori_wo_id = self.df_rfm_ori_wo_id
        df_rfm_scaled = self.df_rfm_scaled
        K_max = self.K_max

        try:
            K_max = self.K_max
        except TypeError:
            print("K_max is None. Please define the K_max in the instance.")
            return

        K = range(2, K_max + 1)
        for n in K:
            plt.figure(figsize=(6, 6))
            self.kmeans_plot_df_ind(df_rfm_scaled, df_rfm_ori_wo_id, n, return_df=False)


    def kmeans_df_all(self):
        '''
        Fit a KMeans model, assign cluster labels, and return the DataFrame with cluster labels for all clusters.

        Returns:
            dict: A dictionary containing DataFrames with cluster labels for different numbers of clusters.
        '''
        df_rfm_ori_wo_id = self.df_rfm_ori_wo_id
        df_rfm_scaled = self.df_rfm_scaled
        K_max = self.K_max

        try:
            K_max = self.K_max
        except TypeError:
            print("K_max is None. Please define the K_max in the instance.")
            return

        K = range(2, K_max + 1)

        # placeholder for the dataframe
        df_kmeans_dict = {}

        for n in K:
            df_new = self.kmeans_plot_df_ind(df_rfm_scaled, df_rfm_ori_wo_id, n, return_df=True)
            df_kmeans_dict[n] = df_new
        
        self.df_kmeans_dict = df_kmeans_dict

        return df_kmeans_dict

    @staticmethod
    def snake_plot(df_rfm_ori, df_rfm_scaled, df_rfm_kmeans):
        '''
        Generate a snake plot for the clusters.

        Args:
            df_rfm_ori (DataFrame): The original DataFrame.
            df_rfm_scaled (DataFrame): The scaled DataFrame.
            df_rfm_kmeans (DataFrame): The DataFrame with cluster labels.

        Returns:
            None
        '''
        df_rfm_scaled['Customer ID'] = df_rfm_ori['Customer ID']
        df_rfm_scaled['Cluster'] = df_rfm_kmeans['Cluster']
        
        # melt data into long format
        df_melt = pd.melt(df_rfm_scaled.reset_index(), 
                        id_vars=['Customer ID', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'Monetary'], 
                        var_name='Metric', 
                        value_name='Value')
        
        n_clusters = df_rfm_scaled['Cluster'].max() + 1
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Snakeplot of {} Clusters'.format(n_clusters))

        return sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
    

    def snake_plot_all(self):
        '''
        Generate snake plots for all clusters.
        '''
        df_rfm_scaled = self.df_rfm_scaled
        df_rfm_ori = self.df_rfm_ori
        df_kmeans_dict = self.df_kmeans_dict
        K_max = self.K_max

        try:
            K_max = self.K_max
        except TypeError:
            print("K_max is None. Please define the K_max in the instance.")
            return 
        
        K = range(2, K_max + 1) 

        for n in K:
            plt.figure(figsize=(10, 4))
            self.snake_plot(df_rfm_ori, df_rfm_scaled, df_kmeans_dict[n])
            plt.show()