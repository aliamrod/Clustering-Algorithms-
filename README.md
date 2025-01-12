# Machine Learning Best Practices: A Preliminary Guide

## A. Introduction **

![Screenshot 2025-01-12 at 9 53 05 AM](https://github.com/user-attachments/assets/d92d12fa-c6f0-4918-8c3d-9b254e5a5850)

The flowchart illustrates the steps for selecting and building machine learning models. (1) Understand the problem type. Determine whether the task is regression, classification, or clustering, (2) Analyze data characteristics. Examine the data size, feature types, relationships, and distribution, (3) Select an appropriate model. Choose a model suited to the problem type and data properties, (4) Preprocess and engineer features. Perform tasks like scaling, encoding, and handle missing values, (5) Build and train the model. Split the data, train the model, and optimize hyperparameters, (6) Evaluate and interpret resuls. Use metrics to assess performance and interpret outcomes.

---

## B. Selecting and building appropriate ML models (e.g., regression, classification, clustering) based on the problem and data characteristics).

### a. Regression 

 Regression = 

 
 
### b. Classification

Classification = 



### c. Clustering

Clustering = Clustering is a type of unsupervised ML technique used to group similar data points together based on their inherent characteristics. The goal is to organize data into clusters such that: (1) Intra-cluster similarity (data points within the same cluster are as similar as possible) and (2) Inter-cluster dissimilarity (data points in different clusters are as distinct as possible). Clustering is widely used when labels or categories for the data are not pre-defined. It helps in identifying patterns, structures, or groupings in data without prior knowledge. 

Key features of clustering include (1) unsupervised learning. No labeled data is required; the algorithm discovers the groupings. (2) Similarity/dissimilarity metrics. Clustering relies on measures like Euclidean distance, cosine similarity, or other metrics to assess how data points are grouped. (3) Applications. Clustering is used in various domains, including customer segmentation, image segmentation, anomaly detection, and document categorization. *For example, suppose you have a dataset with customer purchase histories. A clustering algorithm can group customers into clusters based on their buying behaviors, such as Cluster_1: frequent small purchases, Cluster_2: Rare large purchases, Cluster_3: Moderate and consistent purchases. These clusters can then be used to tailor marketing strategies or improve customer service.


Types of Clustering Algorithms:
* Centroid-Based Algorithms
 	+ K-Means: Clusters are represented by centroids, and the algorithm minimizes the within-cluster sum of squares.
	+ K-Medoids: Similar to K-Means but uses actual data points (medoids) as cluster centers, making it more robust to outliers. "Medoids" instead of "Centroids" representating clustering; medoid = center point within cluster minimizing sum of distances to other points. Unlike K-Means, which is sensitive to outliers due to centroids being influenced by the mean of the data, K-Medoids assigns each data point to the nearest medoid _mi_, which is an actual data point in the dataset. This makes K-Medoids more robust to outliers, as medoids minimize the sum of dissimilarities (e.g., distances) between all data points in the cluster and the medoid, rather than relying on the arithmetic mean.

**Refer to K-Means and K-Medoids tutorial in centroid_algorithms.py. 

* Density-Based Algorithms
	+ DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Identifies clusters based on the density of points in the data, works well for non-spherical clusters and handles noise effectively.
	+ OPTICS (Ordering Points to Identify the Clustering Structure): Similar to DBSCAN but can identify clusters with varying densities. 

* Hierarchical Clustering
	+ Agglomeramative Clustering: Starts with each data point as its own cluster and merges clusters iteratively.
  	+ Divisive Clustering: Starts with one cluster and splits it iteratively (often visualized using dendograms). 
* Distribution-Based Algorithms
	+ Gaussian Mixture Models (GMMs): Assumes that data is generated from a mixture of Gaussian distributions and uses probabilistic assignments for clustering.
	  
* Grid-Based Algorithms
	+ CLIQUE (Clustering in Quest): Partitions the data space into a grid structure and clusters the dense regions.
	+ STING (Statistical Information Grid): Divides the space into a hierarchical grid for clustering.
* Spectral Clustering
* Fuzzy Clustering
* Model-Based Clustering
* Constraint-Based Clustering
* Deep Learning-Based Clustering
    + Item 2a
    + Item 2b



5. Grid-Based Algorithms
CLIQUE (Clustering in Quest): Partitions the data space into a grid structure and clusters the dense regions.
STING (Statistical Information Grid): Divides the data space into a hierarchical grid for clustering.
6. Spectral Clustering
Uses the eigenvalues of the similarity matrix of the data to perform dimensionality reduction before clustering, often applied to non-linear cluster structures.
7. Fuzzy Clustering
Fuzzy C-Means: Allows data points to belong to multiple clusters with varying degrees of membership.
8. Model-Based Clustering
Algorithms like Expectation-Maximization (EM) focus on fitting data into statistical models, often Gaussian distributions.
9. Constraint-Based Clustering
Uses additional constraints or prior knowledge, such as must-link or cannot-link constraints, to guide the clustering process.
10. Deep Learning-Based Clustering
Autoencoder-Based Clustering: Uses neural networks to learn lower-dimensional embeddings and clusters the data in this space.
Self-Organizing Maps (SOMs): A type of neural network trained to produce a low-dimensional representation of input data.
---

C. Evaluate model performance using relevant metrics (e.g., accuracy, precision, recall, F1-score) and fine-tune models to achieve efficient results.

D. Interpret model results, identify key drivers and insights. 


E. 


	1	descriptive statistics, hypothesis testing) to understand data distributions, relationships, and draw meaningful conclusions
	2	Regression, classification, clustering based on the problem and data characteristics
	3	Accuracy, precision, recall, F1-score, and fine-tune models to achieve efficient results
	4	Interpret model results

Some examples of clustering algorithms include:
- K-means clustering: a centroid model that represents each cluster with a single mean vector.

PROS
Simplicity: Easy to understand and implement.
Efficiency: Scales well to large datasets; computationally efficient for spherical, well-separated clusters.
Fast Convergence: Works quickly for datasets with distinct clusters.

CONS
Sensitivity to Initialization: The outcome depends heavily on the initial centroids.
Predefined K: Requires the number of clusters (K) to be defined upfront.
Shape Limitations: Struggles with non-spherical or overlapping clusters.
Outlier Sensitivity: Outliers can skew the centroids significantly.
- Hierarchical clustering: a connectivity model that builds models based on distance connectivity.
Pros:

Hierarchical View: Provides a dendrogram that helps visualize how clusters are formed.
No K Required: Does not require a predefined number of clusters.
Flexible: Works for various distance metrics and linkage methods.
Cons:

Scalability Issues: Computationally expensive for large datasets, as it requires O(n^2) memory and time.
Irreversibility: Once a split or merge is made, it cannot be undone.
Sensitivity to Noise: Outliers can distort the hierarchy significantly.
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise): a density model that defines clustering as a connected dense region in data space.
- Spectral clustering: an algorithm that uses spectral embeddings, eigenvalues, and eigenvectors to segment images, detect communities, and reduce dimensionality.
Pros:
- Effective for Complex Structures: Can identify clusters with non-convex boundaries.
- Dimensionality Reduction: Leverages eigenvalues and eigenvectors for efficient clustering in reduced dimensions.
- Versatility: Works for a wide range of applications, including image segmentation and community detection.

Cons:
- Scalability: Computationally expensive for large datasets due to eigen decomposition.
- Parameter Sensitivity: Requires careful tuning of similarity and graph construction parameters.
- Lack of Interpretability: The algorithm’s reliance on spectral embeddings makes it less intuitive.

Other Clustering Algorithms to Consider:
Gaussian Mixture Models (GMM): Probabilistic clustering that models data as a mixture of Gaussian distributions.
Pros: Handles overlapping clusters; probabilistic output.
Cons: Assumes Gaussian distribution; computationally intensive.

Agglomerative Clustering: A variant of hierarchical clustering that merges clusters iteratively.
Pros: Similar to hierarchical clustering; customizable linkage metrics.
Cons: Similar scalability and sensitivity issues as hierarchical clustering.

OPTICS (Ordering Points to Identify the Clustering Structure):
Pros: Handles clusters of varying densities; no fixed ϵ.
Cons: More complex than DBSCAN.


A. Descriptive Statistics & Hypothesis Testing: Apply statistical methods (e.g., descriptive statistics, hypothesis testing) to understand data distributions, relationships, and draw meaningful conclusions.

Descriptive Statistics: 
  Purpose = Summarize and understand the central tendency, dispersion (ie means the extent to which numerical data is likely to vary about an average value), and distribution of data. 

Understanding the Data: 
1. Data Overview
2. Data Cleaning
3. 
B. 
