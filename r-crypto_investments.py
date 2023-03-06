#!/usr/bin/env python
# coding: utf-8

# # Module 10 Application
# 
# ## Challenge: Crypto Clustering
# 
# In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.
# 
# The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Import the Data (provided in the starter code)
# * Prepare the Data (provided in the starter code)
# * Find the Best Value for `k` Using the Original Data
# * Cluster Cryptocurrencies with K-means Using the Original Data
# * Optimize Clusters with Principal Component Analysis
# * Find the Best Value for `k` Using the PCA Data
# * Cluster the Cryptocurrencies with K-means Using the PCA Data
# * Visualize and Compare the Results

# ### Import the Data
# 
# This section imports the data into a new DataFrame. It follows these steps:
# 
# 1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.
# 
# 2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.
# 
# 
# > **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 

# In[19]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# import and load .env file
from dotenv import load_dotenv
load_dotenv()


# In[20]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[21]:


# Generate summary statistics
df_market_data.describe()


# In[22]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data
# 
# This section prepares the data before running the K-Means algorithm. It follows these steps:
# 
# 1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.
# 
# 2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
# 

# In[23]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)


# In[24]:


# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()

# plot new data frame
# Plot your data to see what's in your DataFrame
df_market_data_scaled.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Find the Best Value for k Using the Original Data
# 
# In this section, you will use the elbow method to find the best value for `k`.
# 
# 1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following question: What is the best value for `k`?

# In[25]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11


k = list(range(1,11))


# In[26]:


# Create an empy list to store the inertia values

inertia = []


# In[27]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters =i, random_state=1)
    k_model.fit(df_market_data_scaled)
    inertia.append(k_model.inertia_)
    
print(f"The length of k is {len(k)}")
print(f"The length of inertia is {len(inertia)}")

# pop inertia items
if len(inertia) > len(k):
    inertia.pop()


# In[28]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {'k': k, 'inertia': inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# review df_elbow
df_elbow.head()


# In[29]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.hvplot(
    x="k",
    y="inertia",
    title="Elbow Curve",
    xlabel="Number of Clusters",
    ylabel="Inertia"
)


# #### Answer the following question: What is the best value for k?
# **Question:** What is the best value for `k`?
# 
# **Answer:** # k =4 ist at elbow so is best value

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data
# 
# In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the original data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.
# 
# 4. Create a copy of the original data and add a new column with the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[30]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)


# In[31]:


# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)

# print intermediate data used in model

# Print the centroids
print(model.cluster_centers_)

# Print the labels assigned to each data point
print(model.labels_)

# Print the inertia value of the model
print(model.inertia_)


# In[32]:


# Predict the clusters to group the cryptocurrencies using the scaled data
k_predict = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
display(k_predict)


# In[33]:


# Create a copy of the DataFrame
df_market_data_predictions = df_market_data_scaled.copy()


# In[34]:


# Add a new column to the DataFrame with the predicted clusters
df_market_data_predictions['clusters_elbow'] = k_predict

# Display sample data
display(df_market_data_predictions)


# In[35]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_predictions.hvplot(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="clusters_elbow",
    kind="scatter",
    hover_cols=["coin_id"],
    title="Cryptocoin Data with Clusters (elbow k)"
)


# In[36]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_200d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_predictions.hvplot(
    x="price_change_percentage_200d",
    y="price_change_percentage_24h",
    by="clusters_elbow",
    kind="scatter",
    hover_cols=["coin_id"],
    title="Cryptocoin Data with Clusters (elbow k)"
)


# ---

# ### Optimize Clusters with Principal Component Analysis
# 
# In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.
# 
# 1. Create a PCA model instance and set `n_components=3`.
# 
# 2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 
# 
# 3. Retrieve the explained variance to determine how much information can be attributed to each principal component.
# 
# 4. Answer the following question: What is the total explained variance of the three principal components?
# 
# 5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

# In[39]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.

# Create a PCA model instance and set `n_components=3`.
pca_model = PCA(n_components=3)

# Fit the model to the data and transform the data
pca_data = pca_model.fit_transform(df_market_data_predictions)

# review 5 rows of data
pca_data[::5]


# In[38]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
print(pca_model.explained_variance_ratio_)


# #### Answer the following question: What is the total explained variance of the three principal components?
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** adding the sum of all three principal components listed above - the total explained variance is 88.4%.

# In[57]:


# Create a new DataFrame with the PCA data.
crypto_pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2", "PCA3"])

# Creating a DataFrame with the PCA data
#df_crypto_pca["coin_id"] = df_market_data_predictions.index

# Copy the crypto names from the original data
# YOUR CODE HERE!

# Set the coinid column as index
# YOUR CODE HERE!

# Display sample data
crypto_pca_df.head()


# ---

# ### Find the Best Value for k Using the PCA Data
# 
# In this section, you will use the elbow method to find the best value for `k` using the PCA data.
# 
# 1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

# In[58]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k_pca = list(range(1,11))


# In[59]:


# Create an empy list to store the inertia values
k_inertia =[]


# In[60]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
# Create a new DataFrame without the `coin_id` column
#df_crypto_pca_numeric = df_crypto_pca.drop("coin_id", axis=1)


# Create a for loop to compute the inertia with each possible value of k
for i in k_pca:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(crypto_pca_df)
    k_inertia.append(model.inertia_)


# In[61]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": k_inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)


# In[62]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca


# #### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** The elbow for PCA data is at 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** They are both same. The elbow curve for the PCA data is more defined than that of original data.

# ---

# ### Cluster Cryptocurrencies with K-means Using the PCA Data
# 
# In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the PCA data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.
# 
# 4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[63]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[64]:


# Fit the K-Means model using the PCA data
model.fit(crypto_pca_df)


# In[65]:


# Predict the clusters to group the cryptocurrencies using the PCA data
crypto_pca_clusters = model.predict(crypto_pca_df)

# View the resulting array of cluster values.
print(crypto_pca_clusters)


# In[66]:


# Create a copy of the DataFrame with the PCA data
crypto_pca_df_copy = crypto_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
crypto_pca_df_copy = pd.DataFrame(crypto_pca_df_copy, columns=["PCA1", "PCA2", "PCA3"])

# Display sample data
display(crypto_pca_df_copy)


# In[70]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

# Add a class column with the labels
crypto_pca_df_copy["customer_segments"] = crypto_pca_clusters

crypto_pca_df_copy.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="customer_segments"
)


# ---

# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.
# 
# 1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.
# 
# 2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.
# 
# 3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# > **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).

# In[ ]:


# Composite plot to contrast the Elbow curves
# YOUR CODE HERE!


# In[ ]:


# Compoosite plot to contrast the clusters
# YOUR CODE HERE!


# #### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** # YOUR ANSWER HERE!
