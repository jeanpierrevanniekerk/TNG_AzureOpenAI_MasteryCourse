# Databricks notebook source
# MAGIC %md
# MAGIC # Classify Documents

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from IPython.core.interactiveshell import InteractiveShell
# MAGIC InteractiveShell.ast_node_interactivity = "all"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install -r ./requirements.txt

# COMMAND ----------

import pandas as pd
df = spark.sql('select * from openai.document_analysis_embeddings').toPandas()

# COMMAND ----------

df[~df['embedding'].isnull()]["embedding"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reduce dimensionality 
# MAGIC By
# MAGIC - TSNE ref: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
# MAGIC - PCA ref: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA 

# COMMAND ----------

if False:
    from sklearn.manifold import TSNE

    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=3, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims_tsne = tsne.fit_transform(df['embedding'].to_list())
    vis_dims_tsne.shape

# COMMAND ----------

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
vis_dims_pca = pca.fit_transform(df['embedding'].to_list())
vis_dims_pca.shape
vis_dims_pca

# COMMAND ----------

df["vis_dims_pca"] = vis_dims_pca.tolist()
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualise embeddings

# COMMAND ----------

# %matplotlib widget InteractivePlot
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
cmap = plt.get_cmap("tab20")

categories = sorted(df['category'].unique())

# Plot each sample category individually
for i, cat in enumerate(categories):
    sub_matrix = np.array(df[df["category"] == cat]["vis_dims_pca"].to_list())
    x=sub_matrix[:, 0]
    y=sub_matrix[:, 1]
    z=sub_matrix[:, 2]
    colors = [cmap(i/len(categories))] * len(sub_matrix)
    _ = ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)

_ = ax.set_xlabel('x')
_ = ax.set_ylabel('y')
_ = ax.set_zlabel('z')
_ = ax.legend()

# COMMAND ----------

#%matplotlib widget
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
cmap = plt.get_cmap("tab20")

categories = sorted(df['category'].unique())

# Plot each sample category individually
for i, cat in enumerate(categories):
    sub_matrix = np.array(df[df["category"] == cat]["vis_dims_pca"].to_list())
    x=sub_matrix[:, 0]
    y=sub_matrix[:, 1]
    z=sub_matrix[:, 2]
    colors = [cmap(i/len(categories))] * len(sub_matrix)
    _ = ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)

_ = ax.set_xlabel('x')
_ = ax.set_ylabel('y')
_ = ax.set_zlabel('z')
_ = ax.legend()

# COMMAND ----------


