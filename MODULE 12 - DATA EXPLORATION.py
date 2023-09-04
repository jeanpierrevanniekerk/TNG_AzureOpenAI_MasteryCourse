# Databricks notebook source
# MAGIC %md
# MAGIC # Explore Data

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

!pwd

# COMMAND ----------

dbutils.fs.cp ("file:/Workspace/Repos/AzureOpenAIMasteryCourse/TNG_AzureOpenAI_MasteryCourse/bbc-news-data.csv", "dbfs:/FileStore/tables/bbc_news_data.csv")

# COMMAND ----------

import pandas as pd
# File location and type
file_location = "/FileStore/tables/bbc_news_data.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format('com.databricks.spark.csv') \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

temp_table_name = "bbc_news_data_embedding_csv"

df.createOrReplaceTempView(temp_table_name)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("openai.bbc_news_data")
df = spark.sql('select * from openai.bbc_news_data')
df_orig = df.toPandas()

# COMMAND ----------

df = df_orig.copy()
df

# COMMAND ----------

df.describe()

# COMMAND ----------

for col in ['category', 'filename']:
    print(df[col].unique())

# COMMAND ----------

df[['category', 'filename']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categories

# COMMAND ----------

#create bar plot to visualize frequency of each team
df['category'].value_counts().plot(kind='bar', xlabel='Category', ylabel='Articles Count', rot=0, title='Number of articles per category')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Length 

# COMMAND ----------

df["word_count"] = df["content"].apply(lambda n: len(n.split()))
df

# COMMAND ----------

# MAGIC %md
# MAGIC ### plot `word_count` per `category`

# COMMAND ----------

import seaborn as sns

sns.set_style("whitegrid")
sns.catplot(data=df, x="category", y="word_count", kind="box", height=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save data

# COMMAND ----------

#df.to_csv("../data/bbc-news-data-00.csv", sep='\t')

# COMMAND ----------


