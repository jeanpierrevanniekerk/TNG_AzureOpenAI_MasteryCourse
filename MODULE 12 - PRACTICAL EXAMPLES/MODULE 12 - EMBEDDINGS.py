# Databricks notebook source
# MAGIC %md
# MAGIC # Get Embeddings

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from IPython.core.interactiveshell import InteractiveShell
# MAGIC InteractiveShell.ast_node_interactivity = "all"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Azure OpenAI

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install -r ../requirements.txt

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv

# Set up Azure OpenAI
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

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
df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy a model

# COMMAND ----------

# list models deployed with embeddings capability
deployment_id = None
result = openai.Deployment.list()

for deployment in result.data:
    if deployment["status"] != "succeeded":
        continue
    
    model = openai.Model.retrieve(deployment["model"])
    if model["capabilities"]["embeddings"] != True:
        continue
    
    deployment_id = deployment["id"]
    break

# if not model deployed, deploy one
if not deployment_id:
    print('No deployment with status: succeeded found.')
    model = "text-similarity-davinci-002"

    # Now let's create the deployment
    print(f'Creating a new deployment with model: {model}')
    result = openai.Deployment.create(model=model, scale_settings={"scale_type":"standard"})
    deployment_id = result["id"]
    print(f'Successfully created {model} with deployment_id {deployment_id}')
else:
    print(f'Found a succeeded deployment that supports embeddings with id: {deployment_id}.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Embeddings
# MAGIC ref: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=bash

# COMMAND ----------

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
df1 = df
print(len(df1))
df1['n_tokens'] = df1['content'].apply(lambda x: len(tokenizer.encode(x)))
df1 = df1[df1.n_tokens<8192]
print(len(df1))
df1

# COMMAND ----------

df['embedding'] = ''
print(len(df))
for i in range(len(df)):    
    try:
        embedding = openai.Embedding.create(input=df['content'][i], deployment_id=deployment_id)
        df['embedding'][i] = embedding['data'][0]['embedding']
    except Exception as err:
        i
        print(f"Unexpected {err=}, {type(err)=}")

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS OpenAI")
df_sp = spark.createDataFrame(df)
df_sp.write.mode('overwrite').option("overwriteSchema", "true").saveAsTable('openai.document_analysis_embeddings')

# COMMAND ----------

sdf = spark.sql('select * from openai.document_analysis_embeddings')
sdf.limit(10).toPandas()

# COMMAND ----------


