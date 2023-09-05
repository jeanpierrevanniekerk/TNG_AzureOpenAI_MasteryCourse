# Databricks notebook source
# MAGIC %md
# MAGIC # Semantic Search on Specific Data Corpus
# MAGIC Query files within specifi corpus. 

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
# MAGIC ## Deploy a Language Model

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
    model = "text-similarity-davinci-001"

    # Now let's create the deployment
    print(f'Creating a new deployment with model: {model}')
    result = openai.Deployment.create(model=model, scale_settings={"scale_type":"standard"})
    deployment_id = result["id"]
    print(f'Successfully created {model} with deployment_id {deployment_id}')
else:
    print(f'Found a succeeded deployment that supports embeddings with id: {deployment_id}.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embeddings
# MAGIC
# MAGIC see [01-get-embeddings.ipynb](./01-get-embeddings.ipynb) on how to get embeddings.
# MAGIC
# MAGIC In this example, we will load embeddings from a file. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import pandas as pd
df_orig = spark.sql('select * from openai.document_analysis_embeddings').limit(100).toPandas()

# COMMAND ----------

import numpy as np

DEVELOPMENT = False

if DEVELOPMENT:
    # Sub-sample for development
    df = df_orig.sample(n=20, replace=False, random_state=9).copy()
else:
    df = df_orig.copy()

# drop rows with NaN
df.dropna(inplace=True)

# convert string to array
df["embedding"] = df['embedding'].apply(np.array)#.apply(eval)
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find documents with similar embeddings to the embeddings of the question

# COMMAND ----------

import numpy as np

def get_embedding(text, deployment_id=deployment_id):
    """ 
    Get embeddings for an input text. 
    """
    result = openai.Embedding.create(
      deployment_id=deployment_id,
      input=text
    )
    result = np.array(result["data"][0]["embedding"])
    return result

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    similarity = np.dot(x, y)
    return similarity 

def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve relevant news 

# COMMAND ----------

def retrieve_relevant_documents(query, contexts = df['embedding']):
    # find text most similar to the query
    answers = order_document_sections_by_query_similarity(query=query, contexts=contexts)[0:3]

    # print top 3
    for answer in answers:
        print(f'similarity score:   {answer[0]}')
        print(df['content'].loc[answer[1]], '\n')

    return

# COMMAND ----------

query = 'News about stock market.'
retrieve_relevant_documents(query=query)

# COMMAND ----------

query = 'What is happening in the rugby world?'
retrieve_relevant_documents(query=query)

# COMMAND ----------

query = 'What happened in Brazil?'
retrieve_relevant_documents(query=query)

# COMMAND ----------


