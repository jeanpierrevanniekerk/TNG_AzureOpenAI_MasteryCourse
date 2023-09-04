# Databricks notebook source
# MAGIC %md
# MAGIC # Retrieve Information from Specific Data Corpus

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
# MAGIC !pip install -r ./requirements.txt

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

DEVELOPMENT = False  # Set to True for development using a subset of data

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
# MAGIC ## Count tokens

# COMMAND ----------

import tiktoken
encoding = tiktoken.get_encoding('gpt2')

df['token_count'] = ''

for idx, title, content in zip(df.index.values, df['title'].loc[df.index.values], df['content'].loc[df.index.values]):
    df['token_count'].loc[idx] = len(encoding.encode(content))

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
# MAGIC ## Construct prompt
# MAGIC Add relevant document sections to the query prompt.

# COMMAND ----------

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

# COMMAND ----------

def construct_prompt(query: str, context_embeddings: pd.DataFrame, df: pd.DataFrame) -> str:
    """
    Append sections of document that are most similar to the query.
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(query, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section['token_count'] + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections, with indexes:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question truthfully using context, if unsure, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"
    
    return prompt

# COMMAND ----------

query = 'News about stock market.'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Information

# COMMAND ----------

def retrieve_information(prompt):
    try:
        # Request API
        response = openai.Completion.create(
            deployment_id= "text-davinci-003", # has to be deployment_id
            prompt=prompt,
            temperature=1,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=1
        )

        # response
        result = response['choices'][0]['text']; print(result)
    except Exception as err:
        print(idx)
        print(f"Unexpected {err=}, {type(err)=}")

    return 

# COMMAND ----------

query = 'News about stock market.'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'What is the state of the economy of the world?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'Summarise the state of the economy of the world?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'What is the most talked about technology?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'List all the celebrities.'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'What are the sports in the news?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'Tell me about all the football games.'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'Who won in the swimming contest?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------

query = 'Who won in the running competition?'
prompt = construct_prompt(query=query, context_embeddings=df['embedding'], df=df); print(prompt)
retrieve_information(prompt=prompt)

# COMMAND ----------


