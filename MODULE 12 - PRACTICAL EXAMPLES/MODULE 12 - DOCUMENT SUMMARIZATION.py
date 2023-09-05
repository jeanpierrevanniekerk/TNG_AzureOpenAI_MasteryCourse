# Databricks notebook source
# MAGIC %md
# MAGIC # Summarise Documents

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
# MAGIC ## Load Data

# COMMAND ----------

import pandas as pd
df = spark.sql('select * from openai.bbc_news_data').limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Request to API

# COMMAND ----------

# create prompt
prompt_postfix = """ \n\nTl;dr """

prompt = df['title'].loc[0] + "\n" + df['content'].loc[0] + prompt_postfix
prompt

# COMMAND ----------

# Request API
response = openai.Completion.create(
  deployment_id="text-davinci-003", # has to be deployment_id
  prompt=prompt,
  temperature=1,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=1
)

# print response
response['choices'][0]['text']

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------

# COMMAND ----------

results = pd.DataFrame(columns=['summary'], index=df.index)

# prompt postifx
prompt_postfix = """ 
  \n\nTl;dr
"""

for idx, title, content in zip(df.index.values, df['title'].loc[df.index.values], df['content'].loc[df.index.values]):
  
  # build prompt
  prompt = title + "\n" + content + prompt_postfix

  try:
    # Request API
    response = openai.Completion.create(
      deployment_id="text-davinci-003", # has to be deployment_id
      prompt=prompt,
      temperature=1,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=1
    )

      # response
    results['summary'].loc[idx] = response['choices'][0]['text']
  except Exception as err:
    idx
    print(f"Unexpected {err=}, {type(err)=}")

# COMMAND ----------

df_results = pd.concat([df, results], axis=1)
df_results.shape
df_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

df_results = spark.createDataFrame(df_results)
df_results.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("openai.document_analysis_summarize")
df_results = spark.sql('select * from openai.document_analysis_predictions')
df_results.limit(10).toPandas()

# COMMAND ----------


