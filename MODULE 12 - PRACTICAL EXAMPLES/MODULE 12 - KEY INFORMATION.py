# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Key Information

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
df = spark.sql('select * from openai.bbc_news_data').limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Request to API

# COMMAND ----------

# create prompt
prompt_prefix = """ 
  Extract key information from this text
"""

prompt = prompt_prefix + df['title'].loc[0] + "\n" + df['content'].loc[0]
print(prompt)

# COMMAND ----------

# Request API
response = openai.Completion.create(
  deployment_id="text-davinci-003", 
  prompt=prompt,
  temperature=0,
  max_tokens=1000,
  top_p=0.95,
  frequency_penalty=1,
  presence_penalty=1
)

# print response
print(response['choices'][0]['text'])

# COMMAND ----------

colname = 'key_info'
results = pd.DataFrame(columns=[colname], index=df.index)

prompt_prefix = """ 
  Extract key information from this text
"""

for idx, title, content in zip(df.index.values, df['title'].loc[df.index.values], df['content'].loc[df.index.values]):
  
  # build prompt
  prompt = prompt_prefix + title + "\n" + content

  try:
    # Request API
    response = openai.Completion.create(
      deployment_id="text-davinci-003", 
      prompt=prompt,
      temperature=0,
      max_tokens=1000,
      top_p=0.95,
      frequency_penalty=1,
      presence_penalty=1
    )

      # response
    results[colname].loc[idx] = response['choices'][0]['text']
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
df_results.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("openai.document_analysis_key_information")
df_results = spark.sql('select * from openai.document_analysis_predictions')
df_results.limit(10).toPandas()

# COMMAND ----------

df_results.cache().count()

# COMMAND ----------


