# Databricks notebook source
import time
import requests as req
import json
import gc
import os
import re
import pandas as pd
import os
import pyspark.sql.functions as psf

import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing.pool import ThreadPool
from pyspark.sql.functions import substring
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DateType,DoubleType
from pyspark.sql.functions import concat
from pyspark.sql.functions import col,round
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC # OpenAI with PySpark

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install -r ./requirements.txt

# COMMAND ----------

from synapse.ml.core.platform import find_secret

# Fill in the following lines with your service information
# Learn more about selecting which embedding model to choose: https://openai.com/blog/new-and-improved-embedding-model
service_name = 'tngpocazureopenai-services'
deployment_name = "ChatGPT"
deployment_name_embeddings = "text-embedding-ada-002"
key = os.getenv("OPENAI_API_KEY")


# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/microsoft/SynapseML#databricks

# COMMAND ----------

# MAGIC %md
# MAGIC https://microsoft.github.io/SynapseML/docs/Explore%20Algorithms/OpenAI/

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas Completion

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
pd.options.display.max_columns = 500

# Set up Azure OpenAI
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

df = spark.createDataFrame([("Hello my name is",),("The best code is code thats",),("SynapseML is ",)]).toDF("prompt").toPandas()
df["text"] = np.nan

for idx, title in zip(df.index.values, df['prompt'].loc[df.index.values]):
  # build prompt
  prompt = title

  try:
    # Request API
    response = openai.Completion.create(deployment_id="ChatGPT", prompt=prompt,temperature=0,max_tokens=1000,top_p=0.95,frequency_penalty=1,presence_penalty=1)
    # response
    df['text'].loc[idx] = response['choices'][0]['text']
  except Exception as err:
    idx
    print(f"Unexpected {err=}, {type(err)=}")
df

# COMMAND ----------

# MAGIC %md
# MAGIC # Pyspark Completion

# COMMAND ----------

df = spark.createDataFrame(
    [("Hello my name is",),("The best code is code thats",),("SynapseML is ",)]).toDF("prompt")

from synapse.ml.cognitive import OpenAICompletion

completion = (
    OpenAICompletion()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    .setCustomServiceName(service_name)
    .setMaxTokens(200)
    .setPromptCol("prompt")
    .setErrorCol("error")
    .setOutputCol("completions"))

from pyspark.sql.functions import col
completed_df = completion.transform(df).cache()
display(completed_df.select( col("prompt"),col("error"),col("completions.choices.text").getItem(0).alias("text")))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas ChatCompletion

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pyspark.sql import Row

def make_message(role, content):
    return Row(role=role, content=content, name=role)

load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

chat_df = spark.createDataFrame(
    [([make_message("system", "You are an AI chatbot with red as your favorite color"),make_message("user", "Whats your favorite color"),],),
     ([make_message("system", "You are very excited"),make_message("user", "How are you today"),],),]).toDF("messages").toPandas()
chat_df["context"] = np.nan

for idx, messages in zip(chat_df.index.values, chat_df['messages'].loc[chat_df.index.values]):
    prompt = messages.tolist()
    try:
        response = openai.ChatCompletion.create(engine="ChatGPT", messages=prompt,temperature=0,max_tokens=800,top_p=0.95,frequency_penalty=0,presence_penalty=0,stop=None)
        chat_df['context'].loc[idx] = response['choices'][0]['message']['content']
    except Exception as err:
        idx
        print(f"Unexpected {err=}, {type(err)=}")
display(chat_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pyspark ChatCompletion

# COMMAND ----------

from synapse.ml.cognitive import OpenAIChatCompletion
from pyspark.sql import Row
from pyspark.sql.types import *


chat_df = spark.createDataFrame(
    [([make_message("system", "You are an AI chatbot with red as your favorite color"),
        make_message("user", "Whats your favorite color"),],),
     ([make_message("system", "You are very excited"),
        make_message("user", "How are you today"),],),]).toDF("messages")

chat_completion = (
    OpenAIChatCompletion()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    .setCustomServiceName(service_name)
    .setMessagesCol("messages")
    .setErrorCol("error")
    .setOutputCol("chat_completions"))

display(chat_completion.transform(chat_df).select("messages", "chat_completions.choices.message.content"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Python Batch Completion

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

batch_df = spark.createDataFrame(
    [(["The time has come", "Pleased to", "Today stocks", "Here's to"],),
        (["The only thing", "Ask not what", "Every litter", "I am"],),]).toDF("batchPrompt").toPandas()
batch_df["text"] = np.nan

for idx, title in zip(batch_df.index.values, batch_df['batchPrompt'].loc[batch_df.index.values]):
  prompt = title.tolist()

  try:
    response = openai.Completion.create(deployment_id="ChatGPT", prompt=prompt,temperature=0,max_tokens=1000,top_p=0.95,frequency_penalty=1,presence_penalty=1)
    batch_df['text'].loc[idx] = response['choices'][0]['text']
  except Exception as err:
    idx
    print(f"Unexpected {err=}, {type(err)=}")
display(batch_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Pyspark Batch Completions

# COMMAND ----------

batch_df = spark.createDataFrame(
    [(["The time has come", "Pleased to", "Today stocks", "Here's to"],),
        (["The only thing", "Ask not what", "Every litter", "I am"],),]).toDF("batchPrompt")

batch_completion = (
    OpenAICompletion()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name)
    .setCustomServiceName(service_name)
    .setMaxTokens(200)
    .setBatchPromptCol("batchPrompt")
    .setErrorCol("error")
    .setOutputCol("completions"))

completed_batch_df = batch_completion.transform(batch_df).cache()
display(completed_batch_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Python Embedding Models

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
pd.options.display.max_columns = 500

# Set up Azure OpenAI
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

df = spark.createDataFrame([("Hello my name is",),("The best code is code thats",),("SynapseML is ",)]).toDF("prompt").toPandas()
df["vector"] = np.nan

for idx, title in zip(df.index.values, df['prompt'].loc[df.index.values]):

  prompt = title
  try:
    results = openai.Embedding.create(input=prompt, deployment_id='text-embedding-ada-002')
    df['vector'].loc[idx] = np.array(results['data'][0]['embedding'], dtype=object)
  except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pyspark Embedding Models

# COMMAND ----------

from synapse.ml.cognitive import OpenAIEmbedding

df = spark.createDataFrame([("Hello my name is",),("The best code is code thats",),("SynapseML is ",)]).toDF("prompt")

embedding = (OpenAIEmbedding()
    .setSubscriptionKey(key)
    .setDeploymentName(deployment_name_embeddings)
    .setCustomServiceName(service_name)
    .setTextCol("prompt")
    .setErrorCol("error")
    .setOutputCol("embeddings"))

display(embedding.transform(df))

# COMMAND ----------


