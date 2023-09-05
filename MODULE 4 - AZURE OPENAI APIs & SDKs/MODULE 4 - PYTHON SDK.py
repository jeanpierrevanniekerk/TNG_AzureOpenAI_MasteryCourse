# Databricks notebook source
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

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
COMPLETION_MODEL = 'text-davinci-003'

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="tell me a funny story",
                                    temperature=1,
                                    max_tokens=5
                                    )
print(response.choices[0].text)

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="tell me a funny story",
                                    temperature=1,
                                    max_tokens=500
                                    )
print(response.choices[0].text)

# COMMAND ----------


