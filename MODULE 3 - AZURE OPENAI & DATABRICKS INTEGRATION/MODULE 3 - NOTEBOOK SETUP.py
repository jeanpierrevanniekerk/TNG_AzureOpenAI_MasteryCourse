# Databricks notebook source
# MAGIC %md
# MAGIC # TEST NOTEBOOK SETUP

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

print('Test')

# COMMAND ----------


