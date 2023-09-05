# Databricks notebook source
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

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
COMPLETION_MODEL = 'text-davinci-003'

# COMMAND ----------

# MAGIC %md
# MAGIC # Write a basic function using NLP

# COMMAND ----------

openai.api_version = "2023-07-01-preview"
response = openai.Completion.create(engine="ChatGPT",
                                    prompt="write a function for binary search in python. Also explain the logic behind this function",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Change Coding Language

# COMMAND ----------

openai.api_version = "2023-07-01-preview"
response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"def print_squares(n):for i in range(1,n+1):print(i**2) convert this python function to C# code"}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Understand Unknown Code

# COMMAND ----------


openai.api_version = "2023-07-01-preview"
response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"def print_squares(n):for i in range(1,n+1):print(i**2) please explain what this code does"}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Complete code and assist with development

# COMMAND ----------

openai.api_version = "2023-07-01-preview"
response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"complete the following function --- #calculate the average of the numbers in an array, but only if they are even. def"}],
  temperature=0,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Unit Tests

# COMMAND ----------


openai.api_version = "2023-07-01-preview"
response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Write 3 unit tests for this function def calculate_average_even(numbers):\n    # Initialize variables\n    count = 0\n    total = 0\n    \n    # Loop through the numbers\n    for number in numbers:\n        # Check if the number is even\n        if number % 2 == 0:\n            count += 1\n            total += number\n    \n    # Calculate the average if there are even numbers\n    if count > 0:\n        average = total / count\n        return average\n    else:\n        return None"},],
  temperature=0,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Add comments to code and create documents

# COMMAND ----------

response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"--- def binary_search(arr, low, high, x):\n    if high >= low:\n        mid = (high + low) // 2\n        if arr[mid] == x:\n            return mid\n        elif arr[mid] > x:\n            return binary_search(arr, low, mid - 1, x)\n        else:\n            return binary_search(arr, mid + 1, high, x)\n    else:\n        return -1 --- Add comments to the above code"}],
  temperature=0,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------


