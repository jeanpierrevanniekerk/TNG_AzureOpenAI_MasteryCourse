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
# MAGIC # Provide Clear Instructions

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="write a product description for a new water bottle",
                                    temperature=0,
                                    max_tokens=500)
print(response.choices[0].text)

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="write a product description for a new water bottle that is 100% recycled. Be sure to include that it comes in natural colors with no dyes, and each purchase removed 10 pounds of plastic from our oceans",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Section Markers

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="Translate the text into French --- What's the weather going to be like today? ---",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Primary and supporting content

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="--- Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. Reinforcement learning differs from supervised learning in not needing labelled input/output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).[1] The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context use dynamic programming techniques.[2] The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible. --- Summarize this article and identify three takeaways in a bulleted fashion.",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="--- Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. Reinforcement learning differs from supervised learning in not needing labelled input/output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).[1] The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context use dynamic programming techniques.[2] The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.  ---                                     \
                                    Topics I am very interested in includes AI, Pros to RL algorithms.00-explore-data\
                                    \
                                    Summarize this article and identify three takeaways in a bulleted fashion.",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cues

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="write a join query to get customer names with purchases in the past year. SELECT",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Request Output Composition

# COMMAND ----------

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt="Write a table in markdown with 6 animals in it, with their genus and color included.",
                                    temperature=0,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

openai.api_version = "2023-07-01-preview"
response = openai.Completion.create(engine="ChatGPT",
                                    prompt="Put fictional characters into JSON of the following format. {firstNameFIctional: jobFictional:}.",
                                    temperature=1,
                                    max_tokens=1000)
print(response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Few Shot Learning

# COMMAND ----------

response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are a marketing writing assistant. You help come up with creative content ideas and content like marketing emails, blog posts, tweets, ad copy and product descriptions. You write in a friendly yet professional tone but can tailor your writing style that best works for a user-specified audience. If you do not know the answer to a question, respond by saying \"I do not know the answer to your question.\""},{"role":"user","content":"What type of assistant are you?"}],
  temperature=1,
  max_tokens=400,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Chain of thought

# COMMAND ----------

response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"what sport is the easiest to learn but hardest to master?"}],
  temperature=0,
  max_tokens=800,
  top_p=0,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------

response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"what sport is the easiest to learn but hardest to master? Explain step-by-step approach of your thoughts, ending in your answer"}],
  temperature=0,
  max_tokens=800,
  top_p=0,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)

# COMMAND ----------


