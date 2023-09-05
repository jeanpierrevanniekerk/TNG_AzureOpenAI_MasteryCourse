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

# COMMAND ----------

# MAGIC %md
# MAGIC # Redis Example

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install redis
# MAGIC !pip install redis-enterprise-python
# MAGIC !pip install feedparser

# COMMAND ----------

import redis
import openai
import os
import requests
from bs4 import BeautifulSoup
import feedparser
import numpy as np 
 
# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
 
# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = '10000'
redis_password = os.getenv('REDIS_PASSWORD')
 
# Connect to the Redis server
conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password)#, encoding='utf-8', decode_responses=True)
if conn.ping():
    print("Connected to Redis")
 
# URL of the RSS feed to parse
url = 'https://blog.baeke.info/feed/'
 
# Parse the RSS feed with feedparser
feed = feedparser.parse(url)
 
p = conn.pipeline(transaction=False)
for i, entry in enumerate(feed.entries[:50]):
    # report progress
    print("Create embedding and save for entry ", i, " of ", entry)
 
    r = requests.get(entry.link)
    soup = BeautifulSoup(r.text, 'html.parser')
    article = soup.find('div', {'class': 'entry-content'}).text
 
    #vectorize with OpenAI text-emebdding-ada-002
    embedding = openai.Embedding.create(input=article,model="text-embedding-ada-002")

    # print the embedding (length = 1536)
    vector = embedding["data"][0]["embedding"]
 
    # convert to numpy array and bytes
    vector = np.array(vector).astype(np.float32).tobytes()
 
    # Create a new hash with url and embedding
    post_hash = {"url": entry.link,"embedding": vector}
 
    # create hash
    conn.hset(name=f"post:{i}", mapping=post_hash)
p.execute()

# COMMAND ----------

#To create an index with Python code, check the code below:
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
 
# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')
 
# Connect to the Redis server
conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)
 
SCHEMA = [TextField("url"),VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"})]
 
# Create the index
try:
    conn.ft("posts").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["post:"], index_type=IndexType.HASH))
except Exception as e:
    print("Index already exists")

# COMMAND ----------

import numpy as np
from redis.commands.search.query import Query
import redis
import openai
import os
 
def search_vectors(query_vector, client, top_k=5):
    base_query = "*=>[KNN 5 @embedding $vector AS vector_score]"
    query = Query(base_query).return_fields("url", "vector_score").sort_by("vector_score").dialect(2)    
 
    try:
        results = client.ft("posts").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None
 
    return results
 
# Connect to the Redis server
conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)
 
if conn.ping():
    print("Connected to Redis")
 
# Enter a query
query = input("Enter your query: ")
 
# Vectorize the query using OpenAI's text-embedding-ada-002 model
print("Vectorizing query...")
embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")
query_vector = embedding["data"][0]["embedding"]
 
# Convert the vector to a numpy array
query_vector = np.array(query_vector).astype(np.float32).tobytes()
 
# Perform the similarity search
print("Searching for similar posts...")
results = search_vectors(query_vector, conn)
 
if results:
    print(f"Found {results.total} results:")
    for i, post in enumerate(results.docs):
        score = 1 - float(post.vector_score)
        print(f"\t{i}. {post.url} (Score: {round(score ,3) })")
else:
    print("No results found")

# COMMAND ----------


