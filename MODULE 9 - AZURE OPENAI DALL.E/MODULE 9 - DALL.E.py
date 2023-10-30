# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install requests
# MAGIC !pip install pillow

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
# MAGIC # Python REST API

# COMMAND ----------

import requests
import time
import os
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
api_version = '2023-06-01-preview'
url = f"{api_base}openai/images/generations:submit?api-version={api_version}"
headers= { "api-key": api_key, "Content-Type": "application/json" }
body = {
    "prompt": "a multi-colored umbrella on the beach, disposable camera",
    "size": "1024x1024",
    "n": 1
}
submission = requests.post(url, headers=headers, json=body)

operation_location = submission.headers['operation-location']
status = ""
while (status != "succeeded"):
    time.sleep(1)
    response = requests.get(operation_location, headers=headers)
    status = response.json()['status']
image_url = response.json()['result']['data'][0]['url']

# COMMAND ----------

image_url

# COMMAND ----------

# MAGIC %md
# MAGIC # Python SDK

# COMMAND ----------

# !pip install --upgrade openai

# COMMAND ----------


import openai
import os
import requests
from PIL import Image

openai.api_base = os.getenv("OPENAI_API_BASE") # Add your endpoint here
openai.api_key = os.getenv("OPENAI_API_KEY")  # Add your api key here

# At the moment Dall-E is only supported by the 2023-06-01-preview API version
openai.api_version = '2023-06-01-preview'

openai.api_type = 'azure'

# Create an image using the image generation API
generation_response = openai.Image.create(
    prompt='A painting of a dog',
    size='1024x1024',
    n=2
)

# Set the directory where we'll store the image
image_dir = os.path.join(os.curdir, 'images')
# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# With the directory in place, we can initialize the image path (note that filetype should be png)
image_path = os.path.join(image_dir, 'generated_image.png')

# COMMAND ----------

# Now we can retrieve the generated image
image_url = generation_response["data"][0]["url"]  # extract image URL from response
generated_image = requests.get(image_url).content  # download the image
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)

# Display the image in the default image viewer
display(Image.open(image_path))

# COMMAND ----------

# Now we can retrieve the generated image
image_url = generation_response["data"][1]["url"]  # extract image URL from response
generated_image = requests.get(image_url).content  # download the image
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)

# Display the image in the default image viewer
display(Image.open(image_path))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


