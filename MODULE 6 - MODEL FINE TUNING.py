# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install -r ./requirements.txt

# COMMAND ----------

!pip install requests
!pip install pillow 

# COMMAND ----------

import os
import openai
from dotenv import load_dotenv

# Set up Azure OpenAI
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_WESTEU_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY_WESTEU")

# COMMAND ----------

# MAGIC %md
# MAGIC # Python SDK

# COMMAND ----------

import openai
from openai import cli
import time
import shutil
import json

# Remember to remove your key from your code when you're done.
openai.api_type = 'azure'
# The API version may change in the future.
openai.api_version = '2023-05-15'

training_file_name = 'training.jsonl'
validation_file_name = 'validation.jsonl'

sample_data = [{"prompt": "When I go to the store, I want an", "completion": "apple"},
               {"prompt": "When I go to work, I want a", "completion": "coffee"},
                {"prompt": "When I go home, I want a", "completion": "soda"}]

# Generate the training dataset file.
print(f'Generating the training file: {training_file_name}')
with open(training_file_name, 'w') as training_file:
    for entry in sample_data:
        json.dump(entry, training_file)
        training_file.write('\n')

# Copy the validation dataset file from the training dataset file.
print(f'Copying the training file to the validation file')
shutil.copy(training_file_name, validation_file_name)

def check_status(training_id, validation_id):
    train_status = openai.File.retrieve(training_id)["status"]
    valid_status = openai.File.retrieve(validation_id)["status"]
    print(f'Status (training_file | validation_file): {train_status} | {valid_status}')
    return (train_status, valid_status)

# Upload the training and validation dataset files to Azure OpenAI.
training_id = cli.FineTune._get_or_upload(training_file_name, False)
validation_id = cli.FineTune._get_or_upload(validation_file_name, False)

# Check on the upload status of the training and validation dataset files.
(train_status, valid_status) = check_status(training_id, validation_id)

# Poll and display the upload status once a second until both files have either succeeded or failed to upload.
while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
    time.sleep(1)
    (train_status, valid_status) = check_status(training_id, validation_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #Create a customized model

# COMMAND ----------

# This example defines a fine-tune job that creates a customized model based on curie, 
# with just a single pass through the training data. The job also provides classification-
# specific metrics, using our validation data, at the end of that epoch.
create_args = {
    "training_file": training_id,
    "validation_file": validation_id,
    "model": "ada",
    "compute_classification_metrics": True,
    "classification_n_classes": 3
}
# Create the fine-tune job and retrieve the job ID
# and status from the response.
resp = openai.FineTune.create(**create_args)
job_id = resp["id"]
status = resp["status"]

# You can use the job ID to monitor the status of the fine-tune job.
# The fine-tune job may take some time to start and complete.
print(f'Fine-tuning model with job ID: {job_id}.')

# COMMAND ----------

# MAGIC %md
# MAGIC Check the status of your customized model

# COMMAND ----------

# Get the status of our fine-tune job.
status = openai.FineTune.retrieve(id=job_id)["status"]

# If the job isn't yet done, poll it every 2 seconds.
if status not in ["succeeded", "failed"]:
    print(f'Job not in terminal status: {status}. Waiting.')
    while status not in ["succeeded", "failed"]:
        time.sleep(2)
        status = openai.FineTune.retrieve(id=job_id)["status"]
        print(f'Status: {status}')
else:
    print(f'Fine-tune job {job_id} finished with status: {status}')

# Check if there are other fine-tune jobs in the subscription. 
# Your fine-tune job may be queued, so this is helpful information to have
# if your fine-tune job hasn't yet started.
print('Checking other fine-tune jobs in the subscription.')
result = openai.FineTune.list()
print(f'Found {len(result)} fine-tune jobs.')

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy a model with Azure OpenAI

# COMMAND ----------

result = openai.FineTune.list()
result

# COMMAND ----------

# Retrieve the name of the customized model from the fine-tune job.
result = openai.FineTune.retrieve(id=job_id)
if result["status"] == 'succeeded':
    model = result["fine_tuned_model"]
    print(model)

# Create the deployment for the customized model, using the standard scale type without specifying a scale
# capacity.
print(f'Creating a new deployment with model: {model}')
result = openai.Deployment.create(model=model, scale_settings={"scale_type":"standard", "capacity": None})
# Retrieve the deployment job ID from the results.
deployment_id = result["id"]

# COMMAND ----------

print('Sending a test completion job')
start_phrase = 'When I go to the store, I want a'
response = openai.Completion.create(engine=deployment_id, prompt=start_phrase, max_tokens=4)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(f'"{start_phrase} {text}"')

# COMMAND ----------

# MAGIC %md
# MAGIC # Analyze your customized model

# COMMAND ----------

# Retrieve the file ID of the first result file from the fine-tune job for
# the customized model.
result = openai.FineTune.retrieve(id=job_id)
if result["status"] == 'succeeded':
    result_file_id = result.result_files[0].id
    result_file_name = result.result_files[0].filename

# Download the result file.
print(f'Downloading result file: {result_file_id}')
# Write the byte array returned by the File.download() method to 
# a local file in the working directory.
with open(result_file_name, "wb") as file:
    result = openai.File.download(id=result_file_id)
    file.write(result)

# COMMAND ----------



# COMMAND ----------


