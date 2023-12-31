# OpenAI Master Class
Make sure you have access to the following Azure resources within your tenant:
* Azure Open AI (Have the necessary quotas available to spin up models).
* Azure Databricks (Alternatively local anaconda or vs-code with python installed).
* Azure DevOps.
* Azure Cognitive Search.
* Azure Redis for Cache.


# Module 1 – OpenAI and Generative AI
* Introduction to Azure OpenAI. Theoretical content contained within the slide deck.


# Module 2 – Azure OpenAI studio Playground
* Introduction to Azure OpenAI Studio Playground. Theoretical content contained within the slide deck.
1) Exercise 1 – Setup and login to Azure OpenAI studio:
   - Setup Azure OpenAI studio.
3) Exercise 2 – Interface with the Azure OpenAI studio Completion Playground:
   - Deploy and interface with the Azure Completion Playground.
3) Exercise 3 – Interface with Azure OpenAI studio Chat Playground:
   - Deploy and interface with the Azure Chat Playground.
4) Exercise 4 – Interface with DALL.E Playground:
   - Test the DALL.E playground and its functionality.


# Module 3 – Azure OpenAI integration with Databricks
* Introduction to Azure OpenAI and Databricks Integration. Theoretical content contained within the slide deck.
1) Exercise 1 – Setup Databricks Environment:
   - Launch Databricks Workspace.
   - Create local library using the following settings:
      - maven coordinates: com.microsoft.azure:synapseml_2.12:0.11.2
      - Repository: https://mmlspark.azureedge.net/maven
   - Ensure you have the requirements.txt file in your home directory to install the relevant libraries when running the notebooks.
   - Setup .ini file containing the required key and base for the Azure OpenAI subscription.
   - Setup the .env file with the required key and base for the Azure OpenAI subscription.
   - Create compute on databricks and install the following libraries:
      - SynapseML
      - com.microsoft.azure:synapseml_2.12:0.11.2
      - com.redislabs:spark-redis:2.3.0
   - Use runtime: 13.3 LTS ML (Scala 2.12, Spark 3.4.1)
   - Also reference the .ini file in your compute cluster.
   - Ensure you have Cognitive Services Usages Reader access is enabled as seen below within the Azure OpenAI workspace:
      - ![image](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/assets/139854126/51b91f46-bd5a-4748-ba98-357088691fb6)
   - Reference Module 3 notebook on the Git Repository to test the connection.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%203%20-%20AZURE%20OPENAI%20%26%20DATABRICKS%20INTEGRATION


# Module 4 – APIs and SDKs
* Introduction to Azure OpenAI APIs and SDKs. Theoretical content contained within the slide deck.
1) Exercise 1 – Login to Azure CLI and then interact with an Azure OpenAI LLM:
   - Make sure you have the necessary permissions to access the Azure CLI
      - az login
      - export accessToken=$(az account get-access-token --resource https://cognitiveservices.azure.com | jq -r .accessToken)
      - curl https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/completions?api-version=2022-12-01 -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{ "prompt": "Tell me a funny story.", "max_tokens":5 }'
      - curl https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/completions?api-version=2022-12-01 -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{ "prompt": "Tell me a funny story.", "max_tokens":500}'

2) Exercise 2 – Access Azure OpenAI LLM functionality using Python SDK:
   - Access Module 4: SDK notebook on the Git Repository.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%204%20-%20AZURE%20OPENAI%20APIs%20%26%20SDKs


# Module 5 – Prompt Engineering
* Introduction to Azure OpenAI Prompt Engineering. Theoretical content contained within the slide deck.
1) Exercise 1 – Test various prompt engineering techniques:
   - Access Module 5: Prompt Engineering Notebook on Git Repository.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%205%20-%20AZURE%20OPENAI%20PROMPT%20ENGINEERING


# Module 6 – Model Fine Tuning
* Introduction to Azure OpenAI Model Fine Tuning. Theoretical content contained within the slide deck.
1) Exercise 1 – Use general model which has not been fine-tuned using CLI:
   - curl https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/completions?api-version=2022-12-01 -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{ "prompt":"When I go to the store, I want an","max_tokens":500}'
2) Exercise 2 – Setup fine-tuned model using Azure OpenAI Python SDK:
    - Access Module 6: Model Fine Tuning Notebook on Git Repository.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%206%20-%20AZURE%20OPENAI%20MODEL%20FINE%20TUNING
    - Setup model and train the model.
3) Exercise 3 – Use fine-tuned model using CLI:
    - curl custom_model_deployment_url -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{ "prompt":"When I go to the store, I want an ","max_tokens":500}'
  

# Module 7 – Embedding Models
* Introduction to Azure OpenAI Embedding Models. Theoretical content contained within the slide deck.
1) Exercise 1 – Setup Redis for cache database and utilise Embedding models:
   - Create an Enterprise Redis for Cache database on the Azure Portal.
   - ENsure that the TLS option is enabled for the Redis for cache database.
   - Download RedisInsights to your local machine and connect the Redis Azure database by using the provided "endpoint", "keys" and "port".​
     - ![image](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/assets/139854126/913b9034-22a7-4f3d-a3b9-6ed0c9e5e251)
   - Retrieve the Redis Host and Password once the database has been provisioned and insert it into the init.sh file.
   - Access Module 7: Embedding Models Notebook to write to Redis for cache database.
      - Also run embedding models to embed text into vectors.
      - Analyse and evaluate output.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%207%20-%20AZURE%20OPENAI%20EMBEDDING%20MODELS


# Module 8 – Codex Models
* Introduction to Azure OpenAI Codex Models. Theoretical content contained within the slide deck.
1) Exercise 1 – Run some code queries to test codex capabilities:
   - Access Module 8: Codex Notebook to run through the various techniques of how the codex models can be used to improve coding productivity and performance.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%208%20-%20AZURE%20OPENAI%20CODEX


# Module 9 – DALL.E
* Introduction to Azure OpenAI DALL.E Models. Theoretical content contained within the slide deck.
1) Exercise 1 – Generate an image using the Azure CLI:
   - curl https://tngpocazureopenai-services.openai.azure.com/openai/images/generations:submit?api-version=2023-06-01-preview -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{"prompt": "An avocado chair","size": "512x512","n": 3,"response_format": "url"}'
2) Exercise 2 – Retrieve an image using the Azure CLI:
    - curl -X GET "https://tngpocazureopenai-services.openai.azure.com/openai/operations/images/88ef2a2e-9a18-497b-988a-eecd86132dbb?api-version=2023-06-01-preview" -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken"
3) Exercise 3 – Generate an image using the Azure Python SDKs:
   - Access Module 9: DALL.E Notebook allowing the user to generate images using the Python SDK.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%209%20-%20AZURE%20OPENAI%20DALL.E
    

# Module 10 – Grounding your model using your own data
* Introduction to Azure OpenAI Grouding Models. Theoretical content contained within the slide deck.
1) Exercise 1 – Use the Azure OpenAI Studio to ground a model:
   - Generate a text file and copy some text in there.
   - Upload it to the Azure OpenAI Studio during model grounding.
   - Ask questions related to the text in the text file.
2) Exercise 2 – Use the Azure CLI to access and interface with the grounding model:
   - curl -i -X POST https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/extensions/chat/completions?api-version=2023-06-01-previe -H "Content-Type: application/json" -H "api-key: key" -H "chatgpt_url: url" -H "chatgpt_key: key" -d '{"dataSources": [{"type": "AzureCognitiveSearch","parameters":{"endpoint":"url","key":"key","indexName": "useyourowndata"}}],"messages": [{"role": "user","content": "Is there a module that touches on Pandas code?"}]}'



# MODULE 11 – Pandas vs Pyspark with Azure OpenAI
* Introduction to Azure OpenAI Pandas vs Pyspark. Theoretical content contained within the slide deck.
1) Exercise 1 – Identify how one would use Pandas and Pyspark to interface with the Azure OpenAI SDKs:
   - Access Module 11: Pandas vs Pyspark Notebook to get a view of how one would leverage Pyspark to scale these LLM solutions.
      - https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/tree/main/MODULE%2011%20-%20AZURE%20OPENAI%20PANDAS%20VS%20PYSPARK


# MODULE 12 – Azure OpenAI Practical Examples
* Introduction to Azure OpenAI Practical Examples. Theoretical content contained within the slide deck.
1) Exercise 1 – Use the Azure OpenAI Example Notebooks to get a good understanding of some practical examples:
   - Access Module 12: Example notebooks to get a view of how we would practically implement the Azure OpenAI LLMs:
      - Data Exploration and Embeddings.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - DATA EXPLORATION.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20DATA%20EXPLORATION.py)
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - EMBEDDINGS.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20EMBEDDINGS.py)
      - Visualize Embeddings and Classification Documents.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - VISUALIZE EMBEDDING.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20VISUALIZE%20EMBEDDING.py)
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - CLASSIFICATION DOCUMENTS.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20CLASSIFICATION%20DOCUMENTS.py)
      - Document Summarization.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - DOCUMENT SUMMARIZATION.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20DOCUMENT%20SUMMARIZATION.py)
      - Key Information.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - KEY INFORMATION.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20KEY%20INFORMATION.py)
      - Key Word Extraction.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - KEY WORD EXTRACTION.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20KEY%20WORD%20EXTRACTION.py)
      - Semantic Search.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - SEMANTIC SEARCH.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20SEMANTIC%20SEARCH.py)
      - Information Retrieval.
         - [TNG_AzureOpenAI_MasteryCourse/MODULE 12 - PRACTICAL EXAMPLES/MODULE 12 - INFORMATION RETRIEVAL.py at main · jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse (github.com)](https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20INFORMATION%20RETRIEVAL.py)https://github.com/jeanpierrevanniekerk/TNG_AzureOpenAI_MasteryCourse/blob/main/MODULE%2012%20-%20PRACTICAL%20EXAMPLES/MODULE%2012%20-%20INFORMATION%20RETRIEVAL.py


# MODULE 13 – Azure OpenAI MLOps
* Introduction to Azure OpenAI MLOPs using Databricks and Azure DevOps. Theoretical content contained within the slide deck.
1) Exercise 1 – Take the “Data Exploration” notebook through the MLOps lifecycle:
   - Create 2 Databricks Environments (1 Dev and 1 Prod).
   - Setup an Azure DevOps repository.
   - Link the repository with your Dev Databricks Environment.
   - Setup the Azure DevOps pipelines and releases.
   - Push Dev notebook to Prod Databricks Environment.
   - Setup scheduling and notification functionalities.


# MODULE 14 – Advanced Use Cases
* Getting some exposure to advances Azure OpenAI use cases. Theoretical content contained within the slide deck.


# MODULE 15 – Summary and Conclusion
* Summarizing the content covered in the Azure OpenAI Mastery course.
* Discussion potential next steps and how TNG can help expedite Azure OpenAI implementations in your organisation.


