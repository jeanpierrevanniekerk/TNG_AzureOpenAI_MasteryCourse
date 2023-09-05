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
      - maven coordinates: com.microsoft.azure:synapseml_2.12:0.9.5
      - Repository: https://mmlspark.azureedge.net/maven
   - Ensure you have the requirements.txt file in your home directory to install the relevant libraries when running the notebooks.
   - Setup .ini file containing the required key and base for the Azure OpenAI subscription.
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
    - curl https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/completions?api-version=2022-12-01 -H "Content-Type: application/json" -H "Authorization: Bearer $accessToken" -d '{ "prompt":"When I go to the store, I want an ","max_tokens":500}'
  

# Module 7 – Embedding Models
* Introduction to Azure OpenAI Embedding Models. Theoretical content contained within the slide deck.
1) Exercise 1 – Setup Redis for cache database and utilise Embedding models:
   - Create an Enterprise Redis for Cache database on the Azure Portal.
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
   - curl -i -X POST https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/extensions/chat/completions?api-version=2023-06-01-preview -H "Content-Type: application/json" -H "api-key: 7079b53b72df4f04bf94a302697561e9" -H "chatgpt_url: https://tngpocazureopenai-services.openai.azure.com/openai/deployments/ChatGPT/extensions/chat/completions?api-version=2023-06-01-preview" -H "chatgpt_key: 7079b53b72df4f04bf94a302697561e9" -d '{"dataSources": [{"type": "AzureCognitiveSearch","parameters":{"endpoint":"https://tngcognitivesearch.search.windows.net/indexes/useyourowndata/docs?api-version=2023-07-01-Preview&search=*","key":"n9ZqMO9M3zdLfpImh30FI9JFV2k8vhc0mTdhLFNRQfAzSeD9y1Ej","indexName": "useyourowndata"}}],"messages": [{"role": "user","content": "Is there a module that touches on Pandas code?"}]}'

