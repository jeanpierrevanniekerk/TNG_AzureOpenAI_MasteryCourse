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
