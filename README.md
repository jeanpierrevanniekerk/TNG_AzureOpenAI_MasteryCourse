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

