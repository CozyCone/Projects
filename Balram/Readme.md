
<p align="center">
<b></b>
</p>

<p align=center>
<a href="https://gitHub.com/Haste171/langchain-chatbot/graphs/commit-activity"><img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">
</a>

<!-- *The LangChain Chatbot is an AI chat interface for the open-source library LangChain. It provides conversational answers to questions about vector ingested documents.* -->
<!-- *Existing repo development is at a freeze while we develop a langchain chat bot website :)* -->


# 🚀 Installation

### Developer-setup
Prerequisites:
- [Cohere API Key](https://dashboard.cohere.com/api-keys) - Trial Keys used in this (can also go for the paid production keys)


Reference to create `.env` file
```python
COHERE_API_KEY = 'Your cohere api key'
```

### Create a conda enviornment venv in working folder
```python
conda create -p venv python==3.9 -y
```

### Activate the conda enviornment when in working folder
```python
conda activate venv/
```

### Install required packages from [requirements.txt](https://github.com/CozyCone/Projects/blob/main/Balram/requirements.txt)
```python
pip install -r requirements.txt
```

### Run server
```python
python Balram-server.py
```

### In another terminal run client 
```python 
streamlit run Balram-client.py

```

### Deactivate conda enviornment
```python
conda deactivate
```

# 🔧 Overview about Balram-A Chatbot for Farmers

✅Balram is a chatbot used by farmers for their different queries regarding farming.

✅It provides information about crop management, pest control,general farming queries, including basic queries and recommendation.  

✅Includes features such as Response generation, integration with external APIs, etc.

✅Built using langchain framework.

✅Used Cohere LLM and Cohere Embeddings.

✅ More features coming very soon

Soon:
- Ability to ingest new information from urls entered by user 
- Ability to access real time data like current market prices,weather forecasting,etc.

# Contributing

If you would like to contribute to Balram, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Write tests for your changes
4. Implement your changes and ensure that all tests pass
5. Submit a pull request

# 💻 Interface
![fixed-prev](https://drive.google.com/file/d/1S6A44GtLFjzc4eoaKbA9WMO2Qb3UHuQf/view?usp=sharing)
