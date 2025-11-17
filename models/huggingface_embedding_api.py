# models/huggingface_embedding_api.py
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # ✅ The new home
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",  
)

text = "My name is Prakash Nandaniya"

result = embeddings.embed_query(text)

print(result)

document=["This is a test document for embedding.", "LangChain is great for building applications with LLMs.", "HuggingFace provides a variety of models for different tasks."]

multiple_results = embeddings.embed_documents(document)

print(multiple_results)