from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"   
)

document=[
    "india has a rich history spanning ancient civilizations like the Indus Valley, great empires such as the Maurya and Gupta, and colonial rule under the British before gaining independence in 1947.",
    "The United States began as British colonies, declared independence in 1776, rapidly grew through industrialization and immigration, and became a global superpower after World War II.",
    "Japan's history includes feudal shogunates, isolationist policies during the Edo period, modernization in the Meiji era, and post-World War II economic growth to become a leading global economy.it is in east side of india.",
    "China boasts one of the world's oldest continuous civilizations, with dynasties like the Han, Tang, and Qing shaping its culture, followed by republican and communist periods in the 20th century.",
    "Europe’s history is marked by the rise and fall of empires such as Rome, medieval kingdoms, the Renaissance, Enlightenment, and two world wars that shaped modern political and social structures."
    ]
result_document = embeddings.embed_documents(document)

query="Give me a brief overview of India's history."
result_query = embeddings.embed_query(query)  

print(cosine_similarity([result_query], result_document))
