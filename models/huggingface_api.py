import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is capital of india?",temperature=0)

print(result.content)