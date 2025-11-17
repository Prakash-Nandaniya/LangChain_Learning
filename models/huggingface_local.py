import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        temperature=0.1,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is capital of india?",temperature=0)

print(result.content)