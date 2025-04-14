from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# from llama_index.llms.gemini import Gemini
# llm = Gemini(
#     model="models/gemini-2.0-flash-thinking-exp-01-21",
# )

from llama_index.llms.google_genai import GoogleGenAI

gemini_llm = GoogleGenAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
)


# Code for when you get open ai api

# from guidance import models

# GuidanceOpenAI = models.OpenAI("gpt-3.5-turbo", api_key="your-openai-api-key")

# from llama_index.question_gen.guidance import GuidanceQuestionGenerator
# question_gen = GuidanceQuestionGenerator.from_defaults(
#     guidance_llm=GuidanceOpenAI, verbose=False
# )
# tools = [
#     ToolMetadata(
#         name="lyft_10k",
#         description="Provides information about Lyft financials for year 2021",
#     ),
#     ToolMetadata(
#         name="uber_10k",
#         description="Provides information about Uber financials for year 2021",
#     ),
# ]


# query_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=tools,
#     use_async=True,
# )
# sub_questions = question_gen.generate(
#     tools=tools,
#     query=QueryBundle("Compare and contrast Uber and Lyft financial in 2021"),
# )

# print(sub_questions)

from llama_index.core.tools import ToolMetadata
from llama_index.core import QueryBundle
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings



lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print(response)