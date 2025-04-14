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

from llama_index.question_gen.guidance import GuidanceQuestionGenerator

# Code for when you get open ai api

# from guidance import models

# GuidanceOpenAI = models.OpenAI("gpt-3.5-turbo", api_key="your-openai-api-key")

# from llama_index.question_gen.guidance import GuidanceQuestionGenerator
# question_gen = GuidanceQuestionGenerator.from_defaults(
#     guidance_llm=GuidanceOpenAI, verbose=False
# )

question_gen = GuidanceQuestionGenerator.from_defaults(
    guidance_llm=gemini_llm, verbose=False
)

from llama_index.core.tools import ToolMetadata
from llama_index.core import QueryBundle

tools = [
    ToolMetadata(
        name="lyft_10k",
        description="Provides information about Lyft financials for year 2021",
    ),
    ToolMetadata(
        name="uber_10k",
        description="Provides information about Uber financials for year 2021",
    ),
]

sub_questions = question_gen.generate(
    tools=tools,
    query=QueryBundle("Compare and contrast Uber and Lyft financial in 2021"),
)

print(sub_questions)