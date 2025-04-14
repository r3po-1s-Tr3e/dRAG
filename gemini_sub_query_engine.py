import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
gemini_llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

# Set up local embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Configure global settings
Settings.llm = gemini_llm
Settings.embed_model = embed_model

# Load documents
communication_doc = SimpleDirectoryReader(
    input_files=["data/Communication.pdf"]
).load_data()
counter_surveillance_doc = SimpleDirectoryReader(
    input_files=["data/Counter_Surveillance.pdf"]
).load_data()
covert_operations_doc = SimpleDirectoryReader(
    input_files=["data/Covert_Operations.pdf"]
).load_data()
emergency_directives_doc = SimpleDirectoryReader(
    input_files=["data/Emergency_Directives.pdf"]
).load_data()
high_risk_protocol_doc = SimpleDirectoryReader(
    input_files=["data/High_Risk_Protocol.pdf"]
).load_data()
safe_houses_doc = SimpleDirectoryReader(
    input_files=["data/Safe_Houses.pdf"]
).load_data()

# Create indices
communication_index = VectorStoreIndex.from_documents(communication_doc)
counter_surveillance_index = VectorStoreIndex.from_documents(counter_surveillance_doc)
covert_operations_index = VectorStoreIndex.from_documents(covert_operations_doc)
emergency_directives_index = VectorStoreIndex.from_documents(emergency_directives_doc)
high_risk_protocol_index = VectorStoreIndex.from_documents(high_risk_protocol_doc)
safe_houses_index = VectorStoreIndex.from_documents(safe_houses_doc)

# Create query engines
communication_engine = communication_index.as_query_engine(similarity_top_k=3)
counter_surveillance_engine = counter_surveillance_index.as_query_engine(similarity_top_k=3)
covert_operations_engine = covert_operations_index.as_query_engine(similarity_top_k=3)
emergency_directives_engine = emergency_directives_index.as_query_engine(similarity_top_k=3)
high_risk_protocol_engine = high_risk_protocol_index.as_query_engine(similarity_top_k=3)
safe_houses_engine = safe_houses_index.as_query_engine(similarity_top_k=3)

# Define query engine tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=communication_engine,
        metadata=ToolMetadata(
            name="communication",
            description="Provides information about communication protocols and procedures",
        ),
    ),
    QueryEngineTool(
        query_engine=counter_surveillance_engine,
        metadata=ToolMetadata(
            name="counter_surveillance",
            description="Provides information about counter-surveillance techniques and strategies",
        ),
    ),
    QueryEngineTool(
        query_engine=covert_operations_engine,
        metadata=ToolMetadata(
            name="covert_operations",
            description="Provides information about covert operations and tactics",
        ),
    ),
    QueryEngineTool(
        query_engine=emergency_directives_engine,
        metadata=ToolMetadata(
            name="emergency_directives",
            description="Provides information about emergency directives and protocols",
        ),
    ),
    QueryEngineTool(
        query_engine=high_risk_protocol_engine,
        metadata=ToolMetadata(
            name="high_risk_protocol",
            description="Provides information about high-risk operation protocols",
        ),
    ),
    QueryEngineTool(
        query_engine=safe_houses_engine,
        metadata=ToolMetadata(
            name="safe_houses",
            description="Provides information about safe house locations and protocols",
        ),
    ),
]

# Initialize SubQuestion Query Engine with Gemini LLM
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=gemini_llm,
    use_async=True,
    verbose=True
)

# Run the query
response = query_engine.query(
    "Compare and contrast the protocols for emergency directives and high-risk operations. Give your response in form of a poem"
)

print("-------------------------------")
print(response)