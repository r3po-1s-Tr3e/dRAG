import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from rule_extraction.rule_or_response import get_rule_o_response

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
gemini_llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

# Set up local embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = SentenceSplitter(chunk_size=10000, chunk_overlap=0)

# Configure global settings
Settings.llm = gemini_llm
Settings.embed_model = embed_model

# Cache document loading and index creation for efficiency
@st.cache_resource
def setup_query_engine():
    # Load documents
    communication_doc = SimpleDirectoryReader(input_files=["data/Communication.pdf"]).load_data()
    counter_surveillance_doc = SimpleDirectoryReader(input_files=["data/Counter_Surveillance.pdf"]).load_data()
    covert_operations_doc = SimpleDirectoryReader(input_files=["data/Covert_Operations.pdf"]).load_data()
    emergency_directives_doc = SimpleDirectoryReader(input_files=["data/Emergency_Directives.pdf"]).load_data()
    high_risk_protocol_doc = SimpleDirectoryReader(input_files=["data/High_Risk_Protocol.pdf"]).load_data()
    safe_houses_doc = SimpleDirectoryReader(input_files=["data/Safe_Houses.pdf"]).load_data()

    # Create indices
    communication_index = VectorStoreIndex.from_documents(communication_doc, transformations=[splitter])
    counter_surveillance_index = VectorStoreIndex.from_documents(counter_surveillance_doc, transformations=[splitter])
    covert_operations_index = VectorStoreIndex.from_documents(covert_operations_doc, transformations=[splitter])
    emergency_directives_index = VectorStoreIndex.from_documents(emergency_directives_doc, transformations=[splitter])
    high_risk_protocol_index = VectorStoreIndex.from_documents(high_risk_protocol_doc, transformations=[splitter])
    safe_houses_index = VectorStoreIndex.from_documents(safe_houses_doc, transformations=[splitter])

    # Create query engines
    communication_engine = communication_index.as_query_engine(similarity_top_k=1)
    counter_surveillance_engine = counter_surveillance_index.as_query_engine(similarity_top_k=1)
    covert_operations_engine = covert_operations_index.as_query_engine(similarity_top_k=1)
    emergency_directives_engine = emergency_directives_index.as_query_engine(similarity_top_k=1)
    high_risk_protocol_engine = high_risk_protocol_index.as_query_engine(similarity_top_k=1)
    safe_houses_engine = safe_houses_index.as_query_engine(similarity_top_k=1)

    # Define query engine tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=communication_engine,
            metadata=ToolMetadata(
                name="communication",
                description="Provides information about communication protocols, verification protocols, Layered Cipher Code (LCC) System, Handshake Protocol and Neural Signatures.",
            ),
        ),
        QueryEngineTool(
            query_engine=counter_surveillance_engine,
            metadata=ToolMetadata(
                name="counter_surveillance",
                description="Provides information about counter-surveillance techniques, ghost step algorithm, how to prevent tracking and how to handle system breach.",
            ),
        ),
        QueryEngineTool(
            query_engine=covert_operations_engine,
            metadata=ToolMetadata(
                name="covert_operations",
                description="Provides information about covert operations, S-29 protocol, directives for long term mission and Shadow step as extraction measure when compromised.",
            ),
        ),
        QueryEngineTool(
            query_engine=emergency_directives_engine,
            metadata=ToolMetadata(
                name="emergency_directives",
                description="Provides information about emergency directives, what to do when captured, Protocol Zeta-5 and neural frequency dampers.",
            ),
        ),
        QueryEngineTool(
            query_engine=high_risk_protocol_engine,
            metadata=ToolMetadata(
                name="high_risk_protocol",
                description="Provides information about high-risk operation protocols, operational termination, Project Eclipse, Omega Wave, Blackout Plan Zeta, Silent Dissolution Agents and Cipher Seed Regeneration Program.",
            ),
        ),
        QueryEngineTool(
            query_engine=safe_houses_engine,
            metadata=ToolMetadata(
                name="safe_houses",
                description="Provides information about safe house information and locations like K-41 (Delhi), H-77 (Berlin), X-17, Silent Room",
            ),
        ),
    ]

    # Initialize SubQuestion Query Engine
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        llm=gemini_llm,
        use_async=False,
        verbose=True
    )
    return query_engine

def greeting_style(agent_level):
    greeting_list = ["Salute, Shadow Cadet.", "Bonjour, Sentinel", "Eyes open, Phantom.", "In the wind, Commander.", "The unseen hand moves, Whisper."]
    response_style = ["Basic and instructional, like a mentor guiding a trainee.", "Tactical and direct, focusing on execution and efficiency.", "Analytical and multi-layered, providing strategic insights.", "Coded language, hints, and only essential confirmations.", "Vague, layered, sometimes answering with counter-questions."]
    agent_no = int(agent_level.split("-")[-1]) - 1
    ret_str = """ \n Start the Greeting with: """ + greeting_list[agent_no] + """ \n Have the following response style: """ + response_style[agent_no]
    return ret_str
def query_parser(user_query, response_rec, agent_level):
    base_string = "My Question: " + user_query + """ \n If: you dont get any appropriate info about any part of question return 'Oops!! No Matching Data Found' for that particular part mentioning the part too."""
    for tup_entry in response_rec:
        str_add = """ \n But: """ + tup_entry[1] + """ IF, question is about: """ + tup_entry[0]
        base_string = base_string + str_add
    greet_style = greeting_style(agent_level)
    base_string = base_string + greet_style
    return base_string

# Streamlit UI
st.title("Agent Query System")

# Dropdown for agent level
agent_level = st.selectbox(
    "Select Agent Level",
    ["Level-1", "Level-2", "Level-3", "Level-4", "Level-5"]
)

# Text input for user query
user_query = st.text_input("Enter your query")

# Submit button
if st.button("Submit"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            # Setup query engine
            query_engine = setup_query_engine()
            
            # Get rule type and response_rec
            rule_type, response_rec = get_rule_o_response(user_query, agent_level)
            
            # Process based on rule_type
            if rule_type == "response":
                output = str(response_rec)
            else:
                query_to_send = query_parser(user_query, response_rec, agent_level)
                response = query_engine.query(query_to_send)
                output = str(response)
            
            # Display output in a text box
            st.text_area("Output", value=output, height=300)