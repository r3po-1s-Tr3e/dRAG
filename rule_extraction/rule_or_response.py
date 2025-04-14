import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import ast

def particular_response(query, agent, df):
    all_rules = df[df["Agent"].str.strip().str.lower() == "all"]
    for _, row in all_rules.iterrows():
        phrase = row["Query"]
        if phrase.lower() in query.lower(): 
            return row["Response"]
    
    return None


def rule_for_query_top_k(query, agent, df, model, k=1):
    nlp = spacy.load("en_core_web_sm")

    # Filter the rules for the given agent
    agent_rules = df[df["Agent"].str.strip().str.lower() == agent.strip().lower()]

    if agent_rules.empty:
        raise ValueError("No rule data for the specified agent.")

    agent_rules = agent_rules.copy()

    # Safely parse the vector strings into numpy arrays
    agent_rules["QueryVector"] = agent_rules["Query_Vector"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.zeros((model.get_sentence_embedding_dimension(),))
    )

    query_vector = model.encode(query.strip())

    # Compute semantic similarity
    agent_rules["Similarity"] = agent_rules["QueryVector"].apply(
        lambda x: cosine_similarity([x], [query_vector])[0][0]
    )

    # Extract nouns from the input query
    query_doc = nlp(query.lower())
    query_nouns = {token.text for token in query_doc if token.pos_ == "NOUN"}

    def calculate_keyword_score(keywords):
        try:
            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            if not isinstance(keywords, list):
                return 0
            keywords = [kw.lower() for kw in keywords if isinstance(kw, str)]
            intersection = len(query_nouns.intersection(keywords))
            total_nouns = len(keywords)
            return intersection / total_nouns if total_nouns > 0 else 0
        except Exception as e:
            print(f"Keyword score error: {e}")
            return 0

    agent_rules["KeywordScore"] = agent_rules["Query_Keyword"].apply(calculate_keyword_score)

    top_k_semantic = agent_rules.nlargest(k, "Similarity")[["Query", "Response", "Similarity"]]
    semantic_results = [(row["Query"], row["Response"]) for _, row in top_k_semantic.iterrows()]

    top_2_keyword = agent_rules.nlargest(2, "KeywordScore")[["Query", "Response", "KeywordScore"]]
    keyword_results = [(row["Query"], row["Response"], row["KeywordScore"]) for _, row in top_2_keyword.iterrows()]

    combined_results = semantic_results[:]
    for kw_query, kw_response, kw_score in keyword_results:
        if not any(kw_query == sem_query for sem_query, _ in semantic_results) and kw_score != 0:
            combined_results.append((kw_query, kw_response))

    return combined_results

def get_rule_o_response(query, agent):
    df = pd.read_csv("rule_extraction/rules_doc.csv")
    phase_resp = particular_response(query,agent, df)
    if phase_resp != None :
        return "response", phase_resp
    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = rule_for_query_top_k(query, agent, df, model)
    return "rules", results

# agent = "Level-4"
# query = "the mission is blown, tell me fail-safes"

# get_rule_o_response(query, agent)