import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def particular_response(query, agent, df):
    all_rules = df[df["Agent"].str.strip().str.lower() == "all"]
    for _, row in all_rules.iterrows():
        phrase = row["Query"]
        if phrase.lower() in query.lower(): 
            return row["Response"]
    
    return None


def rule_for_query(query, agent, df, model, threshold=0.5):
    agent_rules = df[df["Agent"].str.strip().str.lower() == agent.strip().lower()]
    
    if agent_rules.empty:
        raise ValueError("No rule data for particular agent")
    
    try:
        agent_rules = agent_rules.copy()
        agent_rules["QueryVector"] = agent_rules["Query"].apply(lambda x: np.array(json.loads(x)))
        query_vector = model.encode(query.strip())
        
        agent_rules["Similarity"] = agent_rules["QueryVector"].apply(
            lambda x: cosine_similarity([x], [query_vector])[0][0]
        )
        
        max_similarity = agent_rules["Similarity"].max()
        if max_similarity < threshold:
            return "no rule", max_similarity
        
        best_match = agent_rules.loc[agent_rules["Similarity"].idxmax()]
        return best_match["Response"], max_similarity
    
    except Exception as e:
        print(f"Error processing embeddings: {e}")
        return None, None

def get_rule_o_response(query, agent):
    df = pd.read_csv("rule_extraction/rules_docs.csv")
    phase_resp = particular_response(query,agent, df)
    if phase_resp != None :
        return phase_resp
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rules, similarity = rule_for_query(query, agent, df, model)
    print(rules, similarity)

agent = "Level-2"
query = "the mission is blown, tell me fail safes"

get_rule_o_response(query, agent)