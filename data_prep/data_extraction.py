import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# file_loc = r"data_prep\rules_doc.txt"
# with open(file_loc, "r", encoding="utf-8") as f:
#     lines = f.readlines()
# data = {"Agent": [], "Query": [], "Response": []}
# for line in lines:
#     parts = line.strip().split(',', 2)  
#     if len(parts) == 3:
#         data["Agent"].append(parts[0].strip())
#         data["Query"].append(parts[1].strip())
#         data["Response"].append(parts[2].strip())
#     else:
#         print(f"Skipping line (not enough parts): {line.strip()}")
# df = pd.DataFrame(data)
# df.to_csv("rules_doc.csv", index=False, encoding="utf-8-sig")

rule_df = pd.read_csv("data_prep/init_rules_doc.csv", encoding="utf-8-sig")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize tqdm
tqdm.pandas(desc="Embedding Queries")

# Function to embed queries (skipping "All")
def embed_query(row):
    if row['Agent'].strip().lower() == "all":
        return None  # Or you could keep it as the original string if needed
    else:
        return model.encode(row['Query'].strip()).tolist()

# Apply and store embeddings in a new column
rule_df['Query_Vector'] = rule_df.progress_apply(embed_query, axis=1)


import spacy
nlp = spacy.load("en_core_web_sm")
def extract_nouns(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

# Extract nouns from Query column
rule_df['Query_Keyword'] = rule_df['Query'].progress_apply(extract_nouns)

# Save to CSV
rule_df.to_csv("rules_doc.csv", index=False, encoding="utf-8-sig")
