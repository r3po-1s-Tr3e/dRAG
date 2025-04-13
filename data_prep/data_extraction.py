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
model = SentenceTransformer('all-MiniLM-L6-v2')
tqdm.pandas(desc="Embedding Queries")

# Function to embed queries (skipping "All")
def embed_query(row):
    if row['Agent'].strip().lower() == "all":
        return row['Query']  # Keep original string
    else:
        return model.encode(row['Query'].strip()).tolist()

# Apply with progress bar
rule_df['Query'] = rule_df.progress_apply(embed_query, axis=1)

# Save to JSON
# rule_df.to_json("rule_embeddings.json", orient="records", indent=2)
rule_df.to_csv("rule_embeddings.csv", index=False, encoding="utf-8-sig")