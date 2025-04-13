import pandas as pd

file_loc = r"C:\Users\hryad\Desktop\iitm\Sanctity_AI\dRAG\rules_doc.txt"

with open(file_loc, "r", encoding="utf-8") as f:
    lines = f.readlines()

data = {"Agent": [], "Query": [], "Response": []}

for line in lines:
    parts = line.strip().split(',', 2)  
    if len(parts) == 3:
        data["Agent"].append(parts[0].strip())
        data["Query"].append(parts[1].strip())
        data["Response"].append(parts[2].strip())
    else:
        print(f"Skipping line (not enough parts): {line.strip()}")

df = pd.DataFrame(data)
print(df.head())  
print(df["Agent"].value_counts().sort_index())
df.to_csv("rules_doc.csv", index=False, encoding="utf-8-sig")