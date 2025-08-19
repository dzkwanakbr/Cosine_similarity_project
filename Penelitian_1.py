
# ============================================
# IndoBERT + Cosine Similarity Pipeline (for Colab/local)
# Author: Generated for Dzakwan (Polinela)
# ============================================

# --- Install (run in Colab terminal/cell) ---
# !pip install pandas numpy matplotlib networkx scikit-learn torch transformers

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx

# --------- Config ---------
DATA_PATH = "preprocessing_data.csv"   # place your CSV in current working dir
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 64
TOP_K = 10
SIM_THRESHOLD = 0.85

# --------- Load Data ---------
df = pd.read_csv(DATA_PATH)
assert all(col in df.columns for col in ["Preprocessed_Text","year","program_studi"]), "CSV must have columns: Preprocessed_Text, year, program_studi"

# Optional: drop empty titles
df = df.dropna(subset=["Preprocessed_Text"]).reset_index(drop=True)

print("Total entries:", len(df))

# Descriptive stats
print("\nDistribusi tahun:")
print(df["year"].value_counts(dropna=False).sort_index())

print("\nDistribusi program studi:")
print(df["program_studi"].fillna("Unknown").value_counts())

print("\nContoh judul (5):")
print(df["Preprocessed_Text"].sample(5, random_state=42).to_list())

# --------- Load IndoBERT ---------
print("\nLoading tokenizer & model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------- Embedding Function ---------
def embed_texts(texts, batch_size=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            list(batch),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)  # last_hidden_state: [B, T, H]
            cls_vecs = outputs.last_hidden_state[:, 0, :]  # [CLS]
        all_vecs.append(cls_vecs.cpu().numpy())
        print(f"Embedded {i+len(batch)}/{len(texts)}")
    return np.vstack(all_vecs)

# --------- Compute Embeddings ---------
titles = df["Preprocessed_Text"].astype(str).tolist()
embeddings = embed_texts(titles, batch_size=64)
np.save("embeddings_indobert_cls.npy", embeddings)
print("Embeddings shape:", embeddings.shape)

# --------- Cosine Similarity ---------
sim_mat = cosine_similarity(embeddings)

# Get top-K pairs (i<j to avoid duplicates)
pairs = []
n = len(df)
for i in range(n):
    for j in range(i+1, n):
        score = float(sim_mat[i, j])
        # filter: hanya ambil kalau similarity < 0.999 (hindari sama persis)
        if score < 0.999:
            pairs.append((i, j, score))

# Ambil pasangan di atas threshold
THRESHOLD = 0.8
pairs_filtered = [(i, j, s) for (i, j, s) in pairs if s >= THRESHOLD]

# Urutkan dari paling mirip
pairs_sorted = sorted(pairs_filtered, key=lambda x: x[2], reverse=True)

# Ambil Top-K
TOP_K = 10
top_pairs = pairs_sorted[:TOP_K]

print("\nTop-10 pasangan judul skripsi yang mirip (bukan identik):")
for i, j, score in top_pairs:
    print(f"{df['Preprocessed_Text'][i]}  ||  {df['Preprocessed_Text'][j]}  -->  Similarity: {score:.4f}")

# --------- Simpan ke CSV ---------
import pandas as pd

top_rows = []
for i, j, s in top_pairs:
    top_rows.append({
        "similarity": round(s, 4),
        "judul_i": df.loc[i, "Preprocessed_Text"],
        "judul_j": df.loc[j, "Preprocessed_Text"],
        "year_i": df.loc[i, "year"],
        "year_j": df.loc[j, "year"],
        "prodi_i": df.loc[i, "program_studi"],
        "prodi_j": df.loc[j, "program_studi"]
    })

top_df = pd.DataFrame(top_rows)
top_df.to_csv("top10_pairs_filtered.csv", index=False, encoding="utf-8-sig")

print("\nTop-10 pasangan mirip (tanpa duplikat identik) disimpan ke top10_pairs_filtered.csv")
print(top_df)


# --------- Visualization: Heatmap (subset) ---------
# Use matplotlib imshow (no seaborn)
subset = min(30, n)
rng = np.random.default_rng(42)
subset_idx = rng.choice(n, size=subset, replace=False)
sub_mat = sim_mat[np.ix_(subset_idx, subset_idx)]

plt.figure(figsize=(10, 7))
plt.imshow(sub_mat, aspect="auto")
plt.colorbar()
plt.title("Heatmap Kemiripan Judul (Sample {})".format(subset))
plt.xlabel("Index")
plt.ylabel("Index")
plt.tight_layout()
plt.savefig("heatmap_sample.png", dpi=200)
plt.close()

# --------- Visualization: Network Graph ---------
G = nx.Graph()
for i, j, s in pairs:
    if s >= SIM_THRESHOLD:
        G.add_edge(i, j, weight=s)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=30)
nx.draw_networkx_edges(G, pos, alpha=0.3)
# No labels to keep it readable
plt.title(f"Graf Jaringan Kemiripan (Cosine ≥ {SIM_THRESHOLD})")
plt.tight_layout()
plt.savefig("graph_similarity.png", dpi=200)
plt.close()

print("\nSaved: embeddings_indobert_cls.npy, top10_pairs.csv, heatmap_sample.png, graph_similarity.png")
print("Done.")
