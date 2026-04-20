import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("artifacts/index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "auto"


class RecipeRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() and DEVICE == "auto" else ("cpu" if DEVICE == "auto" else DEVICE)
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        
        index_path = INDEX_DIR / "recipes.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Build the index first.")
        
        self.index = faiss.read_index(str(index_path))
        
        sqlite_path = INDEX_DIR / "metadata.sqlite"
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Metadata DB not found at {sqlite_path}. Build the index first.")
            
        # Using check_same_thread=False since Streamlit might spawn background threads
        # and URI ensures read-only mode so we don't accidentally write to it
        self.conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        distances, indices = self.index.search(query_embedding, k)

        valid_indices = [int(idx) for idx in indices[0] if idx != -1]
        if not valid_indices:
            return []

        placeholders = ",".join("?" for _ in valid_indices)
        cursor = self.conn.execute(
            f"""
            SELECT global_row_id, parquet_filename, row_in_file, recipe_text 
            FROM chunk_metadata 
            WHERE global_row_id IN ({placeholders})
            """,
            valid_indices
        )

        # SQLite IN order isn't guaranteed, map by ID to keep FAISS scoring order
        db_rows = {row["global_row_id"]: dict(row) for row in cursor.fetchall()}

        results = []
        for score, global_row_id in zip(distances[0], indices[0]):
            global_row_id = int(global_row_id)
            if global_row_id in db_rows:
                result_dict = db_rows[global_row_id]
                result_dict["score"] = float(score)
                results.append(result_dict)

        return results


if __name__ == "__main__":
    print("Initializing retriever... (this might take a second to load the model)")
    retriever = RecipeRetriever()
    
    # test_query = "spicy chicken noodle soup"
    test_query = input()
    print(f"\nSearching for: '{test_query}'")
    results = retriever.search(test_query, k=3)
    
    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {res['score']:.4f}) ---")
        print(f"Evidence: file {res['parquet_filename']} | row {res['row_in_file']} | global id {res['global_row_id']}")
        print(res["recipe_text"][:800].strip() + "...\n")
