from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATASET_DIR = Path("all-recipes/data")
OUTPUT_DIR = Path("artifacts/index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "auto"
BATCH_SIZE = 512
ENCODE_BATCH_SIZE = 128
ROW_LIMIT: int | None = 10000


def resolve_device(preferred: str) -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return preferred


def discover_parquet_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    return files


def iter_recipe_texts(parquet_path: Path) -> Iterable[str]:
    parquet_file = pq.ParquetFile(parquet_path)
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=["input"])
        yield from table.column("input").to_pylist()


def ensure_output_paths() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "index": OUTPUT_DIR / "recipes.faiss",
        "sqlite": OUTPUT_DIR / "metadata.sqlite",
    }


def init_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE chunk_metadata (
            global_row_id INTEGER PRIMARY KEY,
            parquet_filename TEXT NOT NULL,
            row_in_file INTEGER NOT NULL,
            recipe_text TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX idx_chunk_metadata_file_row
        ON chunk_metadata (parquet_filename, row_in_file);
        """
    )
    return conn


def process_batch(
    model: SentenceTransformer,
    index: faiss.IndexIDMap2,
    conn: sqlite3.Connection,
    batch_records: list[tuple[int, str, int, str]],
) -> None:
    ids = np.array([record[0] for record in batch_records], dtype=np.int64)
    texts = [record[3] for record in batch_records]

    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index.add_with_ids(embeddings, ids)

    conn.executemany(
        """
        INSERT INTO chunk_metadata (global_row_id, parquet_filename, row_in_file, recipe_text)
        VALUES (?, ?, ?, ?)
        """,
        batch_records,
    )


def build_index() -> None:
    files = discover_parquet_files(DATASET_DIR)
    output_paths = ensure_output_paths()

    runtime_device = resolve_device(DEVICE)
    model = SentenceTransformer(MODEL_NAME, device=runtime_device)
    probe_embedding = model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=True)
    vector_dim = int(probe_embedding.shape[1])

    base_index = faiss.IndexFlatIP(vector_dim)
    index = faiss.IndexIDMap2(base_index)

    conn = init_sqlite(output_paths["sqlite"])

    total_indexed = 0
    batch_records: list[tuple[int, str, int, str]] = []
    next_global_row_id = 0

    target = ROW_LIMIT if ROW_LIMIT else 2147248
    pbar = tqdm(total=target, desc="Indexing")

    try:
        for parquet_path in files:
            row_in_file = 0
            for recipe_text in iter_recipe_texts(parquet_path):
                batch_records.append(
                    (
                        next_global_row_id,
                        parquet_path.name,
                        row_in_file,
                        recipe_text,
                    )
                )
                next_global_row_id += 1
                row_in_file += 1
                total_indexed += 1

                if len(batch_records) >= BATCH_SIZE:
                    process_batch(model, index, conn, batch_records)
                    conn.commit()
                    pbar.update(len(batch_records))
                    batch_records.clear()

                if ROW_LIMIT is not None and total_indexed >= ROW_LIMIT:
                    break

            if ROW_LIMIT is not None and total_indexed >= ROW_LIMIT:
                break

        if batch_records:
            process_batch(model, index, conn, batch_records)
            conn.commit()
            pbar.update(len(batch_records))
            batch_records.clear()

    finally:
        pbar.close()
        conn.close()
    faiss.write_index(index, str(output_paths["index"]))

    print(f"Indexed rows: {total_indexed}")
    print(f"Device: {runtime_device}")
    print(f"Vector dimension: {vector_dim}")
    print(f"FAISS index: {output_paths['index']}")
    print(f"Metadata DB: {output_paths['sqlite']}")


if __name__ == "__main__":
    build_index()
