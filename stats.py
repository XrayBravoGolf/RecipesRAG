import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np
import tiktoken
import re
import random

def get_text_columns(schema):
    text_cols = []
    excluded = {'id', 'url', 'source'}
    for name in schema.names:
        if name.lower() in excluded:
            continue
        field = schema.field(name)
        t = field.type
        # Check if it's string or list of strings
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            text_cols.append(name)
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            value_type = t.value_type
            if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
                text_cols.append(name)
    return text_cols

def main():
    dataset = ds.dataset("all-recipes/data/", format="parquet")
    schema = dataset.schema
    text_cols = get_text_columns(schema)
    print(f"Selected columns: {text_cols}")

    total_rows = dataset.count_rows()
    print(f"Total rows in dataset: {total_rows}")

    sample_size = min(50000, total_rows)
    random.seed(42)
    
    # Simple reservoir-like sampling using scanner if possible, 
    # but for 50k from a potentially large dataset, we can pick indices or just shuffle small chunks.
    # To be efficient and match reservoir sampling requirements:
    indices = sorted(random.sample(range(total_rows), sample_size))
    
    # scanner = dataset.scanner(columns=text_cols)
    # However, pyarrow doesn't support direct index-based scanning easily across multiple files without loading.
    # We will use a generator to stream and pick.
    
    word_regex = re.compile(r'\b\w+\b')
    enc = tiktoken.get_encoding("cl100k_base")
    
    word_counts = []
    token_counts = []
    
    current_idx = 0
    indices_set = set(indices)
    
    for batch in dataset.to_batches(columns=text_cols):
        batch_dict = batch.to_pydict()
        batch_len = batch.num_rows
        
        # Check which indices in this batch are selected
        batch_indices = [i for i in range(batch_len) if (current_idx + i) in indices_set]
        
        if batch_indices:
            for i in batch_indices:
                row_parts = []
                for col in text_cols:
                    val = batch_dict[col][i]
                    if val is None:
                        continue
                    if isinstance(val, (list, np.ndarray)):
                        row_parts.append(" ".join(str(v) for v in val if v is not None))
                    else:
                        row_parts.append(str(val))
                
                row_text = " ".join(row_parts)
                word_counts.append(len(word_regex.findall(row_text)))
                token_counts.append(len(enc.encode(row_text)))
        
        current_idx += batch_len
        if len(word_counts) >= sample_size:
            break

    print(f"Sampled row count: {len(word_counts)}")
    
    def stats(data):
        return np.mean(data), np.median(data), np.percentile(data, 90)

    m_w, med_w, p90_w = stats(word_counts)
    m_t, med_t, p90_t = stats(token_counts)
    
    print(f"Words per row - Mean: {m_w:.2f}, Median: {med_w:.2f}, P90: {p90_w:.2f}")
    print(f"Tokens per row - Mean: {m_t:.2f}, Median: {med_t:.2f}, P90: {p90_t:.2f}")
    print(f"Words/Token ratio: {(m_w / m_t):.4f}")

if __name__ == "__main__":
    main()
