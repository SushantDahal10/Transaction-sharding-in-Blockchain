# metis_sharding.py

import time
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import requests
import metis

# === Setup paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def load_transactions():
    return pd.read_csv(DATA_DIR / "eth_txn.csv")

def build_transaction_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        src, dst = row['from_address'], row['to_address']
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)
    return G

def run_metis_sharding(G, num_shards=10):
    _, parts = metis.part_graph(G, nparts=num_shards, recursive=True)
    shard_assignment = {node: parts[i] + 1 for i, node in enumerate(G.nodes())}
    return shard_assignment

def apply_sharding(df, shard_assignment):
    df['shard'] = df['from_address'].apply(lambda x: shard_assignment.get(x, -1))
    df['to_shard'] = df['to_address'].apply(lambda x: shard_assignment.get(x, -1))
    df['inter_shard'] = df.apply(lambda r: 1 if r['shard'] != r['to_shard'] else 0, axis=1)

    latencies = []
    for _, row in df.iterrows():
        shard = row['shard']
        if shard == -1:
            latencies.append(np.nan)
            continue
        try:
            start = time.time()
            res = requests.get(f"http://localhost:{5000 + shard}/ping", timeout=2)
            end = time.time()
            latencies.append((end - start) * 1000 if res.status_code == 200 else np.nan)
        except:
            latencies.append(np.nan)
    df['latency_ms'] = latencies
    df.drop(columns='to_shard', inplace=True)
    return df

def measure_inter_shard_percentage(df):
    return (df['inter_shard'].sum() / len(df)) * 100

def main():
    start = time.time()
    df = load_transactions()
    G = build_transaction_graph(df)
    shard_assignment = run_metis_sharding(G)
    df = apply_sharding(df, shard_assignment)
    inter_shard = measure_inter_shard_percentage(df)
    df.to_csv(RESULTS_DIR / "metis_shards_output.csv", index=False)
    print(f"Done in {time.time() - start:.2f}s — Inter-shard: {inter_shard:.2f}%")

if __name__ == '__main__':
    main()
