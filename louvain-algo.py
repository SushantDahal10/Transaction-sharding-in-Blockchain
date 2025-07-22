import time
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import requests
import community as community_louvain

# === Config Paths ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Step 1: Load CSV ===
def load_transactions(limit=None):
    path = DATA_DIR / "eth_txn.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df

# === Step 2: Build Transaction Graph ===
def build_transaction_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        src, dst = row['from_address'], row['to_address']
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)
    return G

# === Step 3: Louvain Partitioning ===
def run_louvain_sharding(G, max_shards=10):
    partition = community_louvain.best_partition(G, weight='weight')
    unique_coms = list(sorted(set(partition.values())))
    mapping = {com: i+1 for i, com in enumerate(unique_coms[:max_shards])}
    return {node: mapping.get(community, -1) for node, community in partition.items()}

# === Step 4: Ping Each Shard Once ===
def ping_shards(shard_assignment):
    latencies = {}
    for shard_id in sorted(set(shard_assignment.values())):
        if shard_id == -1:
            continue
        try:
            start = time.time()
            res = requests.get(f"http://localhost:{5000 + shard_id}/ping", timeout=2)
            end = time.time()
            if res.status_code == 200:
                latencies[shard_id] = (end - start) * 1000  # ms
            else:
                latencies[shard_id] = np.nan
        except Exception as e:
            latencies[shard_id] = np.nan
    return latencies

# === Step 5: Apply Sharding Info to DataFrame ===
def apply_sharding(df, shard_assignment):
    df['shard'] = df['from_address'].map(shard_assignment).fillna(-1).astype(int)
    df['to_shard'] = df['to_address'].map(shard_assignment).fillna(-1).astype(int)
    df['inter_shard'] = (df['shard'] != df['to_shard']).astype(int)

    latency_map = ping_shards(shard_assignment)
    df['latency_ms'] = df['shard'].map(latency_map)
    df.drop(columns='to_shard', inplace=True)
    return df

# === Step 6: Calculate Inter-shard Traffic ===
def measure_inter_shard_percentage(df):
    total = len(df)
    inter = df['inter_shard'].sum()
    return (inter / total) * 100 if total > 0 else 0

# === Step 7: Main Flow ===
def main():
    start = time.time()
    print("📥 Loading transactions...")
    df = load_transactions()
    print(f"✔️ Loaded {len(df)} rows")

    print("🔧 Building transaction graph...")
    G = build_transaction_graph(df)
    print(f"✔️ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print("🧠 Running Louvain sharding...")
    shard_assignment = run_louvain_sharding(G)

    print("🚚 Applying shard assignments and latency checks...")
    df = apply_sharding(df, shard_assignment)

    inter_shard = measure_inter_shard_percentage(df)
    duration = time.time() - start

    output_path = RESULTS_DIR / "louvain_shards_output.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Done in {duration:.2f}s")
    print(f"🔀 Inter-shard transactions: {inter_shard:.2f}%")
    print(f"📁 Output saved to: {output_path}")

# === Entry Point ===
if __name__ == '__main__':
    main()
