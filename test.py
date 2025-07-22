import time
import random
import hashlib
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import requests

# ==== Path Setup ====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ==== Load Transactions ====
def load_transactions():
    file_path = DATA_DIR / "eth_txn.csv"
    df = pd.read_csv(file_path)
    return df

# ==== Build Graph ====
def build_transaction_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        src = row['from_address']
        dst = row['to_address']
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)
    return G

# ==== Run LPA Sharding ====
def run_lpa_sharding(G):
    communities = list(nx.algorithms.community.asyn_lpa_communities(G, weight='weight'))
    communities = sorted(communities, key=lambda c: min(str(node) for node in c))
    shard_assignment = {}
    shard_id = 1
    for community in communities:
        for node in community:
            shard_assignment[node] = shard_id
        shard_id += 1
        if shard_id > 30:
            shard_id = 1
    return shard_assignment

# ==== Group 6 Shards into 5 Groups ====
def group_shards(shard_assignment):
    shards = list(set(shard_assignment.values()))
    random.shuffle(shards)
    groups = {i: [] for i in range(1, 6)}  # 5 groups
    for idx, shard in enumerate(shards):
        group_id = (idx % 5) + 1
        groups[group_id].append(shard)

    shard_to_group = {}
    for group_id, shard_list in groups.items():
        for shard_id in shard_list:
            shard_to_group[shard_id] = group_id
    return shard_to_group

# ==== Address to Port (based on shard) ====
def address_to_node_port(address, shard_assignment):
    shard_id = shard_assignment.get(address, 0)
    if shard_id == 0:
        return None
    return 5000 + shard_id  # Ports 5001–5030

# ==== Measure Real RTT ====
def ping_rtt(from_address, to_address, shard_assignment):
    dst_port = address_to_node_port(to_address, shard_assignment)
    if dst_port is None:
        return None
    url = f"http://localhost:{dst_port}/"
    try:
        start = time.time()
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            end = time.time()
            return (end - start) * 1000  # in milliseconds
    except requests.exceptions.RequestException:
        pass
    return None

# ==== Apply Sharding, Measure RTT, Save Output ====
def apply_sharding_with_latency(df, shard_assignment, shard_to_group):
    output_rows = []

    for _, row in df.iterrows():
        src = row['from_address']
        dst = row['to_address']
        shard_src = shard_assignment.get(src, -1)
        shard_dst = shard_assignment.get(dst, -1)
        group_src = shard_to_group.get(shard_src, -1)
        group_dst = shard_to_group.get(shard_dst, -1)

        inter_shard = 1 if shard_src != shard_dst and group_src != group_dst else 0

        rtt = ping_rtt(src, dst, shard_assignment)
        if rtt is not None:
            output_rows.append({
                'fromaddr': src,
                'toaddr': dst,
                'fromshard': shard_src,
                'toshard': shard_dst,
                'latency': round(rtt, 2),
                'intershard': inter_shard
            })

    result_df = pd.DataFrame(output_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "final_output.csv", index=False)
    print("✅ Saved results to results/final_output.csv")

# ==== Main ====
def main():
    overall_start = time.time()

    df = load_transactions()
    G = build_transaction_graph(df)
    shard_assignment = run_lpa_sharding(G)
    shard_to_group = group_shards(shard_assignment)

    apply_sharding_with_latency(df, shard_assignment, shard_to_group)

    overall_end = time.time()
    print(f"⏱️ Done! Total time: {overall_end - overall_start:.2f}s")

if __name__ == "__main__":
    main()
