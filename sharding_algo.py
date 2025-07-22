import time
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import requests

# ==== Path Setup ====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def load_transactions():
    file_path = DATA_DIR / "eth_txn.csv"
    # Expecting CSV with at least columns: from_address, to_address
    df = pd.read_csv(file_path)
    return df

def build_transaction_graph(df):
    """Builds weighted graph from transactions. Edge weights represent frequency."""
    G = nx.Graph()
    for index, row in df.iterrows():
        src = row['from_address']
        dst = row['to_address']
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)
    return G

def run_lpa_sharding(G):
    """
    Run asynchronous label propagation algorithm.
    Group the discovered communities into exactly 10 shards by partitioning the communities list.
    Returns a dictionary mapping address -> shard_id.
    """
    # Run asynchronous label propagation
    communities = list(nx.algorithms.community.asyn_lpa_communities(G, weight='weight'))
    
    # Sort communities (for reproducible grouping) by minimum node value (alphabetically)
    
    
    
    
    communities = sorted(communities, key=lambda c: min(str(node) for node in c))
    num_com = len(communities)
    
    # If exactly 10 communities, use them as shards; otherwise partition communities evenly.
    shard_assignment = {}
    if num_com == 10:
        for shard_id, community in enumerate(communities, start=1):
            for node in community:
                shard_assignment[node] = shard_id
    else:
        # Merge communities into 10 groups:
        # First, assign each community an index and then group communities into 10 buckets.
        buckets = {i: [] for i in range(1, 11)}
        for idx, community in enumerate(communities):
            bucket = (idx % 10) + 1  # distribute communities round-robin into 10 buckets
            buckets[bucket].extend(community)
        # Assign shard id to each node
        for shard_id, nodes in buckets.items():
            for node in nodes:
                shard_assignment[node] = shard_id
    return shard_assignment

def apply_sharding(df, shard_assignment):
    """
    Adds a new column 'shard' to the dataframe.
    We assign the shard corresponding to the from_address.
    """
    df['shard'] = df['from_address'].apply(lambda addr: shard_assignment.get(addr, -1))
    # Mark inter-shard transaction if from_address and to_address are in different shards
    df['inter_shard'] = df.apply(lambda row: 1 if (shard_assignment.get(row['from_address'], -1) != 
                                                     shard_assignment.get(row['to_address'], -1)) else 0, axis=1)
    return df

def measure_inter_shard_percentage(df):
    total = len(df)
    inter_shard_count = df['inter_shard'].sum()
    return (inter_shard_count / total) * 100

def test_docker_latency():
    """
    Ping all 30 docker node endpoints (assumed accessible at localhost ports 5001-5030).
    Return the average round-trip latency (in milliseconds).
    """
    latencies = []
    for port in range(5001, 5031):
        url = f"http://localhost:{port}/"
        try:
            start = time.time()
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                end = time.time()
                latency = (end - start) * 1000  # ms
                latencies.append(latency)
        except requests.exceptions.RequestException:
            # If node is not reachable, record a timeout or skip
            continue
    avg_latency = np.mean(latencies) if latencies else float('nan')
    return avg_latency

def main():
    start_time = time.time()
    # Load transaction data
    df = load_transactions()
    
    # Build the transaction graph from addresses and frequency of transactions.
    G = build_transaction_graph(df)
    
    # Run the sharding algorithm (LPA followed by grouping communities into 10 shards)
    shard_assignment = run_lpa_sharding(G)
    
    # Apply shard assignment to the DataFrame and compute inter-shard flag.
    df = apply_sharding(df, shard_assignment)
    
    # Compute inter-shard communication percentage.
    inter_shard_percentage = measure_inter_shard_percentage(df)
    
    # Save the updated DataFrame with shard and inter_shard columns.
    output_path = RESULTS_DIR / "shards_output.csv"
    df.to_csv(output_path, index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Test latency over Docker nodes.
    avg_latency = test_docker_latency()
    
    print("Sharding algorithm complete.")
    print(f"Total transactions: {len(df)}")
    print(f"Inter-shard communication percentage: {inter_shard_percentage:.2f}%")
    print(f"Time taken for sharding algorithm: {total_time:.2f} seconds")
    if not np.isnan(avg_latency):
        print(f"Average Docker node latency: {avg_latency:.2f} ms")
    else:
        print("Could not measure Docker node latency (nodes unreachable).")

if __name__ == '__main__':
    main()
