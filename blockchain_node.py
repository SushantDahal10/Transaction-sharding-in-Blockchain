import hashlib
import json
import time
import os
import threading
import socket
from flask import Flask, request, jsonify
import requests

class Blockchain:
    def __init__(self, node_id, nodes):
        self.chain = []
        self.nodes = nodes
        self.node_id = node_id
        self.create_block(data='Genesis Block', previous_hash='0')

    def create_block(self, data, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'data': data,
            'previous_hash': previous_hash,
            'hash': ""  
        }
        block['hash'] = self.hash(block)
        self.chain.append(block)
        return block

    def hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_last_block(self):
        return self.chain[-1] if self.chain else None

    def sync_chain(self):
        """Synchronize blockchain from the longest chain available"""
        longest_chain = None
        max_length = len(self.chain)

        for node in self.nodes:
            try:
                response = requests.get(f"{node}/chain", timeout=2)
                if response.status_code == 200:
                    node_chain = response.json().get("chain", [])
                    if len(node_chain) > max_length:
                        max_length = len(node_chain)
                        longest_chain = node_chain
            except requests.exceptions.RequestException:
                pass  

        if longest_chain:
            self.chain = longest_chain
            return True
        return False

app = Flask(__name__)

# Determine node_id from environment variable if provided, else extract from hostname
node_id = int(os.environ.get("NODE_ID", socket.gethostname().strip("node") or 1))
# Read peers from environment variable (comma separated)
peers_env = os.environ.get("PEERS", "")
nodes = [peer for peer in peers_env.split(",") if peer]

blockchain = Blockchain(node_id, nodes)

def auto_sync():
    while True:
        blockchain.sync_chain()
        time.sleep(5)  # Sync every 5 seconds

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Blockchain Node Running", "node_id": node_id}), 200

@app.route('/broadcast', methods=['POST'])
def broadcast():
    data = request.json.get('data', '')
    last_block = blockchain.get_last_block()
    new_block = blockchain.create_block(data, last_block['hash'] if last_block else '0')

    for node in blockchain.nodes:
        try:
            requests.post(f'{node}/receive', json=new_block, timeout=2)
        except requests.exceptions.RequestException:
            pass  

    for node in blockchain.nodes:
        try:
            requests.get(f'{node}/sync', timeout=2)
        except requests.exceptions.RequestException:
            pass

    return jsonify(new_block), 200

@app.route('/receive', methods=['POST'])
def receive_block():
    block = request.json
    last_block = blockchain.get_last_block()

    if last_block and block['previous_hash'] == last_block['hash']:
        blockchain.chain.append(block)
        return jsonify({"message": "Block received and added"}), 200
    else:
        blockchain.sync_chain()
        return jsonify({"error": "Invalid block, syncing with network"}), 400

@app.route('/sync', methods=['GET'])
def sync():
    success = blockchain.sync_chain()
    if success:
        return jsonify({"message": "Blockchain synchronized"}), 200
    return jsonify({"message": "Blockchain already up-to-date"}), 200

@app.route('/chain', methods=['GET'])
def get_chain():
    index = request.args.get('index', type=int)
    if index is None:
        return jsonify({"chain": blockchain.chain}), 200
    for block in blockchain.chain:
        if block["index"] == index:
            return jsonify(block), 200
    return jsonify({"error": "Invalid block index"}), 400

if __name__ == '__main__':
    threading.Thread(target=auto_sync, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
