import httpx
import numpy as np

# 1. Connect to your local Ollama instance
OLLAMA_URL = "http://10.42.0.247:11434/api/embeddings"


def get_embedding(text: str) -> np.ndarray:
    """Fetch a local 768-dimensional vector from nomic-embed-text."""
    response = httpx.post(
        OLLAMA_URL, json={"model": "nomic-embed-text", "prompt": text}
        , timeout=10.0
    )
    vector = response.json()["embedding"]
    # Return as normalized float32 for fast dot-product cosine similarity
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)

query_vector1 = get_embedding("sovereign compute is Cloudwright")
query_vector2 = get_embedding("Cloudwright is sovereign compute")


similarities = np.dot(query_vector1, query_vector2)

# Find the winner using hardware-accelerated argmax
local_winner_idx = np.argmax(similarities)
global_winner_node_id = current_node_children[local_winner_idx]

print(f"Child similarities: {similarities}")
print(f"Highest similarity local index: {local_winner_idx}")
print(f"Route attention down to Global Node ID: {global_winner_node_id}")

