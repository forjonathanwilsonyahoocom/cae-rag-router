import numpy as np

# --- System-Wide Setup Specs ---
MAX_ITEMS_BEFORE_SPLIT = 3
MIN_ITEMS_TO_SPLIT = 1

# Maintain our global static projection matrix to bridge spaces on the fly
np.random.seed(42)
JL_PROJECTION_MATRIX = np.random.randn(768, 192).astype(np.float32)
JL_PROJECTION_MATRIX /= np.linalg.norm(JL_PROJECTION_MATRIX, axis=0)

GLOBAL_ID_COUNTER = 0

def new_node(depth: int, spec: str) -> dict:
    global GLOBAL_ID_COUNTER
    GLOBAL_ID_COUNTER += 1
    return {
        "node_id": GLOBAL_ID_COUNTER,
        "depth": depth,
        "spec": spec,
        "split_spec": spec,
        "items": [],
        "children": [],
        "child_node_ids": [],
        "child_centroids": None, # Matrix of child paths
        "parent": -1,
    }

def get_embedding(item: dict, spec: str) -> np.ndarray:
    v = np.array(item["EMBEDDINGS"][spec], dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def anchor_split(X: np.ndarray) -> np.ndarray:
    centered = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    return u[:, 0] > 0


class RealtimePropagationRouter:
    def __init__(self):
        self.node_registry = {}
        self.root = new_node(depth=0, spec="SMOKE_PANEL_192")
        self.node_registry[self.root["node_id"]] = self.root

    def insert_item(self, item: dict):
        current_node = self.root
        
        while len(current_node["children"]) > 0:
            spec = current_node["split_spec"]
            query_vector = get_embedding(item, spec)
            scores = np.dot(current_node["child_centroids"], query_vector)
            current_node = self.node_registry[current_node["child_node_ids"][np.argmax(scores)]]

        current_node["items"].append(item)
        print(f"📥 Inserted {item['id']} -> Node {current_node['node_id']} (Depth {current_node['depth']})")

        # Evaluate instant split
        if len(current_node["items"]) >= MAX_ITEMS_BEFORE_SPLIT:
            # For this test, alternate spaces by depth to verify our cross-dimensional propagation
            next_spec = "FULL_SEMANTIC_768" if current_node["depth"] >= 0 else "SMOKE_PANEL_192"
            self.split_node(current_node, next_spec)

    def split_node(self, node: dict, target_space: str):
        items = node["items"]
        node["items"] = []
        node["split_spec"] = target_space

        child_a = new_node(depth=node["depth"] + 1, spec=target_space)
        child_b = new_node(depth=node["depth"] + 1, spec=target_space)
        child_a["parent"] = node["node_id"]
        child_b["parent"] = node["node_id"]

        # Run our split bisector
        X = np.vstack([get_embedding(i, target_space) for i in items])
        mask = anchor_split(X)
        
        for item, label in zip(items, mask):
            selected_child = child_a if label else child_b
            selected_child["items"].append(item)

        # Bake immediate local centroids
        matrix_a = np.vstack([get_embedding(i, target_space) for i in child_a["items"]])
        matrix_b = np.vstack([get_embedding(i, target_space) for i in child_b["items"]])
        centroid_a = np.mean(matrix_a, axis=0)
        centroid_b = np.mean(matrix_b, axis=0)
        
        node["child_centroids"] = np.vstack([
            centroid_a / np.linalg.norm(centroid_a),
            centroid_b / np.linalg.norm(centroid_b)
        ])
        node["child_node_ids"] = [child_a["node_id"], child_b["node_id"]]
        node["children"] = [child_a["node_id"], child_b["node_id"]]

        self.node_registry[child_a["node_id"]] = child_a
        self.node_registry[child_b["node_id"]] = child_b
        
        print(f"⚡ SPLIT: Node {node['node_id']} spawned Leaves {child_a['node_id']} & {child_b['node_id']} using Space: [{target_space}]")

        # --- CHOOSE REAL-TIME PROPAGATION UPWARD ---
        self.propagate_centroid_signal(node["node_id"])

    def propagate_centroid_signal(self, node_id: int):
        """Recursively cascades centroid adjustments up to the root node, 
        polymorphically compressing dimensions via JL Projection if a space mismatch occurs.
        """
        node = self.node_registry[node_id]
        if node["parent"] == -1:
            print("🔝 Reached Root Node. Ancestral propagation loop concluded successfully.")
            return

        parent_id = node["parent"]
        parent_node = self.node_registry[parent_id]
        parent_spec = parent_node["split_spec"]
        
        print(f"📡 [Propagation Signal] Node {node_id} updating Parent {parent_id}'s tracking row...")

        # 1. Compute the current true mathematical centroid of this node
        # Since this node is a branch, its true identity is the combined mean of its children
        current_node_centroid = np.mean(node["child_centroids"], axis=0)
        current_node_centroid /= np.linalg.norm(current_node_centroid)

        # 2. THE POLYMORPHIC CROSS-DIMENSIONAL BRIDGE
        # If this node has escalated to 768-D but the parent routes via 192-D,
        # we compress the update vector instantly using our static JL projection matrix.
        if node["split_spec"] == "FULL_SEMANTIC_768" and parent_spec == "SMOKE_PANEL_192":
            # Matrix multiplication: (1, 768) dot (768, 192) -> (192,)
            compressed_centroid = np.dot(current_node_centroid, JL_PROJECTION_MATRIX)
            update_vector = compressed_centroid / np.linalg.norm(compressed_centroid)
            print(f"   ⚠️ Dimension Shift Met! Compressed 768-D child vector down to 192-D for Parent context.")
        else:
            # Space configurations match perfectly, pass vector raw
            update_vector = current_node_centroid

        # 3. Find exactly which row index inside the parent's matrix points to us
        local_target_row_idx = parent_node["child_node_ids"].index(node_id)
        
        # Mutate the precise parent matrix vector in-place
        parent_node["child_centroids"][local_target_row_idx] = update_vector

        # 4. Tail-recurse the signal up to the next ancestral layer
        self.propagate_centroid_signal(parent_id)



