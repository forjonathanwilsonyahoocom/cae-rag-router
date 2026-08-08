import os
import json
import numpy as np

# Adjust constraints for easy tracing
MAX_ITEMS_BEFORE_SPLIT = 3
MIN_ITEMS_TO_SPLIT = 1
MIN_SPLIT_BALANCE_RATIO = 0.0
MIN_SPLIT_GAIN_FLOOR = -1.0
MIN_WEIGHTED_GAIN = -1.0
EMBEDDING_SPACE_CHANGE_HYSTERESIS_BOOST = 1.2

GLOBAL_ID_COUNTER = 0

def new_node(depth: int, spec: str) -> dict:
    global GLOBAL_ID_COUNTER
    GLOBAL_ID_COUNTER += 1
    return {
        "node_id": GLOBAL_ID_COUNTER,
        "depth": depth,
        "spec": spec,
        "split_spec": spec,
        "items": [],        # Only populated if it is a LEAF node
        "children": [],     # Populated if it is a BRANCH node
        "child_node_ids": [],
        "child_centroids": None, # Cache matrix: shape (N, Dim)
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

def local_cohesion(X: np.ndarray) -> float:
    centroid = np.mean(X, axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0: return 0.0
    return float(np.mean(np.dot(X, centroid / norm)))


class PreciseCAERouter:
    def __init__(self):
        self.node_registry = {}
        self.root = new_node(depth=0, spec="SMOKE_PANEL_192")
        self.node_registry[self.root["node_id"]] = self.root

    def insert_item(self, item: dict):
        current_node = self.root

        # High-speed routing loop via pre-cached child centroids
        while len(current_node["children"]) > 0:
            spec = current_node["split_spec"]
            query_vector = get_embedding(item, spec)
            
            # Instant matrix multiplication over cached child centroids
            scores = np.dot(current_node["child_centroids"], query_vector)
            winner_local_idx = np.argmax(scores)
            
            next_node_id = current_node["child_node_ids"][winner_local_idx]
            current_node = self.node_registry[next_node_id]

        current_node["items"].append(item)
        print(f"📥 Inserted -> Node {current_node['node_id']} (Depth {current_node['depth']}, Space: {current_node['spec']}) | Size: {len(current_node['items'])}")

        should_split, selected_spec = self.faster_split_test(current_node)
        if should_split:
            self.split_node(current_node, selected_spec)

    def faster_split_test(self, node: dict):
        if len(node["items"]) < 2:
            return False, None
        items = node["items"]
        
        # Space change boundary logic
        available_spaces = [node["spec"]]
        if node["depth"] >= 1:
            available_spaces.append("FULL_SEMANTIC_768")

        decision_specs = {}
        for embedding_type in available_spaces:
            X = np.vstack([get_embedding(i, embedding_type) for i in items])

            mask = anchor_split(X)
            b_size = sum(~mask)
            a_size = sum(mask)
            if a_size == 0 or b_size == 0: continue

            balance_ratio = min(a_size, b_size) / max(a_size, b_size)
            cohesion_node = local_cohesion(X)
            gain_a = local_cohesion(X[mask]) - cohesion_node
            gain_b = local_cohesion(X[~mask]) - cohesion_node
            symmetry = min(gain_a, gain_b) / max(gain_a, gain_b) if (gain_a > 0 and gain_b > 0) else 1.0
            weighted_gain = (a_size * gain_a + b_size * gain_b) / len(items)

            decision_specs[embedding_type] = {
                "key": embedding_type, "mask": mask, "cohesion": cohesion_node,
                "balance_ratio": balance_ratio, "symmetry": symmetry, "weighted_gain": weighted_gain
            }

        if not decision_specs: return False, None

        max_score = float("-inf")
        selected_spec = None
        for space_type, spec_data in decision_specs.items():
            score = spec_data["weighted_gain"] * np.sqrt(spec_data["balance_ratio"]) * np.sqrt(spec_data["symmetry"])
            if space_type == node["spec"]:
                score *= EMBEDDING_SPACE_CHANGE_HYSTERESIS_BOOST
            
            # Clean Depth Pressure adjustment: favor rich representations deeper down
            if node["depth"] >= 1 and space_type == "FULL_SEMANTIC_768":
                score += 2.0

            if score > max_score:
                selected_spec = spec_data
                max_score = score

        return True, selected_spec

    def split_node(self, node: dict, embedding_spec: dict):
        items = node["items"]
        node["items"] = [] # Clear items as it steps up to a Branch Node
        node["split_spec"] = embedding_spec["key"]

        # Spawn children utilizing the target space specification
        child_a = new_node(depth=node["depth"] + 1, spec=embedding_spec["key"])
        child_b = new_node(depth=node["depth"] + 1, spec=embedding_spec["key"])
        child_a["parent"] = node["node_id"]
        child_b["parent"] = node["node_id"]

        mask = embedding_spec["mask"]
        for item, label in zip(items, mask):
            selected_child = child_a if label else child_b
            selected_child["items"].append(item)

        # BAKE CHILDS' CENTROIDS PERMANENTLY INTO PARENT NODE
        matrix_a = np.vstack([get_embedding(i, embedding_spec["key"]) for i in child_a["items"]])
        matrix_b = np.vstack([get_embedding(i, embedding_spec["key"]) for i in child_b["items"]])
        
        centroid_a = np.mean(matrix_a, axis=0)
        centroid_b = np.mean(matrix_b, axis=0)
        
        # Commit cached routing state
        node["child_centroids"] = np.vstack([
            centroid_a / np.linalg.norm(centroid_a),
            centroid_b / np.linalg.norm(centroid_b)
        ])
        node["child_node_ids"] = [child_a["node_id"], child_b["node_id"]]
        node["children"] = [child_a["node_id"], child_b["node_id"]]

        # Commit to global registry
        self.node_registry[child_a["node_id"]] = child_a
        self.node_registry[child_b["node_id"]] = child_b

        print(f"⚡ SPLIT Node {node['node_id']} -> Spawned Leaves {child_a['node_id']} & {child_b['node_id']} via Spec: [{embedding_spec['key']}]")


    # --- THE REHYDRATION ENGINE ---
    def query_and_rehydrate(self, query_item: dict) -> dict:
        """Traverses the tree to find the terminal leaf node for a query, 
        and extracts the exact physical context payload.
        """
        current_node = self.root
        path_taken = [current_node["node_id"]]

        while len(current_node["children"]) > 0:
            spec = current_node["split_spec"]
            query_vector = get_embedding(query_item, spec)
            
            scores = np.dot(current_node["child_centroids"], query_vector)
            winner_local_idx = np.argmax(scores)
            
            next_node_id = current_node["child_node_ids"][winner_local_idx]
            current_node = self.node_registry[next_node_id]
            path_taken.append(current_node["node_id"])

        print(f"\n🔍 [Rehydration Search] Routed path: {' -> '.join(map(str, path_taken))}")
        print(f"📍 Context Found inside Leaf Node {current_node['node_id']} (Contains {len(current_node['items'])} sibling documents)")
        
        # Return the clean rehydrated context payload packet
        return {
            "target_node_id": current_node["node_id"],
            "depth": current_node["depth"],
            "space_resolved": current_node["spec"],
            "fragments": [item["text_payload"] for item in current_node["items"]]
        }

# --- Simulation Run ---
router = PreciseCAERouter()

np.random.seed(42)
print("--- Starting Insertion Sequence ---")
for idx in range(6):
    item = {
        "id": f"doc_{idx}",
        "text_payload": f"This is raw text payload content for document chunk sequence number {idx}.",
        "EMBEDDINGS": {
            "SMOKE_PANEL_192": np.random.randn(192).tolist(),
            "FULL_SEMANTIC_768": np.random.randn(768).tolist()
        }
    }
    router.insert_item(item)

# Test the Rehydration Loop
test_query = {
    "EMBEDDINGS": {
        "SMOKE_PANEL_192": np.random.randn(192).tolist(),
        "FULL_SEMANTIC_768": np.random.randn(768).tolist()
    }
}

rehydrated_context = router.query_and_rehydrate(test_query)
print("\n📦 [Rehydrated Output for LLM Context Window]:")
print(json.dumps(rehydrated_context, indent=2))

