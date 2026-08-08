import numpy as np

# --- System-Wide Phase Transition Configurations ---
MAX_ITEMS_BEFORE_SPLIT = 4
MIN_ITEMS_TO_SPLIT = 1
MIN_SPLIT_BALANCE_RATIO = 0.05
EMBEDDING_SPACE_CHANGE_HYSTERESIS_BOOST = 1.3  # Resists space change unless highly compelled
DEPTH_PRESSURE_LAMBDA = 0.4                  # Controls how fast depth scales the preference for 768-D

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
        "child_centroids": None,
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


class StressTestingCAERouter:
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
        print(f"📥 Inserted {item['id']} into Node {current_node['node_id']} (Depth {current_node['depth']})")

        should_split, selected_spec = self.faster_split_test(current_node)
        if should_split:
            self.split_node(current_node, selected_spec)

    def faster_split_test(self, node: dict):
        if len(node["items"]) < 2:
            return False, None
        items = node["items"]
        
        # Always test both representations to evaluate potential escalation
        available_spaces = ["SMOKE_PANEL_192", "FULL_SEMANTIC_768"]
        decision_specs = {}
        max_dimensions_observed = 0
        for embedding_type in available_spaces:
            X = np.vstack([get_embedding(i, embedding_type) for i in items])
            mask = anchor_split(X)
            b_size, a_size = sum(~mask), sum(mask)
            if a_size == 0 or b_size == 0: continue

            balance_ratio = min(a_size, b_size) / max(a_size, b_size)
            cohesion_node = local_cohesion(X)
            gain_a = local_cohesion(X[mask]) - cohesion_node
            gain_b = local_cohesion(X[~mask]) - cohesion_node
            
            # Avoid division by zero bugs
            max_gain = max(gain_a, gain_b)
            min_gain = min(gain_a, gain_b)
            symmetry = min_gain / max_gain if max_gain > 0 and min_gain > 0 else 0.001

            #locate for later ratio use
            if  X.size > max_dimensions_observed:
                 max_dimensions_observed = X.size
                
            weighted_gain = (a_size * gain_a + b_size * gain_b) / len(items)
            decision_specs[embedding_type] = {
                "key": embedding_type, "dimensions": X.size, "mask": mask, "cohesion": cohesion_node,
                "balance_ratio": balance_ratio, "symmetry": symmetry, "weighted_gain": weighted_gain
            }

        if not decision_specs: return False, None

        # --- THE MATHEMATICAL MATHEMATICAL PHASE TRANSITION GATE ---
        # 1. Calculate Depth Pressure via a saturating exponential curve
        depth_pressure = 1.0 - np.exp(-DEPTH_PRESSURE_LAMBDA * node["depth"])
        
        max_score = float("-inf")
        selected_spec = None

        print(f"🕵️‍♂️ [Evaluating Split for Node {node['node_id']} (Depth {node['depth']})] | Depth Pressure Factor: {depth_pressure:.3f}")
        for space_type, spec_data in decision_specs.items():
            # Base structural fitness of the split
            score = spec_data["weighted_gain"] * np.sqrt(spec_data["balance_ratio"]) * np.sqrt(spec_data["symmetry"])
            
            # Hysteresis: Protect against fluttering space transformations
            if space_type == node["spec"]:
                score *= EMBEDDING_SPACE_CHANGE_HYSTERESIS_BOOST
                
            # Computational Attention Escalation Rule:
            # As Depth Pressure increases, amplify our reward for high-fidelity structures
            if max_dimensions_observed/spec_data["dimensions"] > 0.7:
                # The deeper we go, the more value we place on capturing complex structural symmetry
                score += (depth_pressure * spec_data["symmetry"] * 0.5)

            print(f"   ↳ Space [{space_type}]: raw_weighted_gain={spec_data['weighted_gain']:.4f}, symmetry={spec_data['symmetry']:.4f}, Final Calculated Score={score:.4f}")

            if score > max_score:
                selected_spec = spec_data
                max_score = score

        return True, selected_spec

    def split_node(self, node: dict, embedding_spec: dict):
        items = node["items"]
        node["items"] = []
        node["split_spec"] = embedding_spec["key"]

        child_a = new_node(depth=node["depth"] + 1, spec=embedding_spec["key"])
        child_b = new_node(depth=node["depth"] + 1, spec=embedding_spec["key"])
        child_a["parent"] = node["node_id"]
        child_b["parent"] = node["node_id"]

        mask = embedding_spec["mask"]
        for item, label in zip(items, mask):
            selected_child = child_a if label else child_b
            selected_child["items"].append(item)

        matrix_a = np.vstack([get_embedding(i, embedding_spec["key"]) for i in child_a["items"]])
        matrix_b = np.vstack([get_embedding(i, embedding_spec["key"]) for i in child_b["items"]])
        centroid_a, centroid_b = np.mean(matrix_a, axis=0), np.mean(matrix_b, axis=0)
        
        node["child_centroids"] = np.vstack([centroid_a / np.linalg.norm(centroid_a), centroid_b / np.linalg.norm(centroid_b)])
        node["child_node_ids"] = [child_a["node_id"], child_b["node_id"]]
        node["children"] = [child_a["node_id"], child_b["node_id"]]

        self.node_registry[child_a["node_id"]] = child_a
        self.node_registry[child_b["node_id"]] = child_b
        print(f"⚡ PHASE TRANSITION: Split Node {node['node_id']} into Leaves {child_a['node_id']} & {child_b['node_id']} using Spec: [{embedding_spec['key']}]\n")


# --- RUNNING THE ADVERSARIAL STRESS TEST ---
router = StressTestingCAERouter()

# 1. Base Stream: Uniform items where 192-D captures everything comfortably
np.random.seed(100)
for idx in range(10):
    item = {
        "id": f"standard_doc_{idx}",
        "EMBEDDINGS": {
            "SMOKE_PANEL_192": np.random.randn(192).tolist(),
            "FULL_SEMANTIC_768": np.random.randn(768).tolist()
        }
    }
    router.insert_item(item)

# 2. Inject Adversarial Ambiguity
# These items look completely homogeneous (identical clusters) in 192-D space, 
# but contain highly distinct structural variance inside 768-D space.
print("\n🚨 Injecting Adversarial Items (Hidden complexity only visible in 768-D)...")
adversarial_base_192 = np.random.randn(192) # Shared low-res vector

for idx in range(3):
    # Forcing zero variance in 192-D by making the values identical
    mock_192 = adversarial_base_192.copy().tolist()
    # Forcing massive, hyper-symmetric cluster variance in 768-D
    mock_768 = (np.random.randn(768) * (20.0 if idx % 2 == 0 else -20.0)).tolist()
    
    adv_item = {
        "id": f"adversarial_doc_{idx}",
        "EMBEDDINGS": {
            "SMOKE_PANEL_192": mock_192,
            "FULL_SEMANTIC_768": mock_768
        }
    }
    router.insert_item(adv_item)
