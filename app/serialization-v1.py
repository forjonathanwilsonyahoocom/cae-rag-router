import os
import json
import numpy as np

class CAESerializationManager:
    """Handles ultra-fast hybrid serialization of the routing tree.
    
    Bakes centroids to a flat memory-mappable .npy file and saves 
    structural node metadata to a lean manifest JSON.
    """
    
    @staticmethod
    def serialize(router, base_path: str = "./cae_index"):
        """Saves the live node registry and centroids to disk."""
        os.makedirs(base_path, exist_ok=True)
        manifest_path = os.path.join(base_path, "manifest.json")
        centroids_path = os.path.join(base_path, "centroids.npy")
        
        serializable_registry = {}
        all_centroid_blocks = []
        current_row_offset = 0
        
        for node_id, node in router.node_registry.items():
            # Deep-copy properties to prevent mutability issues during serialization
            node_copy = {k: v for k, v in node.items() if k != "child_centroids"}
            
            # If the node has cached routing vectors, isolate them for the binary file
            if node["child_centroids"] is not None:
                matrix = node["child_centroids"] # Shape: (N, Dim)
                num_rows = matrix.shape[0]
                dimension = matrix.shape[1]
                
                all_centroid_blocks.append(matrix)
                
                # Save the mapping slice metrics inside the JSON block
                node_copy["centroid_slice"] = {
                    "row_start": current_row_offset,
                    "row_end": current_row_offset + num_rows,
                    "shape": [num_rows, dimension]
                }
                current_row_offset += num_rows
            else:
                node_copy["centroid_slice"] = None
                
            serializable_registry[str(node_id)] = node_copy
            
        # 1. Write the packed binary array asset
        if all_centroid_blocks:
            packed_centroids = np.vstack(all_centroid_blocks)
            np.save(centroids_path, packed_centroids)
        else:
            # Handle empty tree edge-case safely
            np.save(centroids_path, np.array([], dtype=np.float32))
            
        # 2. Write the structural JSON map
        with open(manifest_path, "w") as f:
            json.dump(serializable_registry, f, indent=2)
            
        print(f"💾 [Serialization Success]")
        print(f"   ├─ Structural Manifest: {manifest_path} ({len(serializable_registry)} nodes)")
        print(f"   └─ Packed Vectors Matrix: {centroids_path} (Total Rows: {current_row_offset})")

    @staticmethod
    def deserialize(base_path: str = "./cae_index") -> dict:
        """Loads and pieces the router state back together using zero-copy memory maps."""
        manifest_path = os.path.join(base_path, "manifest.json")
        centroids_path = os.path.join(base_path, "centroids.npy")
        
        if not os.path.exists(manifest_path) or not os.path.exists(centroids_path):
            raise FileNotFoundError(f"Missing core index files in path target: {base_path}")
            
        # 1. Memory-map the large binary vector space directly 
        # This keeps RAM overhead low by pulling pages from disk dynamically
        mmap_centroids = np.load(centroids_path, mmap_mode="r")
        
        # 2. Extract structural keys
        with open(manifest_path, "r") as f:
            raw_registry = json.load(f)
            
        reconstructed_registry = {}
        for string_id, node in raw_registry.items():
            node_id = int(string_id)
            slice_info = node["centroid_slice"]
            
            # Slice the specific matrix blocks back into the node layout
            if slice_info is not None:
                start = slice_info["row_start"]
                end = slice_info["row_end"]
                # Zero-copy slicing into the memory map
                node["child_centroids"] = mmap_centroids[start:end].copy()
            else:
                node["child_centroids"] = None
                
            del node["centroid_slice"]
            reconstructed_registry[node_id] = node
            
        print(f"📂 [Deserialization Success] Restored {len(reconstructed_registry)} nodes into memory via mmap.")
        return reconstructed_registry
