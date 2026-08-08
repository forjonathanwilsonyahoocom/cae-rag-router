import os
import json
import numpy as np

class CAESerializationManager:
    """Handles heterogeneous hybrid serialization of polymorphic routing trees.
    
    Groups and packs cached centroids into separate dimension-specific .npy files,
    keeping a lean master JSON manifest tracking slices across files.
    """
    
    @staticmethod
    def serialize(router, base_path: str = "./cae_index"):
        """Saves the live node registry and dimension-segregated centroids to disk."""
        os.makedirs(base_path, exist_ok=True)
        manifest_path = os.path.join(base_path, "manifest.json")
        
        serializable_registry = {}
        # Group centroid matrices by their specific embedding space key
        grouped_centroid_blocks = {}
        # Track independent row offsets for each file structure
        current_offsets = {}
        
        for node_id, node in router.node_registry.items():
            node_copy = {k: v for k, v in node.items() if k != "child_centroids"}
            
            if node["child_centroids"] is not None:
                matrix = node["child_centroids"]
                # The splitting specification determines the dimension profile
                space_key = node["split_spec"] 
                
                num_rows, dimension = matrix.shape
                
                if space_key not in grouped_centroid_blocks:
                    grouped_centroid_blocks[space_key] = []
                    current_offsets[space_key] = 0
                    
                grouped_centroid_blocks[space_key].append(matrix)
                
                # Save coordinates indicating the file target and row span
                node_copy["centroid_slice"] = {
                    "space_key": space_key,
                    "row_start": current_offsets[space_key],
                    "row_end": current_offsets[space_key] + num_rows,
                    "shape": [num_rows, dimension]
                }
                current_offsets[space_key] += num_rows
            else:
                node_copy["centroid_slice"] = None
                
            serializable_registry[str(node_id)] = node_copy
            
        # 1. Save independent dimension-packed binary assets
        print("💾 [Saving Multi-Space Index Assets]")
        for space_key, blocks in grouped_centroid_blocks.items():
            packed_matrix = np.vstack(blocks)
            file_name = f"centroids_{space_key}.npy"
            file_path = os.path.join(base_path, file_name)
            np.save(file_path, packed_matrix)
            print(f"   └─ Packed Vectors Matrix: {file_path} (Rows: {current_offsets[space_key]}, Cols: {packed_matrix.shape[1]})")
            
        # 2. Write the structural JSON master map
        with open(manifest_path, "w") as f:
            json.dump(serializable_registry, f, indent=2)
            
        print(f"   ├─ Structural Manifest Written: {manifest_path} ({len(serializable_registry)} nodes)")

    @staticmethod
    def deserialize(base_path: str = "./cae_index") -> dict:
        """Loads and pieces the multi-space router state back together using isolated memory maps."""
        manifest_path = os.path.join(base_path, "manifest.json")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing master manifest index target: {manifest_path}")
            
        with open(manifest_path, "r") as f:
            raw_registry = json.load(f)
            
        # Dynamically load and cache memory maps on-demand as they are referenced
        mmap_cache = {}
        reconstructed_registry = {}
        
        for string_id, node in raw_registry.items():
            node_id = int(string_id)
            slice_info = node["centroid_slice"]
            
            if slice_info is not None:
                space_key = slice_info["space_key"]
                start = slice_info["row_start"]
                end = slice_info["row_end"]
                
                # Instantly map the binary file if we haven't seen it yet during this load
                if space_key not in mmap_cache:
                    file_path = os.path.join(base_path, f"centroids_{space_key}.npy")
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Expected index file missing: {file_path}")
                    mmap_cache[space_key] = np.load(file_path, mmap_mode="r")
                
                # Fetch a zero-copy pointer slice of the correct dimension profile
                node["child_centroids"] = mmap_cache[space_key][start:end].copy()
            else:
                node["child_centroids"] = None
                
            del node["centroid_slice"]
            reconstructed_registry[node_id] = node
            
        print(f"📂 [Deserialization Success] Hydrated {len(reconstructed_registry)} polymorphically typed nodes via memory-maps.")
        return reconstructed_registry
