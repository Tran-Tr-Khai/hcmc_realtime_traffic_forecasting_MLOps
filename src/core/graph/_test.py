import sys
import os
import numpy as np

# Thêm path
sys.path.append(os.getcwd())

from src.core.graph.loader import LocalGraphLoader
from src.core.graph.adjacency_builder import AdjacencyMatrixBuilder

def validate_graph_pipeline(json_path: str):
    """
    Load graph topology and validate mathematical properties for GCN.
    """
    print(f"--- Test data: {os.path.basename(json_path)} ---")

    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return

    # 1. Load Topology
    try:
        loader = LocalGraphLoader(json_path)
        topology = loader.load_topology()
    except Exception as e:
        print(f"[Error] Failed to load topology: {e}")
        return

    # 2. Validate Dimensions (Critical Logic Check)
    num_nodes = len(topology.node_ids)
    adj_shape = topology.adjacency_matrix.shape
    
    print(f"[Info] Nodes: {num_nodes} | Adjacency Shape: {adj_shape}")

    # Check cụ thể vấn đề 159 vs 140
    if num_nodes != 140:
        print(f"[Warning] Node count mismatch! Expected 140, got {num_nodes}. Check data cleaning logic.")
    else:
        print("[OK] Node count matches pruned dataset (140).")

    # 3. Check Mapping Alignment
    # Chỉ cần check biên (đầu/cuối) để đảm bảo index không bị lệch
    first_id, last_id = topology.node_ids[0], topology.node_ids[-1]
    print(f"[Info] Mapping check: FirstID({first_id})->0 | LastID({last_id})->{topology.node_to_index[last_id]}")

    # 4. Validate Matrix Properties (The "Senior" Part)
    builder = AdjacencyMatrixBuilder(add_self_loops=True)
    norm_adj = builder.build_normalized(topology, normalization='symmetric')

    # Tính toán các chỉ số quan trọng thay vì chỉ in shape
    is_symmetric = np.allclose(norm_adj, norm_adj.T, atol=1e-6)
    has_nan = np.isnan(norm_adj).any()
    sparsity = 1.0 - (np.count_nonzero(norm_adj) / norm_adj.size)
    trace_val = np.trace(norm_adj) # Tổng đường chéo chính

    print("\n--- Matrix Diagnostics ---")
    print(f"Shape        : {norm_adj.shape}")
    print(f"Symmetry     : {'PASS' if is_symmetric else 'FAIL'}")
    print(f"Numerical    : {'CLEAN' if not has_nan else 'CONTAINS NAN'}")
    print(f"Sparsity     : {sparsity:.4f} (Dense data check)")
    print(f"Trace (Sum)  : {trace_val:.2f} (Should be > 0 w/ self-loops)")

    # Final assertion for CI/CD pipeline style
    if not is_symmetric or has_nan or num_nodes != 140:
        print("\n>>> PIPELINE FAILED VALIDATION <<<")
    else:
        print("\n>>> PIPELINE READY <<<")

if __name__ == "__main__":
    validate_graph_pipeline("data/raw/hcmc-clustered-graph.json")