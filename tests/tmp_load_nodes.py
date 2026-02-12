import numpy as np

node_ids = np.load('data/processed/node_ids.npy', allow_pickle=True)
print(f'Shape: {node_ids.shape}')
print(f'Dtype: {node_ids.dtype}')
print(f'Length: {len(node_ids)}')
print()
print('Full list of node IDs in canonical order:')
for i, nid in enumerate(node_ids):
    print(f'  [{i}] {nid}')
