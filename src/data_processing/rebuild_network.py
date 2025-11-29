"""
Rebuild Multi-Layer Network
============================
Master script to rebuild the entire multi-layer transportation network
with all fixes applied.

Run this script to:
1. Fix metro station connectivity
2. Create proper inter-station edges
3. Create comprehensive transfer connections
4. Merge all layers into final files
"""

import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from data_processing.fix_metro_network import fix_metro_network
from data_processing.fix_transfers import create_all_transfers, verify_transfers
import pandas as pd
import json


def rebuild_network(project_root=None):
    """
    Rebuild the entire multi-layer network with fixes.
    
    Parameters:
    -----------
    project_root : str
        Root directory of the project
    """
    
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(script_dir))
    
    multilayer_dir = os.path.join(project_root, 'data/multilayer')
    final_dir = os.path.join(project_root, 'data/final')
    
    print("="*70)
    print("REBUILDING MULTI-LAYER TRANSPORTATION NETWORK")
    print("="*70)
    print(f"Project root: {project_root}")
    print(f"Multilayer dir: {multilayer_dir}")
    print(f"Final dir: {final_dir}")
    
    # =========================================================================
    # Step 1: Fix metro network
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: FIXING METRO NETWORK")
    print("="*70)
    
    metro_nodes, metro_edges = fix_metro_network(
        input_dir=multilayer_dir,
        output_dir=multilayer_dir
    )
    
    if metro_nodes is None:
        print("ERROR: Failed to fix metro network")
        return False
    
    # =========================================================================
    # Step 2: Merge all layers
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: MERGING ALL LAYERS")
    print("="*70)
    
    # Load road nodes and edges
    road_nodes = pd.read_csv(f'{multilayer_dir}/nodes_multilayer.csv')
    road_edges = pd.read_csv(f'{multilayer_dir}/edges_multilayer.csv')
    
    print(f"Road nodes: {len(road_nodes)}")
    print(f"Road edges: {len(road_edges)}")
    
    # Load walk nodes and edges
    walk_nodes_file = f'{multilayer_dir}/walk_nodes.csv'
    walk_edges_file = f'{multilayer_dir}/walk_edges.csv'
    
    if os.path.exists(walk_nodes_file) and os.path.exists(walk_edges_file):
        walk_nodes = pd.read_csv(walk_nodes_file)
        walk_edges = pd.read_csv(walk_edges_file)
        print(f"Walk nodes: {len(walk_nodes)}")
        print(f"Walk edges: {len(walk_edges)}")
    else:
        print("Walk layer files not found - creating from road layer")
        walk_nodes = road_nodes.copy()
        walk_nodes['layer'] = 2
        walk_nodes['node_id'] = walk_nodes['node_id'].str.replace('road_', 'walk_')
        walk_nodes['node_type'] = 'walk'
        
        walk_edges = road_edges.copy()
        walk_edges['layer'] = 2
        walk_edges['edge_id'] = walk_edges['edge_id'].str.replace('road_', 'walk_')
        walk_edges['u'] = walk_edges['u'].str.replace('road_', 'walk_')
        walk_edges['v'] = walk_edges['v'].str.replace('road_', 'walk_')
        walk_edges['mode'] = 'walk'
        walk_edges['speed_mps'] = 1.4  # Walking speed
        walk_edges['travel_time_s'] = walk_edges['length_m'] / 1.4
        
        # Save walk layer
        walk_nodes.to_csv(walk_nodes_file, index=False)
        walk_edges.to_csv(walk_edges_file, index=False)
        print(f"Created walk layer: {len(walk_nodes)} nodes, {len(walk_edges)} edges")
    
    # Merge all nodes
    print("\nMerging nodes...")
    nodes_final = pd.concat([road_nodes, metro_nodes, walk_nodes], ignore_index=True)
    
    # Merge all edges
    print("Merging edges...")
    edges_final = pd.concat([road_edges, metro_edges, walk_edges], ignore_index=True)
    
    # Ensure output directory exists
    os.makedirs(final_dir, exist_ok=True)
    
    # Save merged files
    nodes_final.to_csv(f'{final_dir}/nodes_final.csv', index=False)
    edges_final.to_csv(f'{final_dir}/edges_final.csv', index=False)
    
    print(f"\nMerged network:")
    print(f"  Total nodes: {len(nodes_final)}")
    print(f"  Total edges: {len(edges_final)}")
    
    # =========================================================================
    # Step 3: Create transfers
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATING TRANSFER CONNECTIONS")
    print("="*70)
    
    transfers = create_all_transfers(
        nodes_file=f'{final_dir}/nodes_final.csv',
        output_dir=multilayer_dir
    )
    
    # Copy transfers to final directory
    transfers.to_csv(f'{final_dir}/transfers_final.csv', index=False)
    
    # =========================================================================
    # Step 4: Verify
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: VERIFICATION")
    print("="*70)
    
    verify_transfers(
        nodes_file=f'{final_dir}/nodes_final.csv',
        transfers_file=f'{final_dir}/transfers_final.csv'
    )
    
    # Create summary
    summary = {
        'total_nodes': int(len(nodes_final)),
        'total_edges': int(len(edges_final)),
        'total_transfers': int(len(transfers)),
        'layers': {
            'road': {
                'nodes': int(len(road_nodes)),
                'edges': int(len(road_edges))
            },
            'metro': {
                'nodes': int(len(metro_nodes)),
                'edges': int(len(metro_edges))
            },
            'walk': {
                'nodes': int(len(walk_nodes)),
                'edges': int(len(walk_edges))
            }
        },
        'transfer_types': {
            str(k): int(v) for k, v in transfers['transfer_type'].value_counts().items()
        }
    }
    
    with open(f'{final_dir}/network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("REBUILD COMPLETE")
    print("="*70)
    print(f"\nFinal files saved to: {final_dir}")
    print(f"  - nodes_final.csv ({len(nodes_final):,} nodes)")
    print(f"  - edges_final.csv ({len(edges_final):,} edges)")
    print(f"  - transfers_final.csv ({len(transfers):,} transfers)")
    print(f"  - network_summary.json")
    
    return True


if __name__ == "__main__":
    rebuild_network()
