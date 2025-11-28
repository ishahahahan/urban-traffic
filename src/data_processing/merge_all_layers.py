"""
Merge all layers (road, metro, walk) into final unified multi-layer network.
"""

import pandas as pd
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def merge_all_layers(multilayer_dir, output_dir):
    """
    Merge all layer files into final unified network.
    
    Parameters:
    -----------
    multilayer_dir : str
        Directory containing intermediate layer files
    output_dir : str
        Directory for final output files
        
    Returns:
    --------
    tuple : (nodes_final, edges_final, transfers_final)
    """
    
    print("Merging all layers into final network...")
    
    # Load all node files
    print("\nLoading node files...")
    road_nodes = pd.read_csv(f'{multilayer_dir}/nodes_multilayer.csv')
    metro_nodes = pd.read_csv(f'{multilayer_dir}/metro_nodes.csv')
    walk_nodes = pd.read_csv(f'{multilayer_dir}/walk_nodes.csv')
    
    print(f"  Road nodes: {len(road_nodes)}")
    print(f"  Metro nodes: {len(metro_nodes)}")
    print(f"  Walk nodes: {len(walk_nodes)}")
    
    # Merge nodes
    nodes_final = pd.concat([road_nodes, metro_nodes, walk_nodes], ignore_index=True)
    
    # Load all edge files
    print("\nLoading edge files...")
    road_edges = pd.read_csv(f'{multilayer_dir}/edges_multilayer.csv')
    metro_edges = pd.read_csv(f'{multilayer_dir}/metro_edges.csv')
    walk_edges = pd.read_csv(f'{multilayer_dir}/walk_edges.csv')
    
    print(f"  Road edges: {len(road_edges)}")
    print(f"  Metro edges: {len(metro_edges)}")
    print(f"  Walk edges: {len(walk_edges)}")
    
    # Merge edges
    edges_final = pd.concat([road_edges, metro_edges, walk_edges], ignore_index=True)
    
    # Load transfers
    print("\nLoading transfers...")
    transfers_final = pd.read_csv(f'{multilayer_dir}/transfers.csv')
    print(f"  Total transfers: {len(transfers_final)}")
    
    # Save final files
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_final.to_csv(f'{output_dir}/nodes_final.csv', index=False)
    edges_final.to_csv(f'{output_dir}/edges_final.csv', index=False)
    transfers_final.to_csv(f'{output_dir}/transfers_final.csv', index=False)
    
    print(f"\nFinal network saved to {output_dir}")
    print(f"  Total nodes: {len(nodes_final)}")
    print(f"  Total edges: {len(edges_final)}")
    print(f"  Total transfers: {len(transfers_final)}")
    
    # Generate summary statistics
    summary = {
        'total_nodes': int(len(nodes_final)),
        'total_edges': int(len(edges_final)),
        'total_transfers': int(len(transfers_final)),
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
            transfer_type: int(count)
            for transfer_type, count in transfers_final['transfer_type'].value_counts().items()
        },
        'modes': {
            mode: int(count)
            for mode, count in edges_final['mode'].value_counts().items()
        }
    }
    
    # Save summary
    with open(f'{output_dir}/network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nNetwork summary:")
    print(json.dumps(summary, indent=2))
    
    return nodes_final, edges_final, transfers_final


if __name__ == "__main__":
    # Example usage
    nodes, edges, transfers = merge_all_layers(
        multilayer_dir='data/multilayer',
        output_dir='data/final'
    )
    
    print("\nMerge complete!")
