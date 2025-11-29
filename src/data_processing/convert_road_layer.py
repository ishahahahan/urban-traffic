"""
Convert existing OSMnx road network data to multi-layer format.
This forms Layer 0 (road network) of the multi-layer graph.
"""

import pandas as pd
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_road_layer(nodes_file, edges_file, output_dir):
    """
    Convert single-layer OSMnx data to multi-layer format.
    
    Parameters:
    -----------
    nodes_file : str
        Path to nodes.csv (format: osmid, x, y)
    edges_file : str
        Path to edges.csv (format: edge_id, u, v, key, length_m, 
                           free_flow_speed_mps, free_flow_tt_s)
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    tuple : (nodes_df, edges_df)
        DataFrames with multi-layer schema
    """
    
    print(f"Loading data from {nodes_file} and {edges_file}")
    
    # Load existing data
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # Transform nodes to multi-layer format
    nodes_ml = nodes.copy()
    
    # Create new node_id with layer prefix
    nodes_ml['node_id'] = 'road_' + nodes_ml['osmid'].astype(str)
    
    # Add layer information
    nodes_ml['layer'] = 0  # Road layer
    
    # Add node type
    nodes_ml['node_type'] = 'intersection'
    
    # Add empty name field (can be populated later with OSM data)
    nodes_ml['name'] = ''
    
    # Add empty properties field
    nodes_ml['properties'] = '{}'
    
    # Reorder columns to match schema
    nodes_ml = nodes_ml[[
        'node_id', 'layer', 'osmid', 'x', 'y', 
        'node_type', 'name', 'properties'
    ]]
    
    print(f"Transformed {len(nodes_ml)} nodes to multi-layer format")
    
    # Transform edges to multi-layer format
    edges_ml = edges.copy()
    
    # Create new edge_id with layer prefix
    edges_ml['edge_id'] = 'road_' + edges_ml['edge_id'].astype(str)
    
    # Add layer information
    edges_ml['layer'] = 0
    
    # Transform node references
    edges_ml['u'] = 'road_' + edges_ml['u'].astype(str)
    edges_ml['v'] = 'road_' + edges_ml['v'].astype(str)
    
    # Add mode
    edges_ml['mode'] = 'car'
    
    # Rename speed and travel time columns
    edges_ml['speed_mps'] = edges_ml['free_flow_speed_mps']
    edges_ml['travel_time_s'] = edges_ml['free_flow_tt_s']
    
    # Add edge type (default to 'road', can be enhanced with OSM data)
    edges_ml['edge_type'] = 'road'
    
    # Add empty properties
    edges_ml['properties'] = '{}'
    
    # Reorder columns
    edges_ml = edges_ml[[
        'edge_id', 'layer', 'u', 'v', 'key', 'length_m',
        'mode', 'speed_mps', 'travel_time_s', 'edge_type', 'properties'
    ]]
    
    print(f"Transformed {len(edges_ml)} edges to multi-layer format")
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_output = os.path.join(output_dir, 'nodes_multilayer.csv')
    edges_output = os.path.join(output_dir, 'edges_multilayer.csv')
    
    nodes_ml.to_csv(nodes_output, index=False)
    edges_ml.to_csv(edges_output, index=False)
    
    print(f"Saved to:")
    print(f"  {nodes_output}")
    print(f"  {edges_output}")
    
    return nodes_ml, edges_ml


if __name__ == "__main__":
    # Example usage
    nodes_ml, edges_ml = convert_road_layer(
        nodes_file='data/raw/nodes.csv',
        edges_file='data/raw/edges.csv',
        output_dir='data/multilayer'
    )
    
    print("\nConversion complete!")
    print(f"Multi-layer nodes: {len(nodes_ml)}")
    print(f"Multi-layer edges: {len(edges_ml)}")
