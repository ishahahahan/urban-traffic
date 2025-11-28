"""
Create walking layer (Layer 2) parallel to road network.
Walking uses same graph topology but different speeds and constraints.
"""

import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_walking_layer(road_edges_file, road_nodes_file, output_dir, 
                         max_walk_distance=1000):
    """
    Create walking layer from road network.
    
    Parameters:
    -----------
    road_edges_file : str
        Path to edges_multilayer.csv (road layer)
    road_nodes_file : str
        Path to nodes_multilayer.csv (road layer)
    output_dir : str
        Directory for output
    max_walk_distance : float
        Maximum length for walking edges (meters)
        Edges longer than this won't have walking equivalents
        
    Returns:
    --------
    tuple : (walk_nodes_df, walk_edges_df)
    """
    
    print(f"Creating walking layer from {road_edges_file}")
    
    # Load road network
    road_edges = pd.read_csv(road_edges_file)
    road_nodes = pd.read_csv(road_nodes_file)
    
    print(f"Loaded {len(road_edges)} road edges")
    
    # Create walking nodes (copy of road nodes with layer change)
    walk_nodes = road_nodes.copy()
    walk_nodes['node_id'] = walk_nodes['node_id'].str.replace('road_', 'walk_')
    walk_nodes['layer'] = 2
    walk_nodes['node_type'] = 'pedestrian_node'
    
    # Create walking edges (parallel to road edges)
    walk_edges = road_edges.copy()
    
    # Filter out edges too long for walking
    walk_edges = walk_edges[walk_edges['length_m'] <= max_walk_distance].copy()
    
    print(f"Filtered to {len(walk_edges)} edges suitable for walking " 
          f"(<= {max_walk_distance}m)")
    
    # Modify for walking
    walk_edges['edge_id'] = walk_edges['edge_id'].str.replace('road_', 'walk_')
    walk_edges['layer'] = 2
    walk_edges['u'] = walk_edges['u'].str.replace('road_', 'walk_')
    walk_edges['v'] = walk_edges['v'].str.replace('road_', 'walk_')
    walk_edges['mode'] = 'walk'
    
    # Walking speed: 5 km/h = 1.39 m/s
    walk_speed_mps = 1.39
    walk_edges['speed_mps'] = walk_speed_mps
    walk_edges['travel_time_s'] = walk_edges['length_m'] / walk_speed_mps
    walk_edges['edge_type'] = 'pedestrian'
    
    # Update properties
    walk_edges['properties'] = '{"surface": "paved", "lit": true}'
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    walk_nodes.to_csv(f'{output_dir}/walk_nodes.csv', index=False)
    walk_edges.to_csv(f'{output_dir}/walk_edges.csv', index=False)
    
    print(f"Created walking layer:")
    print(f"  {len(walk_nodes)} pedestrian nodes")
    print(f"  {len(walk_edges)} walking edges")
    print(f"  Average walk time: {walk_edges['travel_time_s'].mean():.1f}s")
    
    return walk_nodes, walk_edges


if __name__ == "__main__":
    # Example usage
    walk_nodes, walk_edges = create_walking_layer(
        road_edges_file='data/multilayer/edges_multilayer.csv',
        road_nodes_file='data/multilayer/nodes_multilayer.csv',
        output_dir='data/multilayer',
        max_walk_distance=1000  # Only create walking edges for segments < 1km
    )
    
    print("\nWalking layer creation complete!")
