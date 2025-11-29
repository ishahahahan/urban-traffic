"""
Generate transfer connections between transportation layers.
This is CRITICAL for multimodal routing to work.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.haversine import haversine_distance


def create_transfer_connections(road_nodes_file, metro_nodes_file, 
                                output_dir, max_transfer_distance=300):
    """
    Create transfer connections between road and metro layers.
    
    Parameters:
    -----------
    road_nodes_file : str
        Path to road layer nodes
    metro_nodes_file : str
        Path to metro layer nodes
    output_dir : str
        Directory for output
    max_transfer_distance : float
        Maximum walking distance to metro station (meters)
        Typical: 300-500m
        
    Returns:
    --------
    DataFrame : transfers
    """
    
    print("Creating transfer connections...")
    print(f"Maximum transfer distance: {max_transfer_distance}m")
    
    # Load nodes
    road_nodes = pd.read_csv(road_nodes_file)
    metro_nodes = pd.read_csv(metro_nodes_file)
    
    print(f"Loaded {len(road_nodes)} road nodes, {len(metro_nodes)} metro stations")
    
    # Build spatial index for road nodes
    road_coords = np.array(list(zip(road_nodes['x'], road_nodes['y'])))
    road_tree = cKDTree(road_coords)
    
    transfers = []
    transfer_id = 0
    
    # For each metro station, find nearby road nodes
    for _, metro in metro_nodes.iterrows():
        metro_coord = [metro['x'], metro['y']]
        
        # Find road nodes within max distance
        # Convert meters to approximate degrees (rough for Delhi latitude)
        max_dist_degrees = max_transfer_distance / 111000
        
        indices = road_tree.query_ball_point(metro_coord, r=max_dist_degrees)
        
        for idx in indices:
            road_node = road_nodes.iloc[idx]
            
            # Calculate exact distance
            distance_m = haversine_distance(
                road_node['y'], road_node['x'],
                metro['y'], metro['x']
            )
            
            if distance_m <= max_transfer_distance:
                # Calculate transfer time
                walk_speed = 1.4  # m/s
                walk_time = distance_m / walk_speed
                
                # Add overhead times
                entry_overhead = 120  # 2 min (stairs, security, ticketing)
                exit_overhead = 60    # 1 min (just stairs/elevator)
                
                # Road → Metro (Entry)
                transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': road_node['node_id'],
                    'to_node': metro['node_id'],
                    'from_layer': 0,
                    'to_layer': 1,
                    'transfer_time_s': walk_time + entry_overhead,
                    'transfer_type': 'metro_entry',
                    'distance_m': distance_m,
                    'properties': json.dumps({
                        'metro_name': metro['name'],
                        'has_elevator': bool(np.random.choice([True, False], p=[0.3, 0.7])),
                        'has_escalator': bool(np.random.choice([True, False], p=[0.5, 0.5])),
                        'accessibility_rating': int(np.random.choice([1, 2, 3, 4, 5]))
                    })
                })
                transfer_id += 1
                
                # Metro → Road (Exit)
                transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': metro['node_id'],
                    'to_node': road_node['node_id'],
                    'from_layer': 1,
                    'to_layer': 0,
                    'transfer_time_s': walk_time + exit_overhead,
                    'transfer_type': 'metro_exit',
                    'distance_m': distance_m,
                    'properties': json.dumps({
                        'metro_name': metro['name'],
                        'has_elevator': bool(np.random.choice([True, False], p=[0.3, 0.7])),
                        'has_escalator': bool(np.random.choice([True, False], p=[0.5, 0.5])),
                        'accessibility_rating': int(np.random.choice([1, 2, 3, 4, 5]))
                    })
                })
                transfer_id += 1
    
    # Create DataFrame
    transfers_df = pd.DataFrame(transfers)
    
    if len(transfers_df) == 0:
        print("WARNING: No transfers created! Check max_transfer_distance.")
        return transfers_df
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    transfers_df.to_csv(f'{output_dir}/transfers.csv', index=False)
    
    print(f"\nCreated {len(transfers_df)} transfer connections:")
    print(f"  Entry transfers: {len(transfers_df[transfers_df['transfer_type']=='metro_entry'])}")
    print(f"  Exit transfers: {len(transfers_df[transfers_df['transfer_type']=='metro_exit'])}")
    print(f"  Average transfer time: {transfers_df['transfer_time_s'].mean():.1f}s")
    print(f"  Average walk distance: {transfers_df['distance_m'].mean():.1f}m")
    
    # Statistics by metro station
    entry_transfers = transfers_df[transfers_df['transfer_type']=='metro_entry']
    if len(entry_transfers) > 0:
        transfer_counts = entry_transfers.groupby('to_node').size()
        print(f"\nTransfers per metro station:")
        print(f"  Min: {transfer_counts.min()}")
        print(f"  Max: {transfer_counts.max()}")
        print(f"  Mean: {transfer_counts.mean():.1f}")
    
    return transfers_df


def add_walking_layer_transfers(walk_nodes_file, road_nodes_file, output_dir):
    """
    Add instant transfers between road and walking layers.
    These represent the ability to park/walk at any road node.
    
    Parameters:
    -----------
    walk_nodes_file : str
        Path to walking layer nodes
    road_nodes_file : str
        Path to road layer nodes
    output_dir : str
        Directory for output
        
    Returns:
    --------
    DataFrame : walk_transfers
    """
    
    print("\nCreating road ↔ walk transfers...")
    
    walk_nodes = pd.read_csv(walk_nodes_file)
    road_nodes = pd.read_csv(road_nodes_file)
    
    walk_transfers = []
    transfer_id = 10000  # Start from high number to avoid conflicts
    
    # For each node, create bidirectional transfer
    for i in range(len(road_nodes)):
        road_node = road_nodes.iloc[i]
        walk_node = walk_nodes.iloc[i]
        
        # Road → Walk (start walking)
        walk_transfers.append({
            'transfer_id': f'trans_{transfer_id}',
            'from_node': road_node['node_id'],
            'to_node': walk_node['node_id'],
            'from_layer': 0,
            'to_layer': 2,
            'transfer_time_s': 0,  # Instant
            'transfer_type': 'mode_switch',
            'distance_m': 0,
            'properties': '{"action": "park_and_walk"}'
        })
        transfer_id += 1
        
        # Walk → Road (get vehicle)
        walk_transfers.append({
            'transfer_id': f'trans_{transfer_id}',
            'from_node': walk_node['node_id'],
            'to_node': road_node['node_id'],
            'from_layer': 2,
            'to_layer': 0,
            'transfer_time_s': 30,  # Small delay to get vehicle
            'transfer_type': 'mode_switch',
            'distance_m': 0,
            'properties': '{"action": "return_to_vehicle"}'
        })
        transfer_id += 1
    
    walk_transfers_df = pd.DataFrame(walk_transfers)
    
    # Append to existing transfers
    if os.path.exists(f'{output_dir}/transfers.csv'):
        existing = pd.read_csv(f'{output_dir}/transfers.csv')
        combined = pd.concat([existing, walk_transfers_df], ignore_index=True)
        combined.to_csv(f'{output_dir}/transfers.csv', index=False)
        print(f"Added {len(walk_transfers_df)} walk transfers to existing file")
    else:
        walk_transfers_df.to_csv(f'{output_dir}/transfers.csv', index=False)
        print(f"Created {len(walk_transfers_df)} walk transfers")
    
    return walk_transfers_df


if __name__ == "__main__":
    # Create metro transfers
    transfers = create_transfer_connections(
        road_nodes_file='data/multilayer/nodes_multilayer.csv',
        metro_nodes_file='data/multilayer/metro_nodes.csv',
        output_dir='data/multilayer',
        max_transfer_distance=300
    )
    
    # Add walking transfers
    walk_transfers = add_walking_layer_transfers(
        walk_nodes_file='data/multilayer/walk_nodes.csv',
        road_nodes_file='data/multilayer/nodes_multilayer.csv',
        output_dir='data/multilayer'
    )
    
    print("\nTransfer creation complete!")
