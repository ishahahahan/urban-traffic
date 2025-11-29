"""
Fix Transfer Connections Between Network Layers
================================================
This script creates proper transfer connections between:
1. Road (Layer 0) <-> Metro (Layer 1)
2. Road (Layer 0) <-> Walk (Layer 2) 
3. Walk (Layer 2) <-> Metro (Layer 1)

Key improvements:
- Proper bidirectional transfers
- Reasonable transfer times
- Walk layer properly connected to metro stations
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def create_all_transfers(nodes_file, output_dir, 
                         max_metro_distance=400,
                         min_connections_per_metro=5):
    """
    Create comprehensive transfer connections between all layers.
    
    Parameters:
    -----------
    nodes_file : str
        Path to nodes_final.csv (contains all layers)
    output_dir : str
        Directory for output
    max_metro_distance : float
        Maximum walking distance to metro (meters)
    min_connections_per_metro : int
        Minimum road connections per metro station
        
    Returns:
    --------
    DataFrame : all transfers
    """
    
    print("="*70)
    print("CREATING COMPREHENSIVE TRANSFER CONNECTIONS")
    print("="*70)
    
    # Load nodes
    nodes_df = pd.read_csv(nodes_file, dtype={'name': str})
    
    # Split by layer
    road_nodes = nodes_df[nodes_df['layer'] == 0].copy()
    metro_nodes = nodes_df[nodes_df['layer'] == 1].copy()
    walk_nodes = nodes_df[nodes_df['layer'] == 2].copy()
    
    print(f"\nNodes by layer:")
    print(f"  Road (Layer 0): {len(road_nodes)}")
    print(f"  Metro (Layer 1): {len(metro_nodes)}")
    print(f"  Walk (Layer 2): {len(walk_nodes)}")
    
    all_transfers = []
    transfer_id = 0
    
    # =========================================================================
    # 1. ROAD <-> METRO TRANSFERS
    # =========================================================================
    print("\n--- Creating Road <-> Metro Transfers ---")
    
    if len(road_nodes) > 0 and len(metro_nodes) > 0:
        # Build spatial index for road nodes
        road_coords = np.array(list(zip(road_nodes['x'], road_nodes['y'])))
        road_tree = cKDTree(road_coords)
        
        # Walk speed and overhead times
        walk_speed = 1.4  # m/s
        metro_entry_overhead = 120  # 2 min (security, ticketing, stairs)
        metro_exit_overhead = 60   # 1 min (just stairs/escalator)
        
        metro_connections = {}
        
        for _, metro in metro_nodes.iterrows():
            metro_coord = [metro['x'], metro['y']]
            metro_name = metro['name'] if pd.notna(metro['name']) else 'Unknown Station'
            
            # Find road nodes within max distance (convert meters to degrees)
            max_dist_degrees = max_metro_distance / 111000
            indices = road_tree.query_ball_point(metro_coord, r=max_dist_degrees)
            
            connections = []
            
            for idx in indices:
                road_node = road_nodes.iloc[idx]
                
                # Calculate exact distance
                distance_m = haversine_distance(
                    road_node['y'], road_node['x'],
                    metro['y'], metro['x']
                )
                
                if distance_m <= max_metro_distance:
                    connections.append((idx, distance_m))
            
            # Ensure minimum connections by extending search if needed
            if len(connections) < min_connections_per_metro:
                distances, indices = road_tree.query(
                    metro_coord, k=min_connections_per_metro
                )
                for i, dist_deg in enumerate(distances):
                    idx = indices[i]
                    road_node = road_nodes.iloc[idx]
                    distance_m = haversine_distance(
                        road_node['y'], road_node['x'],
                        metro['y'], metro['x']
                    )
                    if (idx, distance_m) not in connections:
                        connections.append((idx, distance_m))
            
            # Create transfers for each connection
            for idx, distance_m in connections:
                road_node = road_nodes.iloc[idx]
                walk_time = distance_m / walk_speed
                
                # Road -> Metro (Entry)
                all_transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': road_node['node_id'],
                    'to_node': metro['node_id'],
                    'from_layer': 0,
                    'to_layer': 1,
                    'transfer_time_s': walk_time + metro_entry_overhead,
                    'transfer_type': 'metro_entry',
                    'distance_m': distance_m,
                    'properties': json.dumps({
                        'metro_name': metro_name,
                        'walk_time_s': walk_time,
                        'overhead_s': metro_entry_overhead
                    })
                })
                transfer_id += 1
                
                # Metro -> Road (Exit)
                all_transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': metro['node_id'],
                    'to_node': road_node['node_id'],
                    'from_layer': 1,
                    'to_layer': 0,
                    'transfer_time_s': walk_time + metro_exit_overhead,
                    'transfer_type': 'metro_exit',
                    'distance_m': distance_m,
                    'properties': json.dumps({
                        'metro_name': metro_name,
                        'walk_time_s': walk_time,
                        'overhead_s': metro_exit_overhead
                    })
                })
                transfer_id += 1
            
            metro_connections[metro['node_id']] = len(connections)
        
        avg_connections = np.mean(list(metro_connections.values()))
        print(f"  Created {transfer_id} road<->metro transfers")
        print(f"  Average connections per metro station: {avg_connections:.1f}")
    
    # =========================================================================
    # 2. ROAD <-> WALK TRANSFERS (Mode Switch)
    # =========================================================================
    print("\n--- Creating Road <-> Walk Transfers ---")
    
    road_walk_transfers = 0
    
    if len(road_nodes) > 0 and len(walk_nodes) > 0:
        # These should be instant transfers at same location
        # Represents parking/picking up vehicle
        
        # Build mapping between road and walk nodes
        # Assumption: walk nodes are created from road nodes (same locations)
        walk_coords = np.array(list(zip(walk_nodes['x'], walk_nodes['y'])))
        walk_tree = cKDTree(walk_coords)
        
        for _, road_node in road_nodes.iterrows():
            road_coord = [road_node['x'], road_node['y']]
            
            # Find nearest walk node
            dist, idx = walk_tree.query(road_coord, k=1)
            walk_node = walk_nodes.iloc[idx]
            
            # Check if close enough (should be same location)
            distance_m = haversine_distance(
                road_node['y'], road_node['x'],
                walk_node['y'], walk_node['x']
            )
            
            if distance_m < 50:  # Same location (within 50m)
                # Road -> Walk (park and walk)
                all_transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': road_node['node_id'],
                    'to_node': walk_node['node_id'],
                    'from_layer': 0,
                    'to_layer': 2,
                    'transfer_time_s': 30,  # 30 seconds to park
                    'transfer_type': 'park_and_walk',
                    'distance_m': distance_m,
                    'properties': json.dumps({'action': 'park_vehicle'})
                })
                transfer_id += 1
                
                # Walk -> Road (get vehicle)
                all_transfers.append({
                    'transfer_id': f'trans_{transfer_id}',
                    'from_node': walk_node['node_id'],
                    'to_node': road_node['node_id'],
                    'from_layer': 2,
                    'to_layer': 0,
                    'transfer_time_s': 30,  # 30 seconds to get vehicle
                    'transfer_type': 'get_vehicle',
                    'distance_m': distance_m,
                    'properties': json.dumps({'action': 'get_vehicle'})
                })
                transfer_id += 1
                road_walk_transfers += 2
        
        print(f"  Created {road_walk_transfers} road<->walk transfers")
    
    # =========================================================================
    # 3. WALK <-> METRO TRANSFERS
    # =========================================================================
    print("\n--- Creating Walk <-> Metro Transfers ---")
    
    walk_metro_transfers = 0
    
    if len(walk_nodes) > 0 and len(metro_nodes) > 0:
        # Build spatial index for walk nodes
        walk_coords = np.array(list(zip(walk_nodes['x'], walk_nodes['y'])))
        walk_tree = cKDTree(walk_coords)
        
        walk_speed = 1.4  # m/s
        metro_entry_overhead = 90   # 1.5 min (already walking, just enter)
        metro_exit_overhead = 45    # 45 sec
        
        for _, metro in metro_nodes.iterrows():
            metro_coord = [metro['x'], metro['y']]
            metro_name = metro['name'] if pd.notna(metro['name']) else 'Unknown Station'
            
            # Find walk nodes within distance
            max_dist_degrees = max_metro_distance / 111000
            indices = walk_tree.query_ball_point(metro_coord, r=max_dist_degrees)
            
            for idx in indices:
                walk_node = walk_nodes.iloc[idx]
                
                distance_m = haversine_distance(
                    walk_node['y'], walk_node['x'],
                    metro['y'], metro['x']
                )
                
                if distance_m <= max_metro_distance:
                    walk_time = distance_m / walk_speed
                    
                    # Walk -> Metro
                    all_transfers.append({
                        'transfer_id': f'trans_{transfer_id}',
                        'from_node': walk_node['node_id'],
                        'to_node': metro['node_id'],
                        'from_layer': 2,
                        'to_layer': 1,
                        'transfer_time_s': walk_time + metro_entry_overhead,
                        'transfer_type': 'walk_to_metro',
                        'distance_m': distance_m,
                        'properties': json.dumps({
                            'metro_name': metro_name,
                            'walk_time_s': walk_time
                        })
                    })
                    transfer_id += 1
                    
                    # Metro -> Walk
                    all_transfers.append({
                        'transfer_id': f'trans_{transfer_id}',
                        'from_node': metro['node_id'],
                        'to_node': walk_node['node_id'],
                        'from_layer': 1,
                        'to_layer': 2,
                        'transfer_time_s': walk_time + metro_exit_overhead,
                        'transfer_type': 'metro_to_walk',
                        'distance_m': distance_m,
                        'properties': json.dumps({
                            'metro_name': metro_name,
                            'walk_time_s': walk_time
                        })
                    })
                    transfer_id += 1
                    walk_metro_transfers += 2
        
        print(f"  Created {walk_metro_transfers} walk<->metro transfers")
    
    # Create DataFrame
    transfers_df = pd.DataFrame(all_transfers)
    
    # Summary
    print("\n--- Transfer Summary ---")
    print(f"Total transfers: {len(transfers_df)}")
    
    if len(transfers_df) > 0:
        transfer_counts = transfers_df['transfer_type'].value_counts()
        for ttype, count in transfer_counts.items():
            print(f"  {ttype}: {count}")
        
        print(f"\nTransfer time statistics:")
        print(f"  Mean: {transfers_df['transfer_time_s'].mean():.1f}s")
        print(f"  Min: {transfers_df['transfer_time_s'].min():.1f}s")
        print(f"  Max: {transfers_df['transfer_time_s'].max():.1f}s")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    transfers_df.to_csv(f'{output_dir}/transfers.csv', index=False)
    
    print(f"\n✓ Saved to {output_dir}/transfers.csv")
    
    return transfers_df


def verify_transfers(nodes_file, transfers_file):
    """
    Verify that transfers properly connect all layers.
    """
    
    print("\n" + "="*70)
    print("VERIFYING TRANSFER CONNECTIVITY")
    print("="*70)
    
    import networkx as nx
    
    nodes_df = pd.read_csv(nodes_file, dtype={'name': str})
    transfers_df = pd.read_csv(transfers_file)
    
    # Build graph with only transfers
    G = nx.DiGraph()
    
    # Add all nodes
    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'], layer=node['layer'])
    
    # Add transfer edges
    for _, trans in transfers_df.iterrows():
        G.add_edge(
            trans['from_node'],
            trans['to_node'],
            transfer_type=trans['transfer_type'],
            time=trans['transfer_time_s']
        )
    
    # Check connectivity between layers
    layers = {0: 'Road', 1: 'Metro', 2: 'Walk'}
    
    print("\nConnectivity between layers:")
    
    for from_layer in [0, 1, 2]:
        for to_layer in [0, 1, 2]:
            if from_layer == to_layer:
                continue
            
            # Count transfers between these layers
            count = len(transfers_df[
                (transfers_df['from_layer'] == from_layer) & 
                (transfers_df['to_layer'] == to_layer)
            ])
            
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {layers[from_layer]} -> {layers[to_layer]}: {count} transfers")
    
    # Check if metro stations are reachable from road
    road_nodes = nodes_df[nodes_df['layer'] == 0]['node_id'].tolist()
    metro_nodes = nodes_df[nodes_df['layer'] == 1]['node_id'].tolist()
    
    if road_nodes and metro_nodes:
        # Sample check
        sample_road = road_nodes[0]
        reachable_metros = 0
        
        for metro in metro_nodes[:10]:  # Check first 10 metros
            if nx.has_path(G, sample_road, metro):
                reachable_metros += 1
        
        print(f"\nFrom sample road node, reachable metros: {reachable_metros}/10")


if __name__ == "__main__":
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Create transfers
    transfers_df = create_all_transfers(
        nodes_file=os.path.join(project_root, 'data/final/nodes_final.csv'),
        output_dir=os.path.join(project_root, 'data/multilayer')
    )
    
    # Verify
    verify_transfers(
        nodes_file=os.path.join(project_root, 'data/final/nodes_final.csv'),
        transfers_file=os.path.join(project_root, 'data/multilayer/transfers.csv')
    )
