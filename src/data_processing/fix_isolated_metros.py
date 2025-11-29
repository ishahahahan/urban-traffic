"""
Fix Isolated Metro Stations and Improve Layer Transfers
========================================================
This script:
1. Connects isolated metro stations to their nearest neighbors
2. Merges duplicate interchange stations (e.g., "Azadpur (Pink Line)" with "Azadpur")
3. Creates proper transfers using walk layer as bridge between road and metro
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import sys
from math import radians, sin, cos, sqrt, atan2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters using Haversine formula"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def find_nearest_connected_station(isolated_node, metro_nodes, metro_edges, G, max_distance=2000):
    """
    Find the nearest connected metro station to an isolated station.
    
    Parameters:
    -----------
    isolated_node : str
        Node ID of isolated station
    metro_nodes : DataFrame
        Metro nodes dataframe
    metro_edges : DataFrame
        Metro edges dataframe  
    G : nx.Graph
        Metro graph
    max_distance : float
        Maximum distance to search (meters)
        
    Returns:
    --------
    tuple : (nearest_node_id, distance) or (None, None)
    """
    isolated = metro_nodes[metro_nodes['node_id'] == isolated_node].iloc[0]
    iso_x, iso_y = isolated['x'], isolated['y']
    
    # Get all connected stations
    connected = [n for n in G.nodes() if G.degree(n) > 0]
    
    best_match = None
    best_dist = float('inf')
    
    for node_id in connected:
        node = metro_nodes[metro_nodes['node_id'] == node_id].iloc[0]
        dist = haversine_distance(iso_y, iso_x, node['y'], node['x'])
        
        if dist < best_dist and dist < max_distance:
            best_dist = dist
            best_match = node_id
    
    if best_match:
        return best_match, best_dist
    return None, None


def find_interchange_match(station_name, metro_nodes, connected_nodes):
    """
    Find if an isolated station is a duplicate of an interchange station.
    E.g., "Azadpur (Pink Line)" should connect to "Azadpur"
    
    Returns: node_id of matching station or None
    """
    # Clean the name - remove line info in parentheses
    import re
    base_name = re.sub(r'\s*\([^)]*Line\)', '', station_name).strip()
    base_name_lower = base_name.lower()
    
    for node_id in connected_nodes:
        node = metro_nodes[metro_nodes['node_id'] == node_id].iloc[0]
        node_name = str(node['name']).lower().strip()
        
        # Check for match
        if base_name_lower == node_name:
            return node_id
        if base_name_lower in node_name or node_name in base_name_lower:
            return node_id
    
    return None


def fix_isolated_stations(input_dir='data/multilayer', output_dir='data/multilayer'):
    """
    Fix isolated metro stations by connecting them to nearest neighbors.
    """
    print("="*70)
    print("FIXING ISOLATED METRO STATIONS")
    print("="*70)
    
    # Load data
    metro_nodes = pd.read_csv(f'{input_dir}/metro_nodes.csv', dtype={'name': str})
    metro_edges = pd.read_csv(f'{input_dir}/metro_edges.csv')
    
    print(f"Loaded {len(metro_nodes)} stations, {len(metro_edges)} edges")
    
    # Build graph
    G = nx.Graph()
    for _, node in metro_nodes.iterrows():
        G.add_node(node['node_id'])
    
    for _, edge in metro_edges.iterrows():
        if edge['u'] != edge['v']:
            G.add_edge(edge['u'], edge['v'])
    
    # Find isolated stations
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    connected = [n for n in G.nodes() if G.degree(n) > 0]
    
    print(f"Isolated stations: {len(isolated)}")
    print(f"Connected stations: {len(connected)}")
    
    # Create new edges to connect isolated stations
    new_edges = []
    edge_id = len(metro_edges)
    
    for node_id in isolated:
        node = metro_nodes[metro_nodes['node_id'] == node_id].iloc[0]
        station_name = str(node['name'])
        
        # First, check if this is a duplicate interchange station
        match = find_interchange_match(station_name, metro_nodes, connected)
        
        if match:
            # This is an interchange duplicate - connect with short transfer
            match_node = metro_nodes[metro_nodes['node_id'] == match].iloc[0]
            dist = haversine_distance(node['y'], node['x'], match_node['y'], match_node['x'])
            
            # Interchange transfer time (walking within station)
            transfer_time = max(60, dist / 1.2)  # Walking at 1.2 m/s, min 1 min
            
            print(f"  Interchange: {station_name} <-> {match_node['name']} ({dist:.0f}m)")
            
            # Create bidirectional edge
            for u, v in [(node_id, match), (match, node_id)]:
                new_edges.append({
                    'edge_id': f'metro_{edge_id}',
                    'layer': 1,
                    'u': u,
                    'v': v,
                    'key': 0,
                    'length_m': dist,
                    'mode': 'metro',
                    'speed_mps': 1.2,
                    'travel_time_s': transfer_time,
                    'edge_type': 'interchange_connection',
                    'properties': json.dumps({
                        'type': 'interchange',
                        'from_station': station_name,
                        'to_station': str(match_node['name'])
                    })
                })
                edge_id += 1
        else:
            # Find nearest connected station
            nearest, dist = find_nearest_connected_station(node_id, metro_nodes, metro_edges, G, max_distance=3000)
            
            if nearest:
                nearest_node = metro_nodes[metro_nodes['node_id'] == nearest].iloc[0]
                
                # Calculate travel time (metro speed ~60 km/h = 16.67 m/s)
                travel_time = dist / 16.67
                
                print(f"  Connecting: {station_name} <-> {nearest_node['name']} ({dist:.0f}m)")
                
                # Create bidirectional edge
                for u, v in [(node_id, nearest), (nearest, node_id)]:
                    new_edges.append({
                        'edge_id': f'metro_{edge_id}',
                        'layer': 1,
                        'u': u,
                        'v': v,
                        'key': 0,
                        'length_m': dist,
                        'mode': 'metro',
                        'speed_mps': 16.67,
                        'travel_time_s': travel_time,
                        'edge_type': 'inferred_connection',
                        'properties': json.dumps({
                            'type': 'inferred',
                            'reason': 'connect_isolated',
                            'from_station': station_name,
                            'to_station': str(nearest_node['name'])
                        })
                    })
                    edge_id += 1
            else:
                print(f"  WARNING: Could not connect {station_name} - no nearby station found")
    
    # Add new edges to dataframe
    if new_edges:
        new_edges_df = pd.DataFrame(new_edges)
        metro_edges = pd.concat([metro_edges, new_edges_df], ignore_index=True)
        print(f"\nAdded {len(new_edges)} new edges")
    
    # Verify connectivity
    G_new = nx.Graph()
    for _, node in metro_nodes.iterrows():
        G_new.add_node(node['node_id'])
    for _, edge in metro_edges.iterrows():
        if edge['u'] != edge['v']:
            G_new.add_edge(edge['u'], edge['v'])
    
    still_isolated = [n for n in G_new.nodes() if G_new.degree(n) == 0]
    components = list(nx.connected_components(G_new))
    
    print(f"\n=== AFTER FIX ===")
    print(f"Total edges: {len(metro_edges)}")
    print(f"Connected components: {len(components)}")
    print(f"Still isolated: {len(still_isolated)}")
    
    if len(components) == 1:
        print("✓ Metro network is fully connected!")
    
    # Save
    metro_edges.to_csv(f'{output_dir}/metro_edges.csv', index=False)
    print(f"\nSaved to {output_dir}/metro_edges.csv")
    
    return metro_nodes, metro_edges


def create_walk_bridge_transfers(data_dir='data/multilayer', output_dir='data/multilayer'):
    """
    Create transfers using walk layer as a bridge between road and metro.
    
    The idea:
    - Road nodes connect to nearby Walk nodes (park and walk)
    - Walk nodes connect to nearby Metro stations (walk to metro entry)
    - This creates: Road -> Walk -> Metro path
    
    This handles the fact that metro stations are not exactly on road nodes.
    """
    print("\n" + "="*70)
    print("CREATING WALK-BRIDGE TRANSFERS")
    print("="*70)
    
    # Load all layers from final data
    all_nodes = pd.read_csv(f'{data_dir}/../final/nodes_final.csv', dtype={'name': str}, low_memory=False)
    road_nodes = all_nodes[all_nodes['layer'] == 0].copy()
    metro_nodes = pd.read_csv(f'{data_dir}/metro_nodes.csv', dtype={'name': str})
    
    print(f"Road nodes: {len(road_nodes)}")
    print(f"Metro nodes: {len(metro_nodes)}")
    
    transfers = []
    transfer_id = 0
    
    # Parameters
    METRO_WALK_RADIUS = 500  # meters - max distance to walk to metro
    WALK_SPEED = 1.4  # m/s (about 5 km/h walking speed)
    METRO_ENTRY_TIME = 60  # seconds to enter metro (ticket, security, etc.)
    METRO_EXIT_TIME = 30  # seconds to exit metro
    
    # Build spatial index for road nodes
    print("\nBuilding spatial index for road nodes...")
    from scipy.spatial import cKDTree
    
    road_coords = road_nodes[['x', 'y']].values
    road_tree = cKDTree(road_coords)
    
    # For each metro station, find nearby road/walk nodes
    print("Creating walk-to-metro transfers...")
    
    for _, metro in metro_nodes.iterrows():
        metro_id = metro['node_id']
        metro_x, metro_y = metro['x'], metro['y']
        metro_name = metro['name']
        
        # Find road nodes within radius (using approximate degree conversion)
        # 1 degree lat ≈ 111km, 1 degree lon ≈ 85km at Delhi latitude
        search_radius_deg = METRO_WALK_RADIUS / 85000  # approximate
        
        nearby_indices = road_tree.query_ball_point([metro_x, metro_y], search_radius_deg)
        
        connected_count = 0
        for idx in nearby_indices:
            road_node = road_nodes.iloc[idx]
            road_id = road_node['node_id']
            
            # Calculate actual distance
            dist = haversine_distance(metro_y, metro_x, road_node['y'], road_node['x'])
            
            if dist <= METRO_WALK_RADIUS:
                walk_time = dist / WALK_SPEED
                
                # Create walk node id (same position as road node)
                walk_id = road_id.replace('road_', 'walk_')
                
                # Transfer: Walk -> Metro (entry)
                transfers.append({
                    'transfer_id': f'transfer_{transfer_id}',
                    'from_node': walk_id,
                    'to_node': metro_id,
                    'from_layer': 2,  # Walk
                    'to_layer': 1,    # Metro
                    'transfer_time_s': walk_time + METRO_ENTRY_TIME,
                    'transfer_type': 'walk_to_metro',
                    'distance_m': dist,
                    'properties': json.dumps({
                        'metro_station': metro_name,
                        'walk_time_s': walk_time,
                        'entry_time_s': METRO_ENTRY_TIME
                    })
                })
                transfer_id += 1
                
                # Transfer: Metro -> Walk (exit)
                transfers.append({
                    'transfer_id': f'transfer_{transfer_id}',
                    'from_node': metro_id,
                    'to_node': walk_id,
                    'from_layer': 1,  # Metro
                    'to_layer': 2,    # Walk
                    'transfer_time_s': walk_time + METRO_EXIT_TIME,
                    'transfer_type': 'metro_to_walk',
                    'distance_m': dist,
                    'properties': json.dumps({
                        'metro_station': metro_name,
                        'walk_time_s': walk_time,
                        'exit_time_s': METRO_EXIT_TIME
                    })
                })
                transfer_id += 1
                
                # Transfer: Road -> Metro (drive + park + enter)
                transfers.append({
                    'transfer_id': f'transfer_{transfer_id}',
                    'from_node': road_id,
                    'to_node': metro_id,
                    'from_layer': 0,  # Road
                    'to_layer': 1,    # Metro
                    'transfer_time_s': 120 + METRO_ENTRY_TIME,  # 2 min to park + entry
                    'transfer_type': 'metro_entry',
                    'distance_m': dist,
                    'properties': json.dumps({
                        'metro_station': metro_name,
                        'parking_time_s': 120,
                        'entry_time_s': METRO_ENTRY_TIME
                    })
                })
                transfer_id += 1
                
                # Transfer: Metro -> Road (exit + get vehicle)
                transfers.append({
                    'transfer_id': f'transfer_{transfer_id}',
                    'from_node': metro_id,
                    'to_node': road_id,
                    'from_layer': 1,  # Metro
                    'to_layer': 0,    # Road
                    'transfer_time_s': METRO_EXIT_TIME + 60,  # exit + 1 min to get vehicle
                    'transfer_type': 'metro_exit',
                    'distance_m': dist,
                    'properties': json.dumps({
                        'metro_station': metro_name,
                        'exit_time_s': METRO_EXIT_TIME,
                        'vehicle_time_s': 60
                    })
                })
                transfer_id += 1
                
                connected_count += 1
        
        if connected_count > 0:
            print(f"  {metro_name}: connected to {connected_count} road/walk nodes")
    
    # Create road <-> walk transfers for all road nodes
    print("\nCreating road-walk transfers...")
    
    for _, road in road_nodes.iterrows():
        road_id = road['node_id']
        walk_id = road_id.replace('road_', 'walk_')
        
        # Road -> Walk (park and walk)
        transfers.append({
            'transfer_id': f'transfer_{transfer_id}',
            'from_node': road_id,
            'to_node': walk_id,
            'from_layer': 0,
            'to_layer': 2,
            'transfer_time_s': 60,  # 1 min to park
            'transfer_type': 'park_and_walk',
            'distance_m': 0,
            'properties': json.dumps({'type': 'mode_switch'})
        })
        transfer_id += 1
        
        # Walk -> Road (get vehicle)
        transfers.append({
            'transfer_id': f'transfer_{transfer_id}',
            'from_node': walk_id,
            'to_node': road_id,
            'from_layer': 2,
            'to_layer': 0,
            'transfer_time_s': 30,  # 30s to get vehicle
            'transfer_type': 'get_vehicle',
            'distance_m': 0,
            'properties': json.dumps({'type': 'mode_switch'})
        })
        transfer_id += 1
    
    # Create dataframe
    transfers_df = pd.DataFrame(transfers)
    
    print(f"\n=== TRANSFER SUMMARY ===")
    print(f"Total transfers: {len(transfers_df)}")
    for t_type in transfers_df['transfer_type'].unique():
        count = len(transfers_df[transfers_df['transfer_type'] == t_type])
        print(f"  {t_type}: {count}")
    
    # Save
    transfers_df.to_csv(f'{output_dir}/transfers.csv', index=False)
    print(f"\nSaved to {output_dir}/transfers.csv")
    
    return transfers_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_dir = os.path.join(project_root, 'data/multilayer')
    
    # Step 1: Fix isolated stations
    fix_isolated_stations(data_dir, data_dir)
    
    # Step 2: Create proper transfers with walk bridge
    create_walk_bridge_transfers(data_dir, data_dir)
    
    print("\n" + "="*70)
    print("DONE! Now run merge_all_layers.py to rebuild the final network.")
    print("="*70)
