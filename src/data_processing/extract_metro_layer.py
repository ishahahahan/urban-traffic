"""
Extract metro/subway network from OpenStreetMap or create synthetic metro layer.
This forms Layer 1 (metro network) of the multi-layer graph.

Supports two modes:
1. Synthetic mode: Creates a simplified metro network for testing
2. Real mode: Extracts actual Delhi Metro data from OpenStreetMap
"""

import pandas as pd
import numpy as np
import json
from math import radians, sin, cos, sqrt, atan2
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth using Haversine formula.
    
    Returns:
    --------
    float : Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def create_metro_network(output_dir, use_real_data=False, place_name='Delhi, India'):
    """
    Create metro network - either synthetic or from real OSM data.
    
    Parameters:
    -----------
    output_dir : str
        Directory for output files
    use_real_data : bool
        If True, extract real data from OpenStreetMap
        If False, use synthetic data (default)
    place_name : str
        Place name for OSM extraction (only used if use_real_data=True)
        
    Returns:
    --------
    tuple : (metro_nodes_df, metro_edges_df)
    """
    
    if use_real_data:
        try:
            from data_processing.extract_real_metro_data import extract_real_metro
            print("Extracting real Delhi Metro data from OpenStreetMap...")
            return extract_real_metro(place_name=place_name, output_dir=output_dir)
        except ImportError as e:
            print(f"Warning: Could not import real metro extraction module: {e}")
            print("Falling back to synthetic metro data...")
        except Exception as e:
            print(f"Warning: Error extracting real metro data: {e}")
            print("Falling back to synthetic metro data...")
    
    return create_synthetic_metro(output_dir)


def create_synthetic_metro(output_dir):
    """
    Create synthetic metro network for Delhi near Connaught Place.
    
    Parameters:
    -----------
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    tuple : (metro_nodes_df, metro_edges_df)
    """
    
    print("Creating synthetic metro network...")
    
    # Define synthetic metro stations
    # These are realistic Delhi Metro stations near Connaught Place
    synthetic_stations = [
        {'name': 'Rajiv Chowk', 'x': 77.2088, 'y': 28.6329, 'line': 'Blue', 'interchange': True},
        {'name': 'Barakhamba Road', 'x': 77.2183, 'y': 28.6304, 'line': 'Blue', 'interchange': False},
        {'name': 'Mandi House', 'x': 77.2346, 'y': 28.6261, 'line': 'Blue', 'interchange': True},
        {'name': 'Patel Chowk', 'x': 77.2079, 'y': 28.6217, 'line': 'Yellow', 'interchange': False},
        {'name': 'Central Secretariat', 'x': 77.2095, 'y': 28.6157, 'line': 'Yellow', 'interchange': True},
        {'name': 'Shivaji Stadium', 'x': 77.2025, 'y': 28.6433, 'line': 'Orange', 'interchange': False},
        {'name': 'Ramakrishna Ashram Marg', 'x': 77.1959, 'y': 28.6405, 'line': 'Blue', 'interchange': False},
        {'name': 'Janpath', 'x': 77.2088, 'y': 28.6329, 'line': 'Yellow', 'interchange': True},
    ]
    
    # Create nodes
    metro_nodes = []
    for i, station in enumerate(synthetic_stations):
        metro_nodes.append({
            'node_id': f'metro_{i}',
            'layer': 1,
            'osmid': None,
            'x': station['x'],
            'y': station['y'],
            'node_type': 'metro_station',
            'name': station['name'],
            'properties': json.dumps({
                'line': station['line'],
                'interchange': station['interchange'],
                'synthetic': True
            })
        })
    
    # Create edges (connect sequential stations on same line)
    metro_edges = []
    edge_id = 0
    
    # Group stations by line
    from collections import defaultdict
    lines = defaultdict(list)
    for i, station in enumerate(synthetic_stations):
        lines[station['line']].append(i)
    
    # Connect stations within each line
    for line, station_indices in lines.items():
        for i in range(len(station_indices) - 1):
            idx1 = station_indices[i]
            idx2 = station_indices[i + 1]
            
            s1 = synthetic_stations[idx1]
            s2 = synthetic_stations[idx2]
            
            # Calculate distance
            distance = haversine_distance(s1['y'], s1['x'], s2['y'], s2['x'])
            
            # Metro speed
            speed_mps = 16.67  # 60 km/h
            travel_time = distance / speed_mps
            
            # Create bidirectional edges
            for u, v in [(idx1, idx2), (idx2, idx1)]:
                metro_edges.append({
                    'edge_id': f'metro_{edge_id}',
                    'layer': 1,
                    'u': f'metro_{u}',
                    'v': f'metro_{v}',
                    'key': 0,
                    'length_m': distance,
                    'mode': 'metro',
                    'speed_mps': speed_mps,
                    'travel_time_s': travel_time,
                    'edge_type': 'metro_segment',
                    'properties': json.dumps({
                        'line': line,
                        'frequency_min': 5,
                        'synthetic': True
                    })
                })
                edge_id += 1
    
    # Save
    metro_nodes_df = pd.DataFrame(metro_nodes)
    metro_edges_df = pd.DataFrame(metro_edges)
    
    os.makedirs(output_dir, exist_ok=True)
    metro_nodes_df.to_csv(f'{output_dir}/metro_nodes.csv', index=False)
    metro_edges_df.to_csv(f'{output_dir}/metro_edges.csv', index=False)
    
    print(f"Created synthetic metro network:")
    print(f"  {len(metro_nodes)} stations")
    print(f"  {len(metro_edges)} connections")
    print(f"  Lines: {list(lines.keys())}")
    
    return metro_nodes_df, metro_edges_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract or create metro network')
    parser.add_argument('--real', action='store_true', 
                        help='Extract real data from OpenStreetMap')
    parser.add_argument('--place', default='Delhi, India',
                        help='Place name for OSM extraction')
    parser.add_argument('--output-dir', default='data/multilayer',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create metro network
    metro_nodes, metro_edges = create_metro_network(
        output_dir=args.output_dir,
        use_real_data=args.real,
        place_name=args.place
    )
    
    if metro_nodes is not None:
        print("\nMetro network extraction complete!")
        print(f"Stations: {len(metro_nodes)}")
        print(f"Connections: {len(metro_edges)}")
    else:
        print("\nFailed to create metro network")
