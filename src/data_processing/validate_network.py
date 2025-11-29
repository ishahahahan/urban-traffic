"""
validate_network.py
Validate the multi-layer network for consistency and data quality.
"""

import pandas as pd
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_multilayer_network(data_dir):
    """
    Validate network structure and data quality.
    
    Parameters:
    -----------
    data_dir : str
        Path to data directory (e.g., 'data/final')
        
    Returns:
    --------
    bool : True if validation passed, False otherwise
    """
    
    print("="*70)
    print("VALIDATING MULTI-LAYER NETWORK")
    print("="*70)
    
    issues = []
    warnings = []
    
    # Load data
    try:
        nodes = pd.read_csv(f'{data_dir}/nodes_final.csv')
        print(f"\n✓ Loaded nodes: {len(nodes):,}")
    except FileNotFoundError:
        print(f"\n✗ Error: nodes_final.csv not found in {data_dir}")
        return False
    
    try:
        edges = pd.read_csv(f'{data_dir}/edges_final.csv')
        print(f"✓ Loaded edges: {len(edges):,}")
    except FileNotFoundError:
        print(f"✗ Error: edges_final.csv not found in {data_dir}")
        return False
    
    try:
        transfers = pd.read_csv(f'{data_dir}/transfers_final.csv')
        print(f"✓ Loaded transfers: {len(transfers):,}")
    except FileNotFoundError:
        print(f"✗ Warning: transfers_final.csv not found")
        transfers = pd.DataFrame()
        warnings.append("No transfers file found")
    
    # Check for time-series data
    timeseries_file = f'{data_dir}/multimodal_timeseries.parquet'
    if os.path.exists(timeseries_file):
        try:
            timeseries = pd.read_parquet(timeseries_file)
            print(f"✓ Loaded time-series: {len(timeseries):,} records")
        except Exception as e:
            print(f"✗ Error loading time-series: {e}")
            timeseries = pd.DataFrame()
            warnings.append("Could not load time-series data")
    else:
        print(f"✗ Warning: multimodal_timeseries.parquet not found")
        timeseries = pd.DataFrame()
        warnings.append("No time-series file found")
    
    print("\n" + "-"*70)
    print("RUNNING VALIDATION CHECKS")
    print("-"*70)
    
    # Check 1: All edge nodes exist
    print("\n[1] Checking edge node references...")
    all_node_ids = set(nodes['node_id'])
    
    missing_u = set(edges['u']) - all_node_ids
    missing_v = set(edges['v']) - all_node_ids
    
    if missing_u or missing_v:
        issues.append(f"Missing nodes: {len(missing_u)} in 'u', {len(missing_v)} in 'v'")
        print(f"  ✗ {len(missing_u)} edges reference non-existent 'u' nodes")
        print(f"  ✗ {len(missing_v)} edges reference non-existent 'v' nodes")
        if len(missing_u) <= 5:
            print(f"    Missing 'u' nodes: {list(missing_u)[:5]}")
        if len(missing_v) <= 5:
            print(f"    Missing 'v' nodes: {list(missing_v)[:5]}")
    else:
        print("  ✓ All edges reference valid nodes")
    
    # Check 2: All transfer nodes exist
    if len(transfers) > 0:
        print("\n[2] Checking transfer node references...")
        missing_from = set(transfers['from_node']) - all_node_ids
        missing_to = set(transfers['to_node']) - all_node_ids
        
        if missing_from or missing_to:
            issues.append(f"Missing transfer nodes: {len(missing_from)} from, {len(missing_to)} to")
            print(f"  ✗ {len(missing_from)} transfers reference non-existent 'from_node'")
            print(f"  ✗ {len(missing_to)} transfers reference non-existent 'to_node'")
        else:
            print("  ✓ All transfers reference valid nodes")
    else:
        print("\n[2] Skipping transfer validation (no transfers data)")
    
    # Check 3: Time-series covers all edges
    if len(timeseries) > 0:
        print("\n[3] Checking time-series coverage...")
        edges_with_data = set(timeseries['edge_id'])
        all_edge_ids = set(edges['edge_id'])
        
        missing_data = all_edge_ids - edges_with_data
        extra_data = edges_with_data - all_edge_ids
        
        if missing_data:
            warnings.append(f"{len(missing_data)} edges missing time-series data")
            print(f"  ⚠ {len(missing_data)} edges have no time-series data")
        else:
            print("  ✓ All edges have time-series data")
        
        if extra_data:
            warnings.append(f"{len(extra_data)} time-series records reference non-existent edges")
            print(f"  ⚠ {len(extra_data)} time-series records reference non-existent edges")
    else:
        print("\n[3] Skipping time-series validation (no time-series data)")
    
    # Check 4: Layer consistency
    print("\n[4] Checking layer consistency...")
    node_layers = nodes['layer'].unique()
    edge_layers = edges['layer'].unique()
    
    print(f"  Node layers: {sorted(node_layers)}")
    print(f"  Edge layers: {sorted(edge_layers)}")
    
    # Check nodes by layer
    for layer in sorted(node_layers):
        layer_nodes = len(nodes[nodes['layer'] == layer])
        print(f"  Layer {layer}: {layer_nodes} nodes")
    
    # Check 5: Metro network connectivity
    print("\n[5] Checking metro network connectivity...")
    metro_edges = edges[edges['layer'] == 1]
    metro_nodes_in_edges = set(metro_edges['u']) | set(metro_edges['v'])
    metro_nodes_total = nodes[nodes['layer'] == 1]
    
    if len(metro_nodes_total) > 0:
        metro_station_ids = set(metro_nodes_total['node_id'])
        isolated_metros = metro_station_ids - metro_nodes_in_edges
        
        if len(isolated_metros) > 0:
            warnings.append(f"{len(isolated_metros)} isolated metro stations")
            print(f"  ⚠ {len(isolated_metros)} metro stations are not connected to any edge")
            if len(isolated_metros) <= 5:
                print(f"    Isolated: {list(isolated_metros)[:5]}")
        else:
            print("  ✓ All metro stations are connected")
        
        print(f"  Total metro stations: {len(metro_nodes_total)}")
        print(f"  Metro edges: {len(metro_edges)}")
    else:
        print("  No metro layer found")
    
    # Check 6: Transfer coverage
    if len(transfers) > 0:
        print("\n[6] Checking transfer coverage...")
        metro_stations = set(nodes[nodes['layer'] == 1]['node_id'])
        
        if len(metro_stations) > 0:
            # Check which metro stations have transfers
            metros_with_transfers_to = set(transfers[transfers['to_layer'] == 1]['to_node'])
            metros_with_transfers_from = set(transfers[transfers['from_layer'] == 1]['from_node'])
            metros_with_transfers = metros_with_transfers_to | metros_with_transfers_from
            
            metros_without_access = metro_stations - metros_with_transfers
            coverage = len(metros_with_transfers) / len(metro_stations) * 100 if metro_stations else 0
            
            print(f"  Transfer coverage: {coverage:.1f}%")
            print(f"  Metro stations with transfers: {len(metros_with_transfers)}")
            
            if coverage < 80:
                warnings.append(f"Only {coverage:.1f}% of metro stations have transfers")
                print(f"  ⚠ {len(metros_without_access)} metro stations have no road access")
            else:
                print("  ✓ Good transfer coverage")
        else:
            print("  No metro stations found")
    else:
        print("\n[6] Skipping transfer coverage (no transfers data)")
    
    # Check 7: Data quality checks
    print("\n[7] Checking data quality...")
    
    # Check for negative or zero values
    neg_lengths = edges[edges['length_m'] <= 0]
    if len(neg_lengths) > 0:
        issues.append(f"{len(neg_lengths)} edges with non-positive length")
        print(f"  ✗ {len(neg_lengths)} edges have length <= 0")
    else:
        print("  ✓ All edges have positive length")
    
    neg_speeds = edges[edges['speed_mps'] <= 0]
    if len(neg_speeds) > 0:
        issues.append(f"{len(neg_speeds)} edges with non-positive speed")
        print(f"  ✗ {len(neg_speeds)} edges have speed <= 0")
    else:
        print("  ✓ All edges have positive speed")
    
    neg_times = edges[edges['travel_time_s'] <= 0]
    if len(neg_times) > 0:
        issues.append(f"{len(neg_times)} edges with non-positive travel time")
        print(f"  ✗ {len(neg_times)} edges have travel_time <= 0")
    else:
        print("  ✓ All edges have positive travel time")
    
    # Check 8: Coordinate bounds (Delhi area)
    print("\n[8] Checking coordinate bounds...")
    delhi_bounds = {
        'min_x': 76.8,
        'max_x': 77.5,
        'min_y': 28.4,
        'max_y': 28.9
    }
    
    out_of_bounds = nodes[
        (nodes['x'] < delhi_bounds['min_x']) | 
        (nodes['x'] > delhi_bounds['max_x']) |
        (nodes['y'] < delhi_bounds['min_y']) | 
        (nodes['y'] > delhi_bounds['max_y'])
    ]
    
    if len(out_of_bounds) > 0:
        warnings.append(f"{len(out_of_bounds)} nodes outside expected Delhi bounds")
        print(f"  ⚠ {len(out_of_bounds)} nodes are outside expected Delhi area bounds")
    else:
        print("  ✓ All nodes within expected Delhi bounds")
    
    # Summary
    print("\n" + "="*70)
    
    if issues:
        print("VALIDATION FAILED")
        print(f"\nCritical Issues: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ VALIDATION PASSED - No critical issues found!")
    
    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("="*70)
    
    # Print network statistics
    print("\nNETWORK STATISTICS")
    print("-"*40)
    print(f"Total nodes: {len(nodes):,}")
    print(f"Total edges: {len(edges):,}")
    print(f"Total transfers: {len(transfers):,}")
    
    print("\nNodes by layer:")
    for layer in sorted(nodes['layer'].unique()):
        layer_name = {0: 'Road', 1: 'Metro', 2: 'Walk'}.get(layer, f'Layer {layer}')
        count = len(nodes[nodes['layer'] == layer])
        print(f"  {layer_name}: {count:,}")
    
    print("\nEdges by mode:")
    for mode in edges['mode'].unique():
        count = len(edges[edges['mode'] == mode])
        print(f"  {mode}: {count:,}")
    
    if len(timeseries) > 0:
        print(f"\nTime-series records: {len(timeseries):,}")
        print(f"Time range: {timeseries['timestamp'].min()} to {timeseries['timestamp'].max()}")
    
    print("-"*40)
    
    return len(issues) == 0


def generate_network_summary(data_dir, output_file=None):
    """
    Generate a JSON summary of the network.
    
    Parameters:
    -----------
    data_dir : str
        Path to data directory
    output_file : str, optional
        Output JSON file path
        
    Returns:
    --------
    dict : Network summary
    """
    
    nodes = pd.read_csv(f'{data_dir}/nodes_final.csv')
    edges = pd.read_csv(f'{data_dir}/edges_final.csv')
    
    try:
        transfers = pd.read_csv(f'{data_dir}/transfers_final.csv')
    except:
        transfers = pd.DataFrame()
    
    summary = {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'total_transfers': len(transfers),
        'nodes_by_layer': {},
        'edges_by_mode': {},
        'coordinate_bounds': {
            'min_x': float(nodes['x'].min()),
            'max_x': float(nodes['x'].max()),
            'min_y': float(nodes['y'].min()),
            'max_y': float(nodes['y'].max())
        },
        'edge_statistics': {
            'avg_length_m': float(edges['length_m'].mean()),
            'total_length_km': float(edges['length_m'].sum() / 1000),
            'avg_speed_mps': float(edges['speed_mps'].mean()),
            'avg_travel_time_s': float(edges['travel_time_s'].mean())
        }
    }
    
    # Nodes by layer
    for layer in nodes['layer'].unique():
        layer_name = {0: 'road', 1: 'metro', 2: 'walk'}.get(layer, f'layer_{layer}')
        summary['nodes_by_layer'][layer_name] = int(len(nodes[nodes['layer'] == layer]))
    
    # Edges by mode
    for mode in edges['mode'].unique():
        summary['edges_by_mode'][mode] = int(len(edges[edges['mode'] == mode]))
    
    # Check for time-series
    timeseries_file = f'{data_dir}/multimodal_timeseries.parquet'
    if os.path.exists(timeseries_file):
        timeseries = pd.read_parquet(timeseries_file)
        summary['timeseries'] = {
            'total_records': len(timeseries),
            'unique_edges': int(timeseries['edge_id'].nunique()),
            'time_range': {
                'start': str(timeseries['timestamp'].min()),
                'end': str(timeseries['timestamp'].max())
            }
        }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Network summary saved to {output_file}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate multi-layer network')
    parser.add_argument('--data-dir', default='data/final', 
                        help='Path to data directory')
    parser.add_argument('--summary', action='store_true',
                        help='Generate network summary JSON')
    
    args = parser.parse_args()
    
    # Run validation
    valid = validate_multilayer_network(args.data_dir)
    
    # Generate summary if requested
    if args.summary:
        generate_network_summary(args.data_dir, f'{args.data_dir}/network_summary.json')
    
    sys.exit(0 if valid else 1)
