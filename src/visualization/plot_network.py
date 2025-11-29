"""
Visualization utilities for multi-layer transportation networks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
import json


# Layer color scheme
LAYER_COLORS = {
    0: '#FF6B6B',  # Road - red
    1: '#4ECDC4',  # Metro - teal
    2: "#2633A6",  # Walk - light teal
}

MODE_COLORS = {
    'car': '#FF6B6B',
    'metro': "#65CD4E",
    'walk': '#95E1D3',
    'auto': "#D71717",
}


def plot_network_layers(nodes_df, edges_df, figsize=(15, 12), 
                        show_layers=[0, 1, 2], alpha=0.6):
    """
    Plot all network layers on a single map (optimized version).
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    layer_names = {0: 'Road', 1: 'Metro', 2: 'Walk'}
    
    # Create node lookup dictionary for fast access
    node_coords = nodes_df.set_index('node_id')[['x', 'y']].to_dict('index')
    
    for layer in show_layers:
        layer_nodes = nodes_df[nodes_df['layer'] == layer]
        layer_edges = edges_df[edges_df['layer'] == layer]
        
        # Vectorized line creation
        if len(layer_edges) > 0:
            lines = [
                [(node_coords[u]['x'], node_coords[u]['y']),
                 (node_coords[v]['x'], node_coords[v]['y'])]
                for u, v in zip(layer_edges['u'], layer_edges['v'])
                if u in node_coords and v in node_coords
            ]
            
            if lines:
                lc = LineCollection(lines, colors=LAYER_COLORS[layer], 
                                  alpha=alpha, linewidths=1.5, 
                                  label=f'{layer_names[layer]} ({len(lines)} edges)')
                ax.add_collection(lc)
        
        # Plot nodes
        if len(layer_nodes) > 0:
            ax.scatter(layer_nodes['x'], layer_nodes['y'], 
                      c=LAYER_COLORS[layer], s=20, alpha=alpha*1.5, 
                      edgecolors='white', linewidths=0.5, zorder=3)
    
    # Metro stations with labels
    if 1 in show_layers:
        metro_nodes = nodes_df[nodes_df['layer'] == 1]
        if len(metro_nodes) > 0:
            # Plot all metro stations at once
            ax.scatter(metro_nodes['x'], metro_nodes['y'], c='#4ECDC4', s=100, 
                      marker='s', edgecolors='black', linewidths=2, zorder=5)
            
            # Only annotate stations with names
            named_stations = metro_nodes[metro_nodes['name'].notna() & (metro_nodes['name'] != '')]
            for _, node in named_stations.iterrows():
                ax.annotate(node['name'], (node['x'], node['y']), 
                          fontsize=8, xytext=(5, 5), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Multi-Layer Transportation Network', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig, ax


def plot_route(route, nodes_df, figsize=(12, 10)):
    """
    Visualize a multimodal route with segments highlighted.
    
    Parameters:
    -----------
    route : dict
        Route dictionary from MultimodalRouter
    nodes_df : DataFrame
        Node information
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all nodes in route
    route_nodes = nodes_df[nodes_df['node_id'].isin(route['path'])]
    
    # Plot base network (faded)
    ax.scatter(nodes_df['x'], nodes_df['y'], c='lightgray', s=5, alpha=0.3)
    
    # Plot route segments
    for i, segment in enumerate(route['segments']):
        seg_nodes = nodes_df[nodes_df['node_id'].isin(segment['nodes'])]
        
        # Plot segment edges
        lines = []
        for j in range(len(segment['nodes']) - 1):
            u_node = nodes_df[nodes_df['node_id'] == segment['nodes'][j]].iloc[0]
            v_node = nodes_df[nodes_df['node_id'] == segment['nodes'][j+1]].iloc[0]
            lines.append([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
        
        if lines:
            lc = LineCollection(lines, colors=MODE_COLORS[segment['mode']], 
                              linewidths=3, label=f"{segment['mode'].title()}: {segment['time_s']/60:.1f} min",
                              zorder=2)
            ax.add_collection(lc)
        
        # Plot segment nodes
        ax.scatter(seg_nodes['x'], seg_nodes['y'], 
                  c=MODE_COLORS[segment['mode']], s=30, 
                  edgecolors='white', linewidths=1, zorder=3)
    
    # Highlight start and end
    start_node = nodes_df[nodes_df['node_id'] == route['path'][0]].iloc[0]
    end_node = nodes_df[nodes_df['node_id'] == route['path'][-1]].iloc[0]
    
    ax.scatter(start_node['x'], start_node['y'], c='green', s=200, 
              marker='o', edgecolors='black', linewidths=2, 
              label='Start', zorder=5)
    ax.scatter(end_node['x'], end_node['y'], c='red', s=200, 
              marker='*', edgecolors='black', linewidths=2, 
              label='End', zorder=5)
    
    # Highlight transfers
    for transfer in route['transfers']:
        trans_node = nodes_df[nodes_df['node_id'] == transfer['from_node']].iloc[0]
        ax.scatter(trans_node['x'], trans_node['y'], c='orange', s=150, 
                  marker='D', edgecolors='black', linewidths=2, 
                  alpha=0.8, zorder=4)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Multimodal Route: {route["total_time_s"]/60:.1f} min, {route["num_transfers"]} transfers', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig, ax


def plot_traffic_heatmap(timeseries_df, edges_df, nodes_df, timestamp, 
                         layer=0, figsize=(12, 10)):
    """
    Plot traffic congestion heatmap at specific time.
    
    Parameters:
    -----------
    timeseries_df : DataFrame
        Time-series traffic data
    edges_df : DataFrame
        Edge information
    nodes_df : DataFrame
        Node information
    timestamp : str
        Time to visualize
    layer : int
        Which layer to show
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get traffic data at this time
    traffic = timeseries_df[timeseries_df['timestamp'] == timestamp]
    
    # Get edges for this layer
    layer_edges = edges_df[edges_df['layer'] == layer].copy()
    
    # Merge with traffic data
    layer_edges = layer_edges.merge(traffic[['edge_id', 'congestion_factor']], 
                                    on='edge_id', how='left')
    
    # Create line collection with colors based on congestion
    lines = []
    colors = []
    
    for _, edge in layer_edges.iterrows():
        u_node = nodes_df[nodes_df['node_id'] == edge['u']].iloc[0]
        v_node = nodes_df[nodes_df['node_id'] == edge['v']].iloc[0]
        lines.append([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
        
        # Color based on congestion
        cf = edge.get('congestion_factor', 1.0)
        colors.append(cf)
    
    # Normalize colors
    colors = np.array(colors)
    
    if len(lines) > 0:
        lc = LineCollection(lines, cmap='RdYlGn_r', linewidths=2)
        lc.set_array(colors)
        lc.set_clim(0.8, 2.0)
        ax.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Congestion Factor', fontsize=12)
    
    # Plot nodes
    layer_nodes = nodes_df[nodes_df['layer'] == layer]
    ax.scatter(layer_nodes['x'], layer_nodes['y'], c='gray', s=10, alpha=0.5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Traffic Congestion at {timestamp}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig, ax


def plot_time_series_comparison(timeseries_df, edge_ids, figsize=(14, 6)):
    """
    Plot travel time comparison for multiple edges over time.
    
    Parameters:
    -----------
    timeseries_df : DataFrame
        Time-series data
    edge_ids : list
        List of edge IDs to compare
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for edge_id in edge_ids:
        edge_data = timeseries_df[timeseries_df['edge_id'] == edge_id]
        
        if len(edge_data) > 0:
            mode = edge_data['mode'].iloc[0]
            ax.plot(edge_data['hour'], edge_data['travel_time_s'], 
                   marker='o', linewidth=2, label=f'{edge_id} ({mode})',
                   color=MODE_COLORS.get(mode, 'gray'))
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Travel Time (seconds)', fontsize=12)
    ax.set_title('Travel Time Variation Over 24 Hours', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    return fig, ax


def plot_transfers(nodes_df, transfers_df, figsize=(15, 12), sample_size=1000):
    """
    Visualize transfer connections between layers.
    
    Parameters:
    -----------
    nodes_df : DataFrame
        Node information with x, y, layer
    transfers_df : DataFrame
        Transfer information with from_node, to_node, transfer_type
    figsize : tuple
        Figure size
    sample_size : int
        Number of transfers to sample for visualization (to avoid clutter)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transfer type colors
    TRANSFER_COLORS = {
        'park_and_walk': '#FFA500',    # Orange
        'get_vehicle': '#FFD700',       # Gold
        'metro_entry': '#00CED1',       # Dark Cyan
        'metro_exit': '#20B2AA',        # Light Sea Green
        'walk_to_metro': '#9370DB',     # Medium Purple
        'metro_to_walk': '#BA55D3',     # Medium Orchid
    }
    
    # Create node lookup
    node_coords = nodes_df.set_index('node_id')[['x', 'y', 'layer']].to_dict('index')
    
    # Plot base nodes by layer
    layer_names = {0: 'Road', 1: 'Metro', 2: 'Walk'}
    for layer in [0, 1, 2]:
        layer_nodes = nodes_df[nodes_df['layer'] == layer]
        color = LAYER_COLORS[layer]
        size = 100 if layer == 1 else 5
        ax.scatter(layer_nodes['x'], layer_nodes['y'], 
                  c=color, s=size, alpha=0.3, label=f'{layer_names[layer]} nodes')
    
    # Sample transfers for visualization
    if len(transfers_df) > sample_size:
        sampled = transfers_df.sample(n=sample_size, random_state=42)
    else:
        sampled = transfers_df
    
    # Plot transfers by type
    for t_type in transfers_df['transfer_type'].unique():
        type_transfers = sampled[sampled['transfer_type'] == t_type]
        
        lines = []
        for _, t in type_transfers.iterrows():
            from_node = t['from_node']
            to_node = t['to_node']
            
            if from_node in node_coords and to_node in node_coords:
                lines.append([
                    (node_coords[from_node]['x'], node_coords[from_node]['y']),
                    (node_coords[to_node]['x'], node_coords[to_node]['y'])
                ])
        
        if lines:
            color = TRANSFER_COLORS.get(t_type, 'gray')
            lc = LineCollection(lines, colors=color, alpha=0.5, linewidths=0.5,
                              label=f'{t_type} ({len(type_transfers)} shown)')
            ax.add_collection(lc)
    
    # Highlight metro stations
    metro_nodes = nodes_df[nodes_df['layer'] == 1]
    ax.scatter(metro_nodes['x'], metro_nodes['y'], c='#4ECDC4', s=150, 
              marker='s', edgecolors='black', linewidths=2, zorder=5, label='Metro Stations')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Transfer Connections Between Network Layers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax


def plot_metro_catchment(nodes_df, transfers_df, metro_station_id, figsize=(12, 10)):
    """
    Visualize the catchment area of a metro station (all nodes that can transfer to it).
    
    Parameters:
    -----------
    nodes_df : DataFrame
        Node information
    transfers_df : DataFrame
        Transfer information
    metro_station_id : str
        The metro station node_id to visualize
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get metro station info
    metro_node = nodes_df[nodes_df['node_id'] == metro_station_id].iloc[0]
    station_name = metro_node['name'] if pd.notna(metro_node['name']) else metro_station_id
    
    # Find all nodes that can transfer to this metro station
    to_metro = transfers_df[transfers_df['to_node'] == metro_station_id]
    from_metro = transfers_df[transfers_df['from_node'] == metro_station_id]
    
    connected_nodes = set(to_metro['from_node'].tolist() + from_metro['to_node'].tolist())
    
    # Plot all nodes faded
    ax.scatter(nodes_df['x'], nodes_df['y'], c='lightgray', s=2, alpha=0.3)
    
    # Plot connected nodes
    connected_df = nodes_df[nodes_df['node_id'].isin(connected_nodes)]
    
    # Color by layer
    for layer in [0, 2]:  # Road and Walk
        layer_nodes = connected_df[connected_df['layer'] == layer]
        color = LAYER_COLORS[layer]
        label = 'Road access' if layer == 0 else 'Walk access'
        ax.scatter(layer_nodes['x'], layer_nodes['y'], c=color, s=20, 
                  alpha=0.7, label=f'{label} ({len(layer_nodes)})')
    
    # Plot metro station prominently
    ax.scatter(metro_node['x'], metro_node['y'], c='#4ECDC4', s=400, 
              marker='s', edgecolors='black', linewidths=3, zorder=5)
    ax.annotate(station_name, (metro_node['x'], metro_node['y']), 
               fontsize=12, fontweight='bold', xytext=(10, 10), 
               textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Catchment Area: {station_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example usage
    import os
    
    # Get the project root directory (two levels up from this file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Use absolute paths
    nodes = pd.read_csv(os.path.join(project_root, 'data/final/nodes_final.csv'), 
                        dtype={'name': str}, low_memory=False)
    edges = pd.read_csv(os.path.join(project_root, 'data/final/edges_final.csv'))
    transfers = pd.read_csv(os.path.join(project_root, 'data/final/transfers_final.csv'))
    
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Plot network layers
    print("1. Plotting network layers...")
    fig, ax = plot_network_layers(nodes, edges)
    plt.show()
    
    # 2. Plot transfers
    print("2. Plotting transfer connections...")
    fig, ax = plot_transfers(nodes, transfers, sample_size=2000)
    plt.show()
    
    # 3. Plot metro catchment for a sample station
    print("3. Plotting metro catchment area...")
    metro_nodes = nodes[nodes['layer'] == 1]
    # Use Rajiv Chowk as sample metro station
    rajiv_chowk = metro_nodes[metro_nodes['name'].str.contains('Rajiv Chowk', case=False, na=False)]
    if len(rajiv_chowk) > 0:
        sample_metro = rajiv_chowk.iloc[0]['node_id']
    else:
        sample_metro = metro_nodes[metro_nodes['name'].notna()].iloc[10]['node_id']
    fig, ax = plot_metro_catchment(nodes, transfers, sample_metro)
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("TRANSFER SUMMARY")
    print("="*50)
    print(f"Total transfers: {len(transfers)}")
    print("\nBy type:")
    for t_type in transfers['transfer_type'].unique():
        count = len(transfers[transfers['transfer_type'] == t_type])
        print(f"  {t_type}: {count:,}")
    print("="*50)
