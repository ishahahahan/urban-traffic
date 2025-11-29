"""
Load and manage multi-layer transportation network graph.
Provides unified interface for accessing nodes, edges, and transfers.
"""

import pandas as pd
import networkx as nx
import os


class MultiLayerGraph:
    """
    Manages multi-layer transportation network.
    Each layer is a separate NetworkX graph connected via transfer edges.
    """
    
    def __init__(self, nodes_file, edges_file, transfers_file, timeseries_file=None):
        """
        Initialize multi-layer graph.
        
        Parameters:
        -----------
        nodes_file : str
            Path to nodes_final.csv
        edges_file : str
            Path to edges_final.csv
        transfers_file : str
            Path to transfers_final.csv
        timeseries_file : str, optional
            Path to multimodal_timeseries.parquet
        """
        
        print("Loading multi-layer network...")
        
        # Load data
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.transfers_df = pd.read_csv(transfers_file)
        
        print(f"  Nodes: {len(self.nodes_df):,}")
        print(f"  Edges: {len(self.edges_df):,}")
        print(f"  Transfers: {len(self.transfers_df):,}")
        
        # Load timeseries if provided
        self.timeseries_df = None
        if timeseries_file and os.path.exists(timeseries_file):
            self.timeseries_df = pd.read_parquet(timeseries_file)
            print(f"  Time-series records: {len(self.timeseries_df):,}")
        
        # Create separate graphs for each layer
        self.layers = {}
        self._build_layer_graphs()
        
        # Create unified graph (optional, for visualization)
        self.unified_graph = None
        
        print("Multi-layer network loaded successfully!")
    
    def _build_layer_graphs(self):
        """Build separate NetworkX graphs for each layer"""
        
        layer_ids = self.nodes_df['layer'].unique()
        
        for layer_id in layer_ids:
            # Get nodes and edges for this layer
            layer_nodes = self.nodes_df[self.nodes_df['layer'] == layer_id]
            layer_edges = self.edges_df[self.edges_df['layer'] == layer_id]
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for _, node in layer_nodes.iterrows():
                G.add_node(
                    node['node_id'],
                    layer=node['layer'],
                    x=node['x'],
                    y=node['y'],
                    node_type=node['node_type'],
                    name=node['name']
                )
            
            # Add edges
            for _, edge in layer_edges.iterrows():
                G.add_edge(
                    edge['u'],
                    edge['v'],
                    edge_id=edge['edge_id'],
                    layer=edge['layer'],
                    length_m=edge['length_m'],
                    mode=edge['mode'],
                    speed_mps=edge['speed_mps'],
                    travel_time_s=edge['travel_time_s'],
                    weight=edge['travel_time_s']  # Default weight
                )
            
            self.layers[int(layer_id)] = G
            
            print(f"  Layer {layer_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    def build_unified_graph(self, include_transfers=True):
        """
        Build a single unified graph containing all layers.
        
        Parameters:
        -----------
        include_transfers : bool
            Whether to include inter-layer transfer edges
            
        Returns:
        --------
        nx.DiGraph : Unified graph
        """
        
        print("Building unified graph...")
        
        G = nx.DiGraph()
        
        # Add all nodes from all layers
        for _, node in self.nodes_df.iterrows():
            G.add_node(
                node['node_id'],
                layer=node['layer'],
                x=node['x'],
                y=node['y'],
                node_type=node['node_type'],
                name=node['name']
            )
        
        # Add all edges from all layers
        for _, edge in self.edges_df.iterrows():
            G.add_edge(
                edge['u'],
                edge['v'],
                edge_id=edge['edge_id'],
                layer=edge['layer'],
                length_m=edge['length_m'],
                mode=edge['mode'],
                travel_time_s=edge['travel_time_s'],
                weight=edge['travel_time_s']
            )
        
        # Add transfer edges if requested
        if include_transfers:
            for _, transfer in self.transfers_df.iterrows():
                G.add_edge(
                    transfer['from_node'],
                    transfer['to_node'],
                    transfer_id=transfer['transfer_id'],
                    from_layer=transfer['from_layer'],
                    to_layer=transfer['to_layer'],
                    transfer_time_s=transfer['transfer_time_s'],
                    transfer_type=transfer['transfer_type'],
                    weight=transfer['transfer_time_s'],
                    is_transfer=True
                )
        
        self.unified_graph = G
        
        print(f"Unified graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_travel_times_at_time(self, timestamp):
        """
        Get travel times for all edges at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : str or pd.Timestamp
            Time to query
            
        Returns:
        --------
        dict : {edge_id: travel_time_s}
        """
        
        if self.timeseries_df is None:
            print("Warning: No time-series data loaded")
            return {}
        
        # Convert to timestamp if string
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        # Get data for this timestamp
        snapshot = self.timeseries_df[self.timeseries_df['timestamp'] == timestamp]
        
        if len(snapshot) == 0:
            print(f"Warning: No data found for timestamp {timestamp}")
            return {}
        
        # Create dictionary
        travel_times = dict(zip(snapshot['edge_id'], snapshot['travel_time_s']))
        
        return travel_times
    
    def update_edge_weights(self, graph, travel_times):
        """
        Update edge weights in graph based on time-dependent travel times.
        
        Parameters:
        -----------
        graph : nx.DiGraph
            Graph to update
        travel_times : dict
            {edge_id: travel_time_s}
            
        Returns:
        --------
        None (modifies graph in place)
        """
        
        updated_count = 0
        
        for u, v, data in graph.edges(data=True):
            edge_id = data.get('edge_id')
            
            if edge_id and edge_id in travel_times:
                graph[u][v]['weight'] = travel_times[edge_id]
                graph[u][v]['travel_time_s'] = travel_times[edge_id]
                updated_count += 1
        
        return updated_count
    
    def get_node_by_location(self, x, y, layer=None, max_distance=0.001):
        """
        Find nearest node to a given location.
        
        Parameters:
        -----------
        x : float
            Longitude
        y : float
            Latitude
        layer : int, optional
            Restrict search to specific layer
        max_distance : float
            Maximum distance in degrees (~111m per 0.001 deg at equator)
            
        Returns:
        --------
        str : node_id of nearest node, or None if not found
        """
        
        nodes = self.nodes_df
        
        if layer is not None:
            nodes = nodes[nodes['layer'] == layer]
        
        # Calculate distances
        nodes = nodes.copy()
        nodes['dist'] = ((nodes['x'] - x)**2 + (nodes['y'] - y)**2)**0.5
        
        # Filter by max distance
        nodes = nodes[nodes['dist'] <= max_distance]
        
        if len(nodes) == 0:
            return None
        
        # Return closest
        return nodes.loc[nodes['dist'].idxmin(), 'node_id']


if __name__ == "__main__":
    # Example usage
    mlg = MultiLayerGraph(
        nodes_file='data/final/nodes_final.csv',
        edges_file='data/final/edges_final.csv',
        transfers_file='data/final/transfers_final.csv',
        timeseries_file='data/final/multimodal_timeseries.parquet'
    )
    
    # Build unified graph
    G = mlg.build_unified_graph(include_transfers=True)
    
    # Get travel times at specific time
    travel_times = mlg.get_travel_times_at_time('2025-11-25 08:00:00')
    print(f"\nTravel times at 8:00 AM: {len(travel_times)} edges")
    
    # Update graph with time-dependent weights
    updated = mlg.update_edge_weights(G, travel_times)
    print(f"Updated {updated} edge weights")
