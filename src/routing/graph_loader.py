"""
Load and manage multi-layer transportation network graph.
Provides unified interface for accessing nodes, edges, and transfers.
Supports both synthetic timeseries and HERE Traffic API historical patterns.
"""

import pandas as pd
import networkx as nx
import os
from pathlib import Path


class MultiLayerGraph:
    """
    Manages multi-layer transportation network.
    Each layer is a separate NetworkX graph connected via transfer edges.
    Supports HERE Traffic API historical patterns for time-dependent routing.
    """
    
    def __init__(self, nodes_file, edges_file, transfers_file, 
                 timeseries_file=None, here_patterns_file=None):
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
            Path to multimodal_timeseries.parquet (synthetic data)
        here_patterns_file : str, optional
            Path to HERE historical_patterns.parquet
        """
        
        print("Loading multi-layer network...")
        
        # Load data
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.transfers_df = pd.read_csv(transfers_file)
        
        print(f"  Nodes: {len(self.nodes_df):,}")
        print(f"  Edges: {len(self.edges_df):,}")
        print(f"  Transfers: {len(self.transfers_df):,}")
        
        # Load synthetic timeseries if provided
        self.timeseries_df = None
        if timeseries_file and os.path.exists(timeseries_file):
            self.timeseries_df = pd.read_parquet(timeseries_file)
            print(f"  Synthetic time-series records: {len(self.timeseries_df):,}")
        
        # Load HERE historical patterns if provided
        self.here_patterns_df = None
        self.traffic_source = 'static'  # 'static', 'synthetic', or 'here'
        
        if here_patterns_file and os.path.exists(here_patterns_file):
            self.here_patterns_df = pd.read_parquet(here_patterns_file)
            self.traffic_source = 'here'
            print(f"  HERE historical patterns: {len(self.here_patterns_df):,}")
            
            # Create lookup index for fast access
            self._build_here_index()
        elif self.timeseries_df is not None:
            self.traffic_source = 'synthetic'
        
        print(f"  Traffic data source: {self.traffic_source}")
        
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
    
    def _build_here_index(self):
        """
        Build lookup index for HERE patterns for fast access.
        Creates dictionary: (edge_id, day, hour) -> travel_time_s
        """
        print("  Building HERE patterns index...")
        
        self.here_index = {}
        
        for _, row in self.here_patterns_df.iterrows():
            key = (row['edge_id'], row['day_of_week'], row['hour'])
            self.here_index[key] = {
                'travel_time_s': row['travel_time_s'],
                'speed_mps': row['speed_mps'],
                'jam_factor': row['jam_factor'],
                'congestion_ratio': row.get('congestion_ratio', 0)
            }
        
        print(f"  Indexed {len(self.here_index):,} HERE pattern entries")
    
    def get_travel_times_for_datetime(self, dt):
        """
        Get travel times for all edges at a specific datetime using HERE patterns.
        
        Parameters:
        -----------
        dt : datetime or pd.Timestamp
            Datetime to query
            
        Returns:
        --------
        dict : {edge_id: travel_time_s}
        """
        if self.traffic_source == 'here' and self.here_patterns_df is not None:
            return self._get_here_travel_times(dt)
        elif self.traffic_source == 'synthetic' and self.timeseries_df is not None:
            return self.get_travel_times_at_time(dt)
        else:
            # Return static travel times from edges
            return dict(zip(self.edges_df['edge_id'], self.edges_df['travel_time_s']))
    
    def _get_here_travel_times(self, dt):
        """
        Get travel times from HERE historical patterns.
        
        Parameters:
        -----------
        dt : datetime or pd.Timestamp
            Datetime to query
            
        Returns:
        --------
        dict : {edge_id: travel_time_s}
        """
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        
        # Get day of week and hour
        day_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
        # Handle both datetime.datetime and pd.Timestamp
        weekday = dt.weekday() if hasattr(dt, 'weekday') else dt.dayofweek
        day = day_map[weekday]
        hour = dt.hour
        
        travel_times = {}
        
        # Use index for fast lookup
        if hasattr(self, 'here_index'):
            for edge_id in self.edges_df['edge_id'].unique():
                key = (edge_id, day, hour)
                if key in self.here_index:
                    travel_times[edge_id] = self.here_index[key]['travel_time_s']
        else:
            # Fallback to DataFrame filtering (slower)
            mask = (self.here_patterns_df['day_of_week'] == day) & \
                   (self.here_patterns_df['hour'] == hour)
            snapshot = self.here_patterns_df[mask]
            travel_times = dict(zip(snapshot['edge_id'], snapshot['travel_time_s']))
        
        return travel_times
    
    def get_traffic_info_at_time(self, dt, edge_id):
        """
        Get detailed traffic info for a specific edge at a specific time.
        
        Parameters:
        -----------
        dt : datetime or pd.Timestamp
            Datetime to query
        edge_id : str
            Edge ID
            
        Returns:
        --------
        dict : Traffic info including speed, jam_factor, travel_time
        """
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        
        day_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
        weekday = dt.weekday() if hasattr(dt, 'weekday') else dt.dayofweek
        day = day_map[weekday]
        hour = dt.hour
        
        # Default info from edges
        edge_row = self.edges_df[self.edges_df['edge_id'] == edge_id]
        if len(edge_row) == 0:
            return None
        
        default_info = {
            'edge_id': edge_id,
            'travel_time_s': edge_row.iloc[0]['travel_time_s'],
            'speed_mps': edge_row.iloc[0]['speed_mps'],
            'jam_factor': 0,
            'congestion_ratio': 0,
            'source': 'static'
        }
        
        # Try to get HERE data
        if self.traffic_source == 'here' and hasattr(self, 'here_index'):
            key = (edge_id, day, hour)
            if key in self.here_index:
                info = self.here_index[key].copy()
                info['edge_id'] = edge_id
                info['source'] = 'here'
                return info
        
        return default_info
    
    def get_congestion_summary(self, dt):
        """
        Get network-wide congestion summary for a datetime.
        
        Parameters:
        -----------
        dt : datetime or pd.Timestamp
            Datetime to query
            
        Returns:
        --------
        dict : Congestion summary statistics
        """
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        
        day_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
        weekday = dt.weekday() if hasattr(dt, 'weekday') else dt.dayofweek
        day = day_map[weekday]
        hour = dt.hour
        
        if self.traffic_source != 'here' or self.here_patterns_df is None:
            return {'source': 'static', 'message': 'No historical data available'}
        
        # Filter patterns for this time
        mask = (self.here_patterns_df['day_of_week'] == day) & \
               (self.here_patterns_df['hour'] == hour)
        snapshot = self.here_patterns_df[mask]
        
        if len(snapshot) == 0:
            return {'source': 'here', 'message': 'No data for this time slot'}
        
        return {
            'source': 'here',
            'day': day,
            'hour': hour,
            'edges_count': len(snapshot),
            'avg_speed_kmh': snapshot['speed_kmh'].mean(),
            'avg_jam_factor': snapshot['jam_factor'].mean(),
            'avg_congestion_pct': snapshot['congestion_ratio'].mean() * 100,
            'max_jam_factor': snapshot['jam_factor'].max(),
            'highly_congested_edges': len(snapshot[snapshot['jam_factor'] > 5]),
            'free_flow_edges': len(snapshot[snapshot['jam_factor'] < 1])
        }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent.parent
    
    # Check for HERE patterns
    here_patterns = base_path / 'data' / 'here' / 'historical_patterns.parquet'
    
    mlg = MultiLayerGraph(
        nodes_file=str(base_path / 'data' / 'final' / 'nodes_final.csv'),
        edges_file=str(base_path / 'data' / 'final' / 'edges_final.csv'),
        transfers_file=str(base_path / 'data' / 'final' / 'transfers_final.csv'),
        timeseries_file=str(base_path / 'data' / 'final' / 'multimodal_timeseries.parquet'),
        here_patterns_file=str(here_patterns) if here_patterns.exists() else None
    )
    
    # Build unified graph
    G = mlg.build_unified_graph(include_transfers=True)
    
    # Test time-dependent travel times
    from datetime import datetime
    
    # Monday 8:30 AM (morning rush)
    test_time = datetime(2025, 12, 1, 8, 30)  # Monday
    travel_times = mlg.get_travel_times_for_datetime(test_time)
    print(f"\nTravel times at Monday 8:30 AM: {len(travel_times)} edges")
    
    # Get congestion summary
    summary = mlg.get_congestion_summary(test_time)
    print(f"Congestion summary: {summary}")
    
    # Update graph with time-dependent weights
    updated = mlg.update_edge_weights(G, travel_times)
    print(f"Updated {updated} edge weights")
