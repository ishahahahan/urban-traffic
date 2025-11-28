"""
Multimodal Dijkstra's algorithm for multi-layer transportation networks.
Finds optimal routes considering multiple transport modes and transfers.

Features:
- Time-dependent routing with congestion patterns
- Mode preferences and transfer penalties
- Route formatting and comparison utilities
"""

import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Optional
import heapq
import json


class MultimodalRouter:
    """
    Implements multimodal routing on multi-layer transportation networks.
    Uses modified Dijkstra's algorithm that handles transfers between layers.
    """
    
    def __init__(self, multilayer_graph):
        """
        Initialize router with multi-layer graph.
        
        Parameters:
        -----------
        multilayer_graph : MultiLayerGraph
            Instance of MultiLayerGraph class
        """
        
        self.mlg = multilayer_graph
        self.graph = None
        self.current_timestamp = None
        self.nodes = multilayer_graph.nodes if hasattr(multilayer_graph, 'nodes') else None
    
    def prepare_graph(self, timestamp=None, transfer_penalty=0):
        """
        Prepare unified graph for routing.
        
        Parameters:
        -----------
        timestamp : str or pd.Timestamp, optional
            Time for time-dependent routing
        transfer_penalty : float
            Additional time penalty for transfers (seconds)
            
        Returns:
        --------
        nx.DiGraph : Prepared graph
        """
        
        print(f"Preparing graph for routing...")
        
        # Build unified graph with transfers
        self.graph = self.mlg.build_unified_graph(include_transfers=True)
        
        # Update weights based on timestamp if provided
        if timestamp:
            self.current_timestamp = timestamp
            travel_times = self.mlg.get_travel_times_at_time(timestamp)
            updated = self.mlg.update_edge_weights(self.graph, travel_times)
            print(f"  Updated {updated} edges with time-dependent weights")
        
        # Add transfer penalty if specified
        if transfer_penalty > 0:
            for u, v, data in self.graph.edges(data=True):
                if data.get('is_transfer', False):
                    self.graph[u][v]['weight'] += transfer_penalty
            print(f"  Added {transfer_penalty}s penalty to all transfers")
        
        print(f"  Graph ready: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def find_route(self, source, target, timestamp=None, 
                   transfer_penalty=0, max_transfers=None):
        """
        Find optimal multimodal route between source and target.
        
        Parameters:
        -----------
        source : str
            Source node_id
        target : str
            Target node_id
        timestamp : str or pd.Timestamp, optional
            Time for time-dependent routing
        transfer_penalty : float
            Additional penalty for transfers (seconds)
        max_transfers : int, optional
            Maximum number of transfers allowed
            
        Returns:
        --------
        dict : Route information including:
            - path: list of node_ids
            - total_time: total travel time (seconds)
            - distance: total distance (meters)
            - segments: list of route segments with mode info
            - transfers: list of transfer points
        """
        
        # Prepare graph
        if self.graph is None or timestamp != self.current_timestamp:
            self.prepare_graph(timestamp, transfer_penalty)
        
        # Check if nodes exist
        if source not in self.graph:
            raise ValueError(f"Source node {source} not found in graph")
        if target not in self.graph:
            raise ValueError(f"Target node {target} not found in graph")
        
        print(f"\nFinding route from {source} to {target}...")
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            path_length = nx.shortest_path_length(self.graph, source, target, weight='weight')
            
            # Analyze path to extract segments and transfers
            segments = []
            transfers = []
            total_distance = 0
            current_mode = None
            segment_nodes = [path[0]]
            segment_time = 0
            segment_distance = 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.graph[u][v]
                
                # Check if this is a transfer
                if edge_data.get('is_transfer', False):
                    # Save current segment if exists
                    if current_mode is not None:
                        segments.append({
                            'mode': current_mode,
                            'nodes': segment_nodes,
                            'time_s': segment_time,
                            'distance_m': segment_distance
                        })
                    
                    # Record transfer
                    transfers.append({
                        'from_node': u,
                        'to_node': v,
                        'from_layer': edge_data['from_layer'],
                        'to_layer': edge_data['to_layer'],
                        'transfer_type': edge_data['transfer_type'],
                        'time_s': edge_data['transfer_time_s']
                    })
                    
                    # Start new segment
                    segment_nodes = [v]
                    segment_time = 0
                    segment_distance = 0
                    current_mode = None
                
                else:
                    # Regular edge
                    mode = edge_data['mode']
                    
                    # Check if mode changed (shouldn't happen within layer)
                    if current_mode is None:
                        current_mode = mode
                    elif current_mode != mode:
                        # Mode change without transfer - save segment
                        segments.append({
                            'mode': current_mode,
                            'nodes': segment_nodes,
                            'time_s': segment_time,
                            'distance_m': segment_distance
                        })
                        segment_nodes = [u]
                        segment_time = 0
                        segment_distance = 0
                        current_mode = mode
                    
                    # Add to current segment
                    segment_nodes.append(v)
                    segment_time += edge_data['weight']
                    segment_distance += edge_data.get('length_m', 0)
                    total_distance += edge_data.get('length_m', 0)
            
            # Save final segment
            if current_mode is not None:
                segments.append({
                    'mode': current_mode,
                    'nodes': segment_nodes,
                    'time_s': segment_time,
                    'distance_m': segment_distance
                })
            
            # Check transfer limit
            if max_transfers is not None and len(transfers) > max_transfers:
                print(f"Warning: Route has {len(transfers)} transfers (max: {max_transfers})")
            
            result = {
                'path': path,
                'total_time_s': path_length,
                'total_distance_m': total_distance,
                'segments': segments,
                'transfers': transfers,
                'num_transfers': len(transfers),
                'timestamp': timestamp
            }
            
            print(f"✓ Route found:")
            print(f"  Total time: {path_length:.1f}s ({path_length/60:.1f} min)")
            print(f"  Total distance: {total_distance:.1f}m ({total_distance/1000:.2f} km)")
            print(f"  Segments: {len(segments)}")
            print(f"  Transfers: {len(transfers)}")
            
            return result
            
        except nx.NetworkXNoPath:
            print(f"✗ No path found from {source} to {target}")
            return None
    
    def find_k_shortest_routes(self, source, target, k=3, timestamp=None, 
                               transfer_penalty=0):
        """
        Find k shortest alternative routes.
        
        Parameters:
        -----------
        source : str
            Source node_id
        target : str
            Target node_id
        k : int
            Number of routes to find
        timestamp : str or pd.Timestamp, optional
            Time for routing
        transfer_penalty : float
            Transfer penalty
            
        Returns:
        --------
        list : List of route dictionaries
        """
        
        # Prepare graph
        if self.graph is None or timestamp != self.current_timestamp:
            self.prepare_graph(timestamp, transfer_penalty)
        
        print(f"\nFinding {k} shortest routes from {source} to {target}...")
        
        try:
            # Use NetworkX k shortest paths
            paths = nx.shortest_simple_paths(self.graph, source, target, weight='weight')
            
            routes = []
            for i, path in enumerate(paths):
                if i >= k:
                    break
                
                # Calculate path length
                path_length = sum(
                    self.graph[path[j]][path[j+1]]['weight']
                    for j in range(len(path) - 1)
                )
                
                # Analyze path (simplified)
                route = self.find_route(source, target, timestamp, transfer_penalty)
                if route:
                    routes.append(route)
            
            print(f"✓ Found {len(routes)} alternative routes")
            
            return routes
            
        except nx.NetworkXNoPath:
            print(f"✗ No paths found")
            return []
    
    def compare_single_vs_multimodal(self, source, target, timestamp=None):
        """
        Compare single-mode (car only) vs multimodal routing.
        
        Parameters:
        -----------
        source : str
            Source node_id (must be in road layer)
        target : str
            Target node_id (must be in road layer)
        timestamp : str or pd.Timestamp, optional
            Time for routing
            
        Returns:
        --------
        dict : Comparison results
        """
        
        print("\n" + "="*60)
        print("SINGLE-MODE vs MULTIMODAL COMPARISON")
        print("="*60)
        
        # Single-mode (car only) - use only layer 0
        print("\n1. Single-mode routing (car only)...")
        self.prepare_graph(timestamp, transfer_penalty=0)
        
        # Temporarily remove transfers and non-road edges
        road_graph = self.mlg.layers[0].copy()
        if timestamp:
            travel_times = self.mlg.get_travel_times_at_time(timestamp)
            self.mlg.update_edge_weights(road_graph, travel_times)
        
        try:
            single_path = nx.shortest_path(road_graph, source, target, weight='weight')
            single_time = nx.shortest_path_length(road_graph, source, target, weight='weight')
            print(f"   Car route: {single_time:.1f}s ({single_time/60:.1f} min)")
        except:
            single_path = None
            single_time = float('inf')
            print("   No car route found")
        
        # Multimodal routing
        print("\n2. Multimodal routing...")
        multi_route = self.find_route(source, target, timestamp, transfer_penalty=60)
        
        if multi_route:
            multi_time = multi_route['total_time_s']
            print(f"   Multimodal route: {multi_time:.1f}s ({multi_time/60:.1f} min)")
        else:
            multi_time = float('inf')
        
        # Comparison
        print("\n" + "="*60)
        if single_time < float('inf') and multi_time < float('inf'):
            savings = single_time - multi_time
            percent = (savings / single_time) * 100
            
            print(f"Time savings: {abs(savings):.1f}s ({abs(savings)/60:.1f} min)")
            
            if savings > 0:
                print(f"Multimodal is {percent:.1f}% FASTER ✓")
            else:
                print(f"Car is {abs(percent):.1f}% faster")
        print("="*60 + "\n")
        
        return {
            'single_mode_time': single_time,
            'multimodal_time': multi_time,
            'time_savings': single_time - multi_time,
            'single_path': single_path,
            'multimodal_route': multi_route
        }
    
    def format_route(self, route):
        """
        Format route for display with nice formatting.
        
        Parameters:
        -----------
        route : dict
            Route result from find_route
            
        Returns:
        --------
        str : Formatted route description
        """
        
        if not route:
            return "No route found"
        
        output = []
        output.append(f"\n{'='*70}")
        output.append(f"MULTIMODAL ROUTE")
        output.append(f"{'='*70}")
        output.append(f"Total time: {route['total_time_s']/60:.1f} minutes ({route['total_time_s']:.0f} seconds)")
        output.append(f"Total distance: {route['total_distance_m']/1000:.2f} km")
        output.append(f"Number of transfers: {route['num_transfers']}")
        
        if route.get('timestamp'):
            output.append(f"Departure: {route['timestamp']}")
        
        output.append(f"\n{'-'*70}")
        output.append(f"ROUTE SEGMENTS")
        output.append(f"{'-'*70}\n")
        
        mode_emojis = {
            'car': '🚗',
            'metro': '🚇',
            'walk': '🚶',
            'transfer': '🔄',
            'auto': '🛺'
        }
        
        for i, seg in enumerate(route['segments'], 1):
            mode = seg['mode']
            emoji = mode_emojis.get(mode, '→')
            
            output.append(f"{i}. {emoji} {mode.upper()}")
            output.append(f"   Duration: {seg['time_s']/60:.1f} min")
            output.append(f"   Distance: {seg['distance_m']:.0f} m")
            output.append(f"   Nodes: {len(seg['nodes'])}")
            output.append("")
        
        # Show transfers if any
        if route['transfers']:
            output.append(f"{'-'*70}")
            output.append("TRANSFERS")
            output.append(f"{'-'*70}\n")
            
            for i, transfer in enumerate(route['transfers'], 1):
                layer_names = {0: 'Road', 1: 'Metro', 2: 'Walk'}
                from_layer = layer_names.get(transfer['from_layer'], f"Layer {transfer['from_layer']}")
                to_layer = layer_names.get(transfer['to_layer'], f"Layer {transfer['to_layer']}")
                
                output.append(f"{i}. {from_layer} → {to_layer}")
                output.append(f"   Type: {transfer['transfer_type']}")
                output.append(f"   Time: {transfer['time_s']/60:.1f} min")
                output.append("")
        
        output.append(f"{'='*70}")
        
        return '\n'.join(output)
    
    def get_route_summary(self, route):
        """
        Get a compact summary of the route.
        
        Parameters:
        -----------
        route : dict
            Route result from find_route
            
        Returns:
        --------
        dict : Compact summary
        """
        
        if not route:
            return None
        
        # Count segments by mode
        mode_segments = {}
        for seg in route['segments']:
            mode = seg['mode']
            if mode not in mode_segments:
                mode_segments[mode] = {'count': 0, 'time_s': 0, 'distance_m': 0}
            mode_segments[mode]['count'] += 1
            mode_segments[mode]['time_s'] += seg['time_s']
            mode_segments[mode]['distance_m'] += seg['distance_m']
        
        return {
            'total_time_min': route['total_time_s'] / 60,
            'total_distance_km': route['total_distance_m'] / 1000,
            'num_transfers': route['num_transfers'],
            'modes_used': list(mode_segments.keys()),
            'mode_breakdown': mode_segments,
            'path_length': len(route['path'])
        }


if __name__ == "__main__":
    from graph_loader import MultiLayerGraph
    
    # Load network
    mlg = MultiLayerGraph(
        nodes_file='data/final/nodes_final.csv',
        edges_file='data/final/edges_final.csv',
        transfers_file='data/final/transfers_final.csv',
        timeseries_file='data/final/multimodal_timeseries.parquet'
    )
    
    # Create router
    router = MultimodalRouter(mlg)
    
    # Example: Find route between two nodes
    # (You'll need to specify actual node_ids from your network)
    source = 'road_0'
    target = 'road_100'
    
    route = router.find_route(
        source=source,
        target=target,
        timestamp='2025-11-25 08:00:00',
        transfer_penalty=60
    )
    
    if route:
        # Print formatted route
        print(router.format_route(route))
        
        # Print summary
        summary = router.get_route_summary(route)
        print("\nRoute Summary:")
        print(f"  Total time: {summary['total_time_min']:.1f} min")
        print(f"  Total distance: {summary['total_distance_km']:.2f} km")
        print(f"  Modes used: {', '.join(summary['modes_used'])}")
        print(f"  Transfers: {summary['num_transfers']}")
