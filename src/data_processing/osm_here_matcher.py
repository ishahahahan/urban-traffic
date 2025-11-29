"""
OSM to HERE Road Matcher

This module matches OSM road edges to HERE Traffic API road segments
using spatial proximity and geometry matching.

The matching is done by:
1. Building a spatial index (R-tree) of HERE road segments
2. For each OSM edge, finding nearby HERE segments
3. Selecting the best match based on distance and direction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import math

# Try to import rtree for spatial indexing (optional but recommended)
try:
    from rtree import index as rtree_index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
    print("Warning: rtree not installed. Using slower matching algorithm.")
    print("Install with: pip install rtree")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def bearing_difference(bearing1: float, bearing2: float) -> float:
    """
    Calculate minimum angular difference between two bearings.
    """
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)


class OSMHEREMatcher:
    """
    Matches OSM road edges to HERE traffic segments.
    """
    
    def __init__(self, nodes_file: str, edges_file: str):
        """
        Initialize matcher with OSM network data.
        
        Parameters:
        -----------
        nodes_file : str
            Path to nodes_final.csv
        edges_file : str
            Path to edges_final.csv
        """
        print("Loading OSM network data...")
        
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        
        # Filter to road layer only (layer 0)
        self.road_nodes = self.nodes_df[self.nodes_df['layer'] == 0].copy()
        self.road_edges = self.edges_df[self.edges_df['layer'] == 0].copy()
        
        print(f"  Road nodes: {len(self.road_nodes):,}")
        print(f"  Road edges: {len(self.road_edges):,}")
        
        # Build node lookup for coordinates
        self.node_coords = {}
        for _, node in self.road_nodes.iterrows():
            self.node_coords[node['node_id']] = (node['y'], node['x'])  # (lat, lon)
        
        # HERE segments storage
        self.here_segments = []
        self.spatial_index = None
        
        # Matching results
        self.matches = {}  # edge_id -> here_segment_index
        
    def load_here_segments(self, segments: List[dict]):
        """
        Load HERE traffic segments for matching.
        
        Parameters:
        -----------
        segments : list
            List of parsed HERE segments from HERETrafficClient.parse_flow_results()
        """
        self.here_segments = segments
        print(f"Loaded {len(segments)} HERE traffic segments")
        
        # Build spatial index if rtree available
        if HAS_RTREE:
            self._build_spatial_index()
        
    def _build_spatial_index(self):
        """Build R-tree spatial index for HERE segments."""
        print("Building spatial index...")
        
        self.spatial_index = rtree_index.Index()
        
        for i, seg in enumerate(self.here_segments):
            # Get bounding box of segment
            points = seg.get('points', [])
            if not points:
                # Use center point
                lat, lon = seg['center_lat'], seg['center_lng']
                bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
            else:
                lats = [p['lat'] for p in points]
                lons = [p['lng'] for p in points]
                bbox = (min(lons), min(lats), max(lons), max(lats))
            
            self.spatial_index.insert(i, bbox)
        
        print(f"  Indexed {len(self.here_segments)} segments")
    
    def _find_nearby_segments(self, lat: float, lon: float, 
                              radius_deg: float = 0.002) -> List[int]:
        """
        Find HERE segments near a point.
        
        Parameters:
        -----------
        lat, lon : float
            Query point coordinates
        radius_deg : float
            Search radius in degrees (~200m at 0.002)
            
        Returns:
        --------
        list : Indices of nearby segments
        """
        if HAS_RTREE and self.spatial_index:
            # Use spatial index
            bbox = (lon - radius_deg, lat - radius_deg, 
                    lon + radius_deg, lat + radius_deg)
            return list(self.spatial_index.intersection(bbox))
        else:
            # Brute force search
            nearby = []
            for i, seg in enumerate(self.here_segments):
                dist = haversine_distance(lat, lon, 
                                          seg['center_lat'], seg['center_lng'])
                if dist < radius_deg * 111000:  # Convert to meters
                    nearby.append(i)
            return nearby
    
    def match_edge(self, edge_id: str, u: str, v: str, 
                   max_distance: float = 100,
                   max_bearing_diff: float = 45) -> Optional[int]:
        """
        Match a single OSM edge to best HERE segment.
        
        Parameters:
        -----------
        edge_id : str
            Edge ID
        u, v : str
            Source and target node IDs
        max_distance : float
            Maximum distance in meters for match
        max_bearing_diff : float
            Maximum bearing difference in degrees
            
        Returns:
        --------
        int or None : Index of matched HERE segment, or None
        """
        # Get edge coordinates
        if u not in self.node_coords or v not in self.node_coords:
            return None
        
        lat1, lon1 = self.node_coords[u]
        lat2, lon2 = self.node_coords[v]
        
        # Calculate edge midpoint and bearing
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        edge_bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Find nearby HERE segments
        nearby = self._find_nearby_segments(mid_lat, mid_lon)
        
        if not nearby:
            return None
        
        # Score each candidate
        best_match = None
        best_score = float('inf')
        
        for seg_idx in nearby:
            seg = self.here_segments[seg_idx]
            
            # Calculate distance to segment center
            dist = haversine_distance(mid_lat, mid_lon,
                                      seg['center_lat'], seg['center_lng'])
            
            if dist > max_distance:
                continue
            
            # Calculate bearing of HERE segment
            if seg.get('start_lat') and seg.get('end_lat'):
                seg_bearing = calculate_bearing(
                    seg['start_lat'], seg['start_lng'],
                    seg['end_lat'], seg['end_lng']
                )
            else:
                seg_bearing = edge_bearing  # Assume same direction
            
            # Check bearing difference (allow opposite direction for bidirectional roads)
            bearing_diff = bearing_difference(edge_bearing, seg_bearing)
            if bearing_diff > max_bearing_diff and bearing_diff < (180 - max_bearing_diff):
                continue
            
            # Calculate score (lower is better)
            # Weight distance more heavily than bearing
            score = dist + bearing_diff * 2
            
            if score < best_score:
                best_score = score
                best_match = seg_idx
        
        return best_match
    
    def match_all_edges(self, max_distance: float = 100, 
                        max_bearing_diff: float = 45,
                        verbose: bool = True) -> Dict[str, int]:
        """
        Match all OSM road edges to HERE segments.
        
        Parameters:
        -----------
        max_distance : float
            Maximum distance in meters
        max_bearing_diff : float
            Maximum bearing difference in degrees
        verbose : bool
            Print progress
            
        Returns:
        --------
        dict : {edge_id: here_segment_index}
        """
        print(f"\nMatching {len(self.road_edges):,} OSM edges to HERE segments...")
        
        matched = 0
        unmatched = 0
        
        for idx, edge in self.road_edges.iterrows():
            edge_id = edge['edge_id']
            u = edge['u']
            v = edge['v']
            
            match = self.match_edge(edge_id, u, v, max_distance, max_bearing_diff)
            
            if match is not None:
                self.matches[edge_id] = match
                matched += 1
            else:
                unmatched += 1
            
            if verbose and (matched + unmatched) % 10000 == 0:
                total = matched + unmatched
                pct = matched / total * 100 if total > 0 else 0
                print(f"  Progress: {total:,}/{len(self.road_edges):,} "
                      f"({pct:.1f}% matched)")
        
        match_rate = matched / (matched + unmatched) * 100 if (matched + unmatched) > 0 else 0
        print(f"\nMatching complete:")
        print(f"  Matched: {matched:,} ({match_rate:.1f}%)")
        print(f"  Unmatched: {unmatched:,} ({100-match_rate:.1f}%)")
        
        return self.matches
    
    def save_mapping(self, output_file: str):
        """
        Save edge-to-HERE mapping to CSV.
        
        Parameters:
        -----------
        output_file : str
            Path to output CSV file
        """
        # Create mapping dataframe
        records = []
        
        for edge_id, seg_idx in self.matches.items():
            seg = self.here_segments[seg_idx]
            records.append({
                'edge_id': edge_id,
                'here_segment_idx': seg_idx,
                'here_center_lat': seg['center_lat'],
                'here_center_lng': seg['center_lng'],
                'here_free_flow_kmh': seg.get('free_flow_kmh', 0),
                'here_length_m': seg.get('length_m', 0)
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"Saved mapping to {output_file}")
        print(f"  {len(df):,} edges mapped")
        
        return df
    
    def get_traffic_for_edges(self, segments: List[dict] = None) -> Dict[str, dict]:
        """
        Get traffic data for all matched edges.
        
        Parameters:
        -----------
        segments : list, optional
            Updated HERE segments. Uses stored segments if not provided.
            
        Returns:
        --------
        dict : {edge_id: traffic_data}
        """
        if segments:
            self.here_segments = segments
        
        traffic_data = {}
        
        for edge_id, seg_idx in self.matches.items():
            if seg_idx < len(self.here_segments):
                seg = self.here_segments[seg_idx]
                traffic_data[edge_id] = {
                    'speed_mps': seg.get('speed_mps', 0),
                    'speed_kmh': seg.get('speed_kmh', 0),
                    'free_flow_mps': seg.get('free_flow_mps', 0),
                    'free_flow_kmh': seg.get('free_flow_kmh', 0),
                    'jam_factor': seg.get('jam_factor', 0),
                    'congestion_ratio': seg.get('congestion_ratio', 0)
                }
        
        return traffic_data
    
    def calculate_travel_times(self, segments: List[dict] = None) -> Dict[str, float]:
        """
        Calculate travel times for all matched edges based on HERE traffic.
        
        Parameters:
        -----------
        segments : list, optional
            Updated HERE segments with current traffic.
            
        Returns:
        --------
        dict : {edge_id: travel_time_seconds}
        """
        if segments:
            self.here_segments = segments
        
        travel_times = {}
        
        for edge_id, seg_idx in self.matches.items():
            if seg_idx >= len(self.here_segments):
                continue
            
            seg = self.here_segments[seg_idx]
            
            # Get edge length from original data
            edge_row = self.road_edges[self.road_edges['edge_id'] == edge_id]
            if len(edge_row) == 0:
                continue
            
            length_m = edge_row.iloc[0]['length_m']
            
            # Get speed from HERE (with fallback)
            speed_mps = seg.get('speed_mps', 0)
            
            if speed_mps <= 0:
                # Fallback to free flow or original speed
                speed_mps = seg.get('free_flow_mps', edge_row.iloc[0]['speed_mps'])
            
            if speed_mps > 0:
                travel_times[edge_id] = length_m / speed_mps
            else:
                # Use original travel time as fallback
                travel_times[edge_id] = edge_row.iloc[0]['travel_time_s']
        
        return travel_times


def test_matcher():
    """Test the OSM-HERE matcher."""
    print("=" * 60)
    print("Testing OSM-HERE Matcher")
    print("=" * 60)
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    nodes_file = base_path / 'data' / 'final' / 'nodes_final.csv'
    edges_file = base_path / 'data' / 'final' / 'edges_final.csv'
    
    # Check files exist
    if not nodes_file.exists() or not edges_file.exists():
        print(f"Error: Network files not found")
        print(f"  Expected: {nodes_file}")
        print(f"  Expected: {edges_file}")
        return False
    
    # Initialize matcher
    matcher = OSMHEREMatcher(str(nodes_file), str(edges_file))
    
    # Create dummy HERE segments for testing
    # In real usage, these come from HERETrafficClient
    dummy_segments = []
    for i in range(100):
        lat = 28.5 + np.random.random() * 0.3
        lon = 77.0 + np.random.random() * 0.4
        dummy_segments.append({
            'center_lat': lat,
            'center_lng': lon,
            'start_lat': lat - 0.001,
            'start_lng': lon - 0.001,
            'end_lat': lat + 0.001,
            'end_lng': lon + 0.001,
            'points': [
                {'lat': lat - 0.001, 'lng': lon - 0.001},
                {'lat': lat + 0.001, 'lng': lon + 0.001}
            ],
            'speed_kmh': 30 + np.random.random() * 30,
            'free_flow_kmh': 50 + np.random.random() * 20,
            'jam_factor': np.random.random() * 5,
            'speed_mps': (30 + np.random.random() * 30) / 3.6,
            'free_flow_mps': (50 + np.random.random() * 20) / 3.6,
            'congestion_ratio': np.random.random() * 0.5
        })
    
    # Load segments
    matcher.load_here_segments(dummy_segments)
    
    # Match a sample of edges
    print("\nMatching sample edges...")
    sample_edges = matcher.road_edges.head(100)
    
    matched = 0
    for _, edge in sample_edges.iterrows():
        match = matcher.match_edge(edge['edge_id'], edge['u'], edge['v'])
        if match is not None:
            matched += 1
    
    print(f"  Matched {matched}/100 sample edges")
    
    print("\n" + "=" * 60)
    print("Matcher test completed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_matcher()
