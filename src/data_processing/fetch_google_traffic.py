"""
Fetch real traffic data from Google Maps API for the Delhi road network.

This script:
1. Samples edges from the road network
2. Fetches real-time traffic data from Google Maps
3. Builds historical patterns based on current traffic conditions
4. Saves to parquet for use in routing
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.google_maps_provider import GoogleMapsTrafficProvider


class GoogleMapsTrafficFetcher:
    """
    Fetches real traffic data from Google Maps for road network edges.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize fetcher."""
        self.project_root = Path(__file__).parent.parent.parent
        self.provider = GoogleMapsTrafficProvider(api_key=api_key)
        
        # Load network
        self.nodes_df = pd.read_csv(
            self.project_root / 'data' / 'final' / 'nodes_final.csv',
            low_memory=False
        )
        self.edges_df = pd.read_csv(
            self.project_root / 'data' / 'final' / 'edges_final.csv'
        )
        
        # Filter to road layer
        self.road_nodes = self.nodes_df[self.nodes_df['layer'] == 0]
        self.road_edges = self.edges_df[self.edges_df['layer'] == 0]
        
        # Build node coordinate lookup
        self.node_coords = {}
        for _, node in self.road_nodes.iterrows():
            self.node_coords[node['node_id']] = (node['y'], node['x'])  # (lat, lon)
        
        print(f"Loaded {len(self.road_edges):,} road edges")
        
        # Output directory
        self.output_dir = self.project_root / 'data' / 'here'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def sample_edges(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Sample edges for traffic fetching.
        Prioritizes longer edges (major roads).
        """
        # Sort by length and sample from top
        sorted_edges = self.road_edges.sort_values('length_m', ascending=False)
        
        # Take top edges (major roads)
        sample = sorted_edges.head(n_samples).copy()
        
        print(f"Sampled {len(sample)} edges (avg length: {sample['length_m'].mean():.0f}m)")
        
        return sample
    
    def fetch_traffic_for_edges(self, edges: pd.DataFrame, 
                                 max_requests: int = 100) -> dict:
        """
        Fetch traffic data for sampled edges.
        
        Parameters:
        -----------
        edges : pd.DataFrame
            Edges to fetch traffic for
        max_requests : int
            Maximum API requests to make
            
        Returns:
        --------
        dict : {edge_id: traffic_data}
        """
        results = {}
        count = 0
        
        print(f"\nFetching traffic for {min(len(edges), max_requests)} edges...")
        
        for _, edge in edges.iterrows():
            if count >= max_requests:
                break
            
            edge_id = edge['edge_id']
            u, v = edge['u'], edge['v']
            
            if u not in self.node_coords or v not in self.node_coords:
                continue
            
            origin = self.node_coords[u]
            dest = self.node_coords[v]
            
            # Fetch from Google Maps
            result = self.provider.get_travel_time(origin, dest, mode='driving')
            
            if 'error' not in result:
                # Calculate speed
                distance_m = result['distance_m']
                duration_s = result.get('duration_traffic_s', result['duration_s'])
                
                if duration_s > 0:
                    speed_mps = distance_m / duration_s
                    speed_kmh = speed_mps * 3.6
                    
                    results[edge_id] = {
                        'google_distance_m': distance_m,
                        'google_duration_s': duration_s,
                        'google_speed_mps': speed_mps,
                        'google_speed_kmh': speed_kmh,
                        'congestion_ratio': result.get('congestion_ratio', 0),
                        'osm_length_m': edge['length_m'],
                        'osm_speed_mps': edge['speed_mps']
                    }
                    
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"  Fetched {count} edges...")
        
        print(f"Successfully fetched traffic for {len(results)} edges")
        
        return results
    
    def calculate_speed_adjustment(self, traffic_data: dict) -> float:
        """
        Calculate average speed adjustment factor from Google Maps data.
        
        Returns:
        --------
        float : Adjustment factor (Google speed / OSM speed)
        """
        if not traffic_data:
            return 1.0
        
        ratios = []
        for edge_id, data in traffic_data.items():
            osm_speed = data['osm_speed_mps']
            google_speed = data['google_speed_mps']
            
            if osm_speed > 0:
                ratios.append(google_speed / osm_speed)
        
        if not ratios:
            return 1.0
        
        avg_ratio = np.mean(ratios)
        print(f"\nSpeed adjustment factor: {avg_ratio:.2f}")
        print(f"  Google avg: {np.mean([d['google_speed_kmh'] for d in traffic_data.values()]):.1f} km/h")
        print(f"  OSM avg: {np.mean([d['osm_speed_mps'] * 3.6 for d in traffic_data.values()]):.1f} km/h")
        
        return avg_ratio
    
    def update_patterns_with_google(self, traffic_data: dict, 
                                     patterns_file: str = None) -> pd.DataFrame:
        """
        Update historical patterns using Google Maps speed adjustment.
        
        Parameters:
        -----------
        traffic_data : dict
            Traffic data from Google Maps
        patterns_file : str, optional
            Path to existing patterns file
            
        Returns:
        --------
        pd.DataFrame : Updated patterns
        """
        if patterns_file is None:
            patterns_file = self.output_dir / 'historical_patterns.parquet'
        
        if not Path(patterns_file).exists():
            print("No existing patterns file. Run fetch_here_historical.py first.")
            return None
        
        print(f"\nUpdating patterns with Google Maps data...")
        
        # Load existing patterns
        df = pd.read_parquet(patterns_file)
        print(f"  Loaded {len(df):,} pattern records")
        
        # Calculate adjustment factor
        adjustment = self.calculate_speed_adjustment(traffic_data)
        
        # Apply adjustment to all speeds
        df['speed_mps'] = df['speed_mps'] * adjustment
        df['speed_kmh'] = df['speed_kmh'] * adjustment
        
        # Recalculate travel times
        # We need edge lengths - merge with edges
        edge_lengths = self.road_edges[['edge_id', 'length_m']].copy()
        df = df.merge(edge_lengths, on='edge_id', how='left')
        
        # Calculate new travel times
        df['travel_time_s'] = df['length_m'] / df['speed_mps'].clip(lower=1.0)
        
        # Drop length column
        df = df.drop(columns=['length_m'])
        
        # Save updated patterns
        output_file = self.output_dir / 'historical_patterns_google.parquet'
        df.to_parquet(output_file, index=False)
        print(f"  Saved to {output_file}")
        
        # Also update the main file
        df.to_parquet(patterns_file, index=False)
        print(f"  Updated {patterns_file}")
        
        return df
    
    def save_traffic_sample(self, traffic_data: dict):
        """Save raw Google Maps traffic sample."""
        output_file = self.output_dir / 'google_traffic_sample.json'
        
        with open(output_file, 'w') as f:
            json.dump(traffic_data, f, indent=2)
        
        print(f"\nSaved traffic sample to {output_file}")
    
    def run(self, n_samples: int = 50, max_requests: int = 50):
        """
        Run the complete traffic fetching pipeline.
        
        Parameters:
        -----------
        n_samples : int
            Number of edges to sample
        max_requests : int
            Maximum API requests
        """
        print("=" * 60)
        print("Google Maps Traffic Fetcher")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Sample edges
        sample_edges = self.sample_edges(n_samples)
        
        # Fetch traffic
        traffic_data = self.fetch_traffic_for_edges(sample_edges, max_requests)
        
        # Save raw sample
        self.save_traffic_sample(traffic_data)
        
        # Update patterns
        self.update_patterns_with_google(traffic_data)
        
        print("\n" + "=" * 60)
        print("✅ Traffic data updated with Google Maps!")
        print("=" * 60)


def main():
    """Main entry point."""
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        print("❌ GOOGLE_MAPS_API_KEY not set")
        print("Run: $env:GOOGLE_MAPS_API_KEY='your_key'")
        return
    
    fetcher = GoogleMapsTrafficFetcher(api_key=api_key)
    
    # Fetch traffic for 50 major road edges
    # This uses ~50 API requests ($0.25 at $5/1000)
    fetcher.run(n_samples=50, max_requests=50)


if __name__ == "__main__":
    main()
