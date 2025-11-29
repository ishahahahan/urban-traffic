"""
Fetch and cache historical traffic patterns from HERE API.

This script:
1. Fetches real-time traffic from HERE API
2. Uses time-based adjustment factors for Delhi traffic patterns
3. Builds a complete historical pattern dataset
4. Matches HERE data to OSM edges
5. Creates parquet file for fast loading

Run this script once to build the historical traffic dataset.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.here_traffic_provider import HERETrafficClient
from src.data_processing.osm_here_matcher import OSMHEREMatcher


class HistoricalTrafficBuilder:
    """
    Builds historical traffic dataset from HERE API and OSM network.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the builder.
        
        Parameters:
        -----------
        api_key : str, optional
            HERE API key. Reads from environment if not provided.
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize HERE client
        self.here_client = HERETrafficClient(api_key=api_key)
        
        # Initialize OSM-HERE matcher
        nodes_file = self.project_root / 'data' / 'final' / 'nodes_final.csv'
        edges_file = self.project_root / 'data' / 'final' / 'edges_final.csv'
        
        self.matcher = OSMHEREMatcher(str(nodes_file), str(edges_file))
        
        # Output paths
        self.output_dir = self.project_root / 'data' / 'here'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_base_traffic(self) -> list:
        """
        Fetch real-time traffic as base for historical patterns.
        
        Returns:
        --------
        list : Parsed HERE traffic segments
        """
        print("\n" + "=" * 60)
        print("Fetching base traffic data from HERE API")
        print("=" * 60)
        
        # Fetch real-time flow
        flow_data = self.here_client.get_traffic_flow(use_cache=False)
        
        if 'error' in flow_data:
            raise RuntimeError(f"HERE API error: {flow_data['error']}")
        
        # Parse results
        segments = self.here_client.parse_flow_results(flow_data)
        
        print(f"Received {len(segments)} road segments from HERE")
        
        # Save raw segments
        segments_file = self.output_dir / 'here_segments_raw.json'
        with open(segments_file, 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"Saved raw segments to {segments_file}")
        
        return segments
    
    def match_segments_to_edges(self, segments: list) -> dict:
        """
        Match HERE segments to OSM edges.
        
        Parameters:
        -----------
        segments : list
            HERE traffic segments
            
        Returns:
        --------
        dict : Edge to segment mapping
        """
        print("\n" + "=" * 60)
        print("Matching HERE segments to OSM edges")
        print("=" * 60)
        
        # Load segments into matcher
        self.matcher.load_here_segments(segments)
        
        # Match all edges
        matches = self.matcher.match_all_edges(
            max_distance=150,  # 150 meters
            max_bearing_diff=45  # 45 degrees
        )
        
        # Save mapping
        mapping_file = self.output_dir / 'edge_here_mapping.csv'
        self.matcher.save_mapping(str(mapping_file))
        
        return matches
    
    def build_historical_patterns(self, segments: list) -> pd.DataFrame:
        """
        Build historical traffic patterns for all time slots.
        
        Parameters:
        -----------
        segments : list
            Base HERE traffic segments
            
        Returns:
        --------
        pd.DataFrame : Historical patterns for all edges and time slots
        """
        print("\n" + "=" * 60)
        print("Building historical traffic patterns")
        print("=" * 60)
        
        days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        hours = list(range(24))
        
        # Get all matched edges
        edge_ids = list(self.matcher.matches.keys())
        print(f"Building patterns for {len(edge_ids):,} matched edges")
        print(f"Time slots: {len(days)} days × {len(hours)} hours = {len(days) * len(hours)} slots")
        
        records = []
        total_slots = len(days) * len(hours)
        slot_count = 0
        
        for day in days:
            for hour in hours:
                slot_count += 1
                
                # Get adjustment factor for this time
                adjustment = self.here_client._get_time_adjustment(day, hour)
                
                # Adjust traffic for each matched edge
                for edge_id, seg_idx in self.matcher.matches.items():
                    if seg_idx >= len(segments):
                        continue
                    
                    seg = segments[seg_idx]
                    
                    # Get base values
                    base_jam = seg.get('jam_factor', 0)
                    free_flow_mps = seg.get('free_flow_mps', 10)  # Default ~36 km/h
                    
                    # Apply time-based adjustment
                    adjusted_jam = min(10, base_jam * adjustment)
                    
                    # Calculate adjusted speed
                    # Jam factor 0 = free flow, 10 = standstill
                    speed_factor = 1 - (adjusted_jam * 0.08)  # 0-80% reduction
                    speed_mps = max(1.0, free_flow_mps * speed_factor)  # Min 1 m/s
                    
                    # Get edge length
                    edge_row = self.matcher.road_edges[
                        self.matcher.road_edges['edge_id'] == edge_id
                    ]
                    if len(edge_row) == 0:
                        continue
                    
                    length_m = edge_row.iloc[0]['length_m']
                    travel_time_s = length_m / speed_mps
                    
                    records.append({
                        'edge_id': edge_id,
                        'day_of_week': day,
                        'hour': hour,
                        'speed_mps': speed_mps,
                        'speed_kmh': speed_mps * 3.6,
                        'jam_factor': adjusted_jam,
                        'travel_time_s': travel_time_s,
                        'free_flow_mps': free_flow_mps,
                        'congestion_ratio': 1 - speed_factor
                    })
                
                if slot_count % 24 == 0:
                    pct = slot_count / total_slots * 100
                    print(f"  Progress: {slot_count}/{total_slots} time slots ({pct:.0f}%)")
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        print(f"\nCreated {len(df):,} historical pattern records")
        print(f"  Unique edges: {df['edge_id'].nunique():,}")
        print(f"  Time slots: {len(days) * len(hours)}")
        
        return df
    
    def add_unmatched_edges(self, patterns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add static patterns for unmatched edges using original OSM speeds.
        
        Parameters:
        -----------
        patterns_df : pd.DataFrame
            Patterns for matched edges
            
        Returns:
        --------
        pd.DataFrame : Complete patterns including unmatched edges
        """
        print("\n" + "=" * 60)
        print("Adding patterns for unmatched edges")
        print("=" * 60)
        
        # Get unmatched edges
        matched_edges = set(self.matcher.matches.keys())
        all_road_edges = set(self.matcher.road_edges['edge_id'].unique())
        unmatched_edges = all_road_edges - matched_edges
        
        print(f"Unmatched edges: {len(unmatched_edges):,}")
        
        if len(unmatched_edges) == 0:
            return patterns_df
        
        # Build records for unmatched edges using original OSM speeds
        days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        hours = list(range(24))
        
        records = []
        
        for edge_id in unmatched_edges:
            edge_row = self.matcher.road_edges[
                self.matcher.road_edges['edge_id'] == edge_id
            ]
            if len(edge_row) == 0:
                continue
            
            edge = edge_row.iloc[0]
            base_speed_mps = edge['speed_mps']
            length_m = edge['length_m']
            
            for day in days:
                for hour in hours:
                    # Apply time-based adjustment
                    adjustment = self.here_client._get_time_adjustment(day, hour)
                    
                    # Adjust speed (less variation for non-HERE roads)
                    speed_factor = 1 - (adjustment - 1) * 0.2  # Smaller variation
                    speed_mps = max(1.0, base_speed_mps * speed_factor)
                    travel_time_s = length_m / speed_mps
                    
                    records.append({
                        'edge_id': edge_id,
                        'day_of_week': day,
                        'hour': hour,
                        'speed_mps': speed_mps,
                        'speed_kmh': speed_mps * 3.6,
                        'jam_factor': (1 - speed_factor) * 10,  # Estimated
                        'travel_time_s': travel_time_s,
                        'free_flow_mps': base_speed_mps,
                        'congestion_ratio': 1 - speed_factor
                    })
        
        # Combine with matched patterns
        unmatched_df = pd.DataFrame(records)
        combined_df = pd.concat([patterns_df, unmatched_df], ignore_index=True)
        
        print(f"Added {len(records):,} records for unmatched edges")
        print(f"Total records: {len(combined_df):,}")
        
        return combined_df
    
    def save_patterns(self, patterns_df: pd.DataFrame):
        """
        Save historical patterns to parquet file.
        
        Parameters:
        -----------
        patterns_df : pd.DataFrame
            Complete historical patterns
        """
        print("\n" + "=" * 60)
        print("Saving historical patterns")
        print("=" * 60)
        
        # Save to parquet (efficient for large datasets)
        parquet_file = self.output_dir / 'historical_patterns.parquet'
        patterns_df.to_parquet(parquet_file, index=False)
        print(f"Saved to {parquet_file}")
        print(f"  File size: {parquet_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Also save summary statistics
        summary = {
            'total_records': len(patterns_df),
            'unique_edges': int(patterns_df['edge_id'].nunique()),
            'days': patterns_df['day_of_week'].unique().tolist(),
            'hours': sorted(patterns_df['hour'].unique().tolist()),
            'avg_speed_kmh': float(patterns_df['speed_kmh'].mean()),
            'avg_jam_factor': float(patterns_df['jam_factor'].mean()),
            'created_at': datetime.now().isoformat(),
            'source': 'HERE Traffic API + OSM'
        }
        
        summary_file = self.output_dir / 'historical_patterns_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")
        
        # Print statistics by time
        print("\nTraffic patterns summary:")
        print("-" * 40)
        
        # By day
        print("\nAverage speed by day:")
        day_stats = patterns_df.groupby('day_of_week')['speed_kmh'].mean()
        for day in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']:
            if day in day_stats.index:
                print(f"  {day}: {day_stats[day]:.1f} km/h")
        
        # Peak hours
        print("\nAverage speed by hour (weekdays):")
        weekday_df = patterns_df[patterns_df['day_of_week'].isin(['MON', 'TUE', 'WED', 'THU', 'FRI'])]
        hour_stats = weekday_df.groupby('hour')['speed_kmh'].mean()
        
        peak_hours = [7, 8, 9, 17, 18, 19]
        for hour in peak_hours:
            if hour in hour_stats.index:
                print(f"  {hour:02d}:00 - {hour_stats[hour]:.1f} km/h")
    
    def run(self):
        """
        Run the complete historical traffic building pipeline.
        """
        print("\n" + "=" * 60)
        print("HERE Historical Traffic Builder")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Fetch base traffic
            segments = self.fetch_base_traffic()
            
            # Step 2: Match to OSM edges
            self.match_segments_to_edges(segments)
            
            # Step 3: Build historical patterns for matched edges
            patterns_df = self.build_historical_patterns(segments)
            
            # Step 4: Add unmatched edges with OSM-based patterns
            complete_df = self.add_unmatched_edges(patterns_df)
            
            # Step 5: Save patterns
            self.save_patterns(complete_df)
            
            print("\n" + "=" * 60)
            print("Historical traffic data built successfully!")
            print("=" * 60)
            
            return complete_df
            
        except Exception as e:
            print(f"\nError building historical traffic: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    print("=" * 60)
    print("HERE Traffic Historical Data Builder")
    print("=" * 60)
    
    # Check for API key
    api_key = os.environ.get('HERE_API_KEY')
    
    if not api_key:
        print("\nNo HERE_API_KEY found in environment.")
        print("\nTo use HERE Traffic API:")
        print("  1. Sign up at https://developer.here.com/")
        print("  2. Create a project and get an API key")
        print("  3. Set environment variable:")
        print("     set HERE_API_KEY=your_api_key_here")
        print("\nGenerating synthetic historical patterns instead...")
        
        # Generate synthetic patterns without API
        generate_synthetic_patterns()
        return
    
    # Run with API
    builder = HistoricalTrafficBuilder(api_key=api_key)
    builder.run()


def generate_synthetic_patterns():
    """
    Generate synthetic historical patterns when HERE API is not available.
    Uses OSM base speeds with time-based adjustments.
    """
    print("\n" + "=" * 60)
    print("Generating synthetic historical patterns")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load edges
    edges_file = project_root / 'data' / 'final' / 'edges_final.csv'
    edges_df = pd.read_csv(edges_file)
    
    # Filter to road layer
    road_edges = edges_df[edges_df['layer'] == 0].copy()
    print(f"Processing {len(road_edges):,} road edges")
    
    # Time patterns
    days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    hours = list(range(24))
    
    # Delhi-specific traffic patterns
    def get_adjustment(day: str, hour: int) -> float:
        is_weekend = day in ['SAT', 'SUN']
        
        if is_weekend:
            if 0 <= hour < 6:
                return 0.3
            elif 6 <= hour < 10:
                return 0.6
            elif 10 <= hour < 14:
                return 0.9
            elif 14 <= hour < 18:
                return 0.8
            elif 18 <= hour < 22:
                return 1.0
            else:
                return 0.5
        else:
            if 0 <= hour < 6:
                return 0.3
            elif 6 <= hour < 8:
                return 0.8
            elif 8 <= hour < 10:
                return 1.5
            elif 10 <= hour < 12:
                return 1.0
            elif 12 <= hour < 14:
                return 0.9
            elif 14 <= hour < 17:
                return 1.0
            elif 17 <= hour < 20:
                return 1.6
            elif 20 <= hour < 22:
                return 1.0
            else:
                return 0.5
    
    print("Building patterns...")
    records = []
    
    total = len(road_edges) * len(days) * len(hours)
    count = 0
    
    for _, edge in road_edges.iterrows():
        edge_id = edge['edge_id']
        base_speed_mps = edge['speed_mps']
        length_m = edge['length_m']
        
        for day in days:
            for hour in hours:
                count += 1
                
                # Get adjustment factor
                adjustment = get_adjustment(day, hour)
                
                # Calculate jam factor (higher adjustment = more congestion)
                jam_factor = min(10, max(0, (adjustment - 0.5) * 5))
                
                # Calculate speed (inverse of congestion)
                speed_factor = 1 - (jam_factor * 0.08)
                speed_mps = max(1.0, base_speed_mps * speed_factor)
                travel_time_s = length_m / speed_mps
                
                records.append({
                    'edge_id': edge_id,
                    'day_of_week': day,
                    'hour': hour,
                    'speed_mps': speed_mps,
                    'speed_kmh': speed_mps * 3.6,
                    'jam_factor': jam_factor,
                    'travel_time_s': travel_time_s,
                    'free_flow_mps': base_speed_mps,
                    'congestion_ratio': 1 - speed_factor
                })
        
        if count % 1000000 == 0:
            pct = count / total * 100
            print(f"  Progress: {count:,}/{total:,} ({pct:.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    print(f"\nCreated {len(df):,} pattern records")
    
    # Save
    output_dir = project_root / 'data' / 'here'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_file = output_dir / 'historical_patterns.parquet'
    df.to_parquet(parquet_file, index=False)
    print(f"Saved to {parquet_file}")
    print(f"  File size: {parquet_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Summary
    summary = {
        'total_records': len(df),
        'unique_edges': int(df['edge_id'].nunique()),
        'source': 'Synthetic (OSM-based)',
        'created_at': datetime.now().isoformat()
    }
    
    summary_file = output_dir / 'historical_patterns_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSynthetic historical patterns generated successfully!")


if __name__ == "__main__":
    main()
