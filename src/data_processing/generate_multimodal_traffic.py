"""
Generate synthetic time-series traffic data for multi-layer network.
Extends existing single-layer traffic generation to all transport modes.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tod_multiplier(hour, mode='car'):
    """
    Time-of-day congestion multiplier for different transport modes.
    
    Parameters:
    -----------
    hour : int
        Hour of day (0-23)
    mode : str
        Transport mode ('car', 'metro', 'walk')
        
    Returns:
    --------
    float : Congestion multiplier (1.0 = free flow, >1.0 = congested)
    """
    
    if mode == 'car':
        # Road traffic: heavy congestion during rush hours
        if 0 <= hour < 5:
            return 0.9
        elif 5 <= hour < 8:
            return 1.2
        elif 8 <= hour < 10:
            return 1.6  # Morning rush
        elif 10 <= hour < 16:
            return 1.1
        elif 16 <= hour < 19:
            return 1.7  # Evening rush
        elif 19 <= hour < 22:
            return 1.2
        else:
            return 0.95
            
    elif mode == 'metro':
        # Metro: slight delays during rush hours due to crowding
        if 0 <= hour < 6:
            return 1.0  # Empty trains, on time
        elif 6 <= hour < 9:
            return 1.1  # Morning rush, slight delays
        elif 9 <= hour < 17:
            return 1.0
        elif 17 <= hour < 20:
            return 1.15  # Evening rush
        elif 20 <= hour < 24:
            return 1.0
            
    elif mode == 'walk':
        # Walking: mostly constant, slight slowdown in crowds
        if 8 <= hour < 10 or 17 <= hour < 19:
            return 1.05  # Crowded sidewalks
        else:
            return 1.0
    
    return 1.0


def generate_multimodal_traffic(edges_file, output_file, 
                                steps_per_hour=12, hours=24,
                                chunk_size=10000):
    """
    Generate time-series traffic data for all transport modes.
    Uses chunked writing to avoid memory issues.
    
    Parameters:
    -----------
    edges_file : str
        Path to edges_final.csv
    output_file : str
        Output parquet file path
    steps_per_hour : int
        Number of time steps per hour (default: 12 = 5 min intervals)
    hours : int
        Number of hours to simulate (default: 24)
    chunk_size : int
        Number of rows to accumulate before flushing
        
    Returns:
    --------
    None (writes to file)
    """
    
    print(f"Generating multimodal traffic data...")
    print(f"Time resolution: {60//steps_per_hour} minutes")
    print(f"Duration: {hours} hours")
    
    # Load edges
    edges = pd.read_csv(edges_file)
    print(f"Loaded {len(edges)} edges")
    
    # Create time index
    T = hours * steps_per_hour
    time_index = pd.date_range("2025-11-25 00:00", periods=T, 
                              freq=f"{60//steps_per_hour}min")
    
    # Remove existing output if any
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing {output_file}")
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Add spatial zone (for correlation)
    edges['zone'] = edges['u'].apply(lambda x: hash(x) % 10)
    
    # Generate data in chunks
    buffer = []
    total_rows = 0
    first_write = True
    
    print("\nGenerating time-series data...")
    for idx, edge in tqdm(edges.iterrows(), total=len(edges)):
        base_speed = edge['speed_mps']
        length = edge['length_m']
        mode = edge['mode']
        edge_id = edge['edge_id']
        layer = edge['layer']
        zone = int(edge['zone'])
        
        # Per-edge deviation
        edge_bias = np.random.normal(loc=1.0, scale=0.05)
        
        for t, ts in enumerate(time_index):
            hour = ts.hour
            mult = tod_multiplier(hour, mode)
            
            # Zone effect (small spatial correlation)
            zone_effect = 1.0 + (zone - 5) * 0.01
            
            # Random jitter
            jitter = np.random.normal(0, 0.05)
            
            # Calculate congestion factor
            congestion_factor = mult * zone_effect * edge_bias + jitter
            
            # Apply to speed
            current_speed = max(0.5, base_speed / congestion_factor)
            travel_time = length / current_speed
            
            buffer.append({
                'edge_id': edge_id,
                'timestamp': ts,
                'hour': hour,
                'mode': mode,
                'layer': layer,
                'current_speed_mps': current_speed,
                'travel_time_s': travel_time,
                'congestion_factor': congestion_factor
            })
            total_rows += 1
            
            # Flush buffer if needed
            if len(buffer) >= chunk_size:
                df_chunk = pd.DataFrame(buffer)
                if first_write:
                    df_chunk.to_parquet(output_file, index=False)
                    first_write = False
                else:
                    # Append to existing file
                    df_existing = pd.read_parquet(output_file)
                    df_combined = pd.concat([df_existing, df_chunk], ignore_index=True)
                    df_combined.to_parquet(output_file, index=False)
                buffer = []
    
    # Final flush
    if len(buffer) > 0:
        df_chunk = pd.DataFrame(buffer)
        if first_write:
            df_chunk.to_parquet(output_file, index=False)
        else:
            df_existing = pd.read_parquet(output_file)
            df_combined = pd.concat([df_existing, df_chunk], ignore_index=True)
            df_combined.to_parquet(output_file, index=False)
    
    print(f"\nGenerated {total_rows:,} time-series records")
    print(f"Saved to {output_file}")
    
    # Load and display summary
    df = pd.read_parquet(output_file)
    print("\nData summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique edges: {df['edge_id'].nunique()}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Modes: {df['mode'].value_counts().to_dict()}")
    print(f"\nAverage travel times by mode:")
    print(df.groupby('mode')['travel_time_s'].mean().round(1))
    print(f"\nAverage congestion by mode:")
    print(df.groupby('mode')['congestion_factor'].mean().round(3))


if __name__ == "__main__":
    # Example usage
    generate_multimodal_traffic(
        edges_file='data/final/edges_final.csv',
        output_file='data/final/multimodal_timeseries.parquet',
        steps_per_hour=12,
        hours=24,
        chunk_size=10000
    )
    
    print("\nMultimodal traffic generation complete!")
