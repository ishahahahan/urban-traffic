"""
generate_real_timeseries.py
Generate time-series data using real metro network extracted from OSM/GTFS.
Enhanced version with realistic traffic patterns for all transport modes.

MEMORY-EFFICIENT: Uses chunked processing with incremental parquet writes.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Pre-computed time-of-day multipliers (24 hours)
# Metro: Line-specific adjustments
METRO_BASE_TOD = np.array([
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95,  # 0-5: Late night
    1.05, 1.10, 1.15, 1.15, 1.00, 1.00,  # 6-11: Morning rush
    1.00, 1.00, 1.00, 1.00, 1.00, 1.05,  # 12-17: Midday to early evening
    1.20, 1.20, 1.05, 1.00, 0.95, 0.95   # 18-23: Evening rush
])

# Metro line congestion factors (busier lines)
METRO_LINE_FACTORS = {
    'Blue': 1.05, 'Blue Line': 1.05,
    'Yellow': 1.03, 'Yellow Line': 1.03,
    'Red': 1.02, 'Red Line': 1.02,
    'Magenta': 1.04, 'Magenta Line': 1.04,
    'Violet': 1.03, 'Violet Line': 1.03,
    'Pink': 1.02, 'Pink Line': 1.02,
    'Green': 1.01, 'Green Line': 1.01,
    'Orange': 1.01, 'Orange Line': 1.01,
    'Airport Express': 1.0,
}

# Road congestion pattern (inverse - lower = more congestion)
ROAD_SPEED_FACTOR = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 0.9,        # 0-5: Night/early morning
    0.7, 0.5, 0.4, 0.45, 0.7, 0.75,      # 6-11: Morning rush
    0.75, 0.75, 0.75, 0.75, 0.6, 0.4,    # 12-17: Midday to evening rush
    0.35, 0.45, 0.7, 0.85, 0.95, 1.0     # 18-23: Evening rush to night
])

# Walking is mostly constant
WALK_TOD = np.array([
    1.1, 1.1, 1.1, 1.1, 1.1, 1.0,        # 0-5: Night (slower for safety)
    1.0, 1.0, 1.05, 1.05, 1.0, 1.0,      # 6-11
    1.0, 1.0, 1.0, 1.0, 1.0, 1.05,       # 12-17
    1.05, 1.0, 1.0, 1.05, 1.1, 1.1       # 18-23
])


def generate_combined_timeseries(edges_final_file, output_dir,
                                 start_time='00:00', duration_hours=24,
                                 interval_minutes=5, edge_chunk_size=5000):
    """
    MEMORY-EFFICIENT: Generate time-series for ALL modes using chunked writes.
    
    Parameters:
    -----------
    edges_final_file : str
        Path to edges_final.csv (all layers)
    output_dir : str
        Output directory
    start_time : str
        Start time (ignored, always starts at 00:00)
    duration_hours : int
        Duration in hours
    interval_minutes : int
        Time granularity in minutes
    edge_chunk_size : int
        Edges per chunk (default: 5000)
        
    Returns:
    --------
    None (writes to parquet file)
    """
    
    print("\n" + "="*70)
    print("GENERATING MULTI-MODAL TIME-SERIES DATA (MEMORY-EFFICIENT)")
    print("="*70)
    
    # Load all edges
    edges = pd.read_csv(edges_final_file)
    n_edges = len(edges)
    
    print(f"\nTotal edges: {n_edges:,}")
    print("Edges by mode:")
    for mode, count in edges['mode'].value_counts().items():
        print(f"  {mode}: {count:,}")
    
    # Time parameters
    steps_per_hour = 60 // interval_minutes
    T = duration_hours * steps_per_hour
    
    print(f"\nTime intervals: {T} ({interval_minutes} min each)")
    print(f"Expected records: {n_edges * T:,}")
    print(f"Edge chunk size: {edge_chunk_size:,}")
    
    # Create time index
    time_index = pd.date_range("2025-11-25 00:00", periods=T, 
                               freq=f"{interval_minutes}min")
    hours_array = np.array([t.hour for t in time_index])
    minutes_array = np.array([t.minute for t in time_index])
    timestamps_str = time_index.strftime('%Y-%m-%d %H:%M:%S').values
    
    # Rush hour mask
    rush_mask = ((hours_array >= 8) & (hours_array < 10)) | \
                ((hours_array >= 18) & (hours_array < 20))
    
    # Set random seed
    np.random.seed(42)
    
    # Output file
    output_file = f'{output_dir}/multimodal_timeseries.parquet'
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Statistics tracking
    total_records = 0
    mode_stats = {}
    
    writer = None
    
    try:
        # Process each mode separately
        for mode in edges['mode'].unique():
            mode_edges = edges[edges['mode'] == mode].copy().reset_index(drop=True)
            n_mode = len(mode_edges)
            
            if n_mode == 0:
                continue
            
            print(f"\nProcessing {mode}: {n_mode:,} edges...")
            mode_stats[mode] = {'count': 0, 'congestion_sum': 0.0}
            
            # Process in chunks
            n_chunks = (n_mode + edge_chunk_size - 1) // edge_chunk_size
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * edge_chunk_size
                end_idx = min((chunk_idx + 1) * edge_chunk_size, n_mode)
                chunk_edges = mode_edges.iloc[start_idx:end_idx]
                n_chunk = len(chunk_edges)
                
                # Extract edge properties
                edge_ids = chunk_edges['edge_id'].values
                base_speeds = chunk_edges['speed_mps'].values
                lengths = chunk_edges['length_m'].values
                layers = chunk_edges['layer'].values
                
                # Per-edge random bias
                edge_biases = np.random.normal(1.0, 0.03, n_chunk)
                
                if mode == 'metro':
                    # Get line for each metro edge
                    lines = []
                    for _, edge in chunk_edges.iterrows():
                        try:
                            props = json.loads(edge['properties'])
                            lines.append(props.get('line', 'Unknown'))
                        except:
                            lines.append('Unknown')
                    
                    line_factors = np.array([METRO_LINE_FACTORS.get(l, 1.0) for l in lines])
                    tod_by_hour = METRO_BASE_TOD[hours_array]
                    
                    base_congestion = (line_factors[:, np.newaxis] * 
                                       edge_biases[:, np.newaxis] * 
                                       tod_by_hour[np.newaxis, :])
                    
                    delay_mask = np.random.random((n_chunk, T)) < 0.02
                    delay_factors = np.where(delay_mask, 
                                             np.random.uniform(1.1, 1.3, (n_chunk, T)), 
                                             1.0)
                    
                    boarding_delay = np.where(rush_mask, np.random.normal(15, 5, T), 0)
                    
                    congestion = base_congestion * delay_factors
                    congestion = np.clip(congestion, 0.95, 2.5)
                    
                    travel_times = (lengths[:, np.newaxis] / base_speeds[:, np.newaxis]) * congestion
                    travel_times += boarding_delay[np.newaxis, :]
                    
                    current_speeds = lengths[:, np.newaxis] / travel_times
                    
                elif mode == 'car':
                    speed_factors = ROAD_SPEED_FACTOR[hours_array]
                    noise = np.random.normal(0, 0.1, (n_chunk, T))
                    
                    effective_factors = (speed_factors[np.newaxis, :] * 
                                         edge_biases[:, np.newaxis] + noise)
                    effective_factors = np.clip(effective_factors, 0.2, 1.2)
                    
                    current_speeds = base_speeds[:, np.newaxis] * effective_factors
                    current_speeds = np.maximum(current_speeds, 1.0)
                    
                    travel_times = lengths[:, np.newaxis] / current_speeds
                    congestion = 1.0 / effective_factors
                    
                elif mode == 'walk':
                    tod_factors = WALK_TOD[hours_array]
                    noise = np.random.normal(0, 0.02, (n_chunk, T))
                    
                    congestion = (tod_factors[np.newaxis, :] * 
                                  edge_biases[:, np.newaxis] + noise)
                    congestion = np.clip(congestion, 0.9, 1.3)
                    
                    current_speeds = base_speeds[:, np.newaxis] / congestion
                    travel_times = lengths[:, np.newaxis] / current_speeds
                    
                else:
                    speed_factors = ROAD_SPEED_FACTOR[hours_array] * 1.05
                    noise = np.random.normal(0, 0.08, (n_chunk, T))
                    
                    effective_factors = speed_factors[np.newaxis, :] * edge_biases[:, np.newaxis] + noise
                    effective_factors = np.clip(effective_factors, 0.25, 1.15)
                    
                    current_speeds = base_speeds[:, np.newaxis] * effective_factors
                    travel_times = lengths[:, np.newaxis] / current_speeds
                    congestion = 1.0 / effective_factors
                
                # Build DataFrame for this chunk
                n_records = n_chunk * T
                
                df_chunk = pd.DataFrame({
                    'edge_id': np.repeat(edge_ids, T),
                    'timestamp': np.tile(timestamps_str, n_chunk),
                    'hour': np.tile(hours_array, n_chunk),
                    'minute': np.tile(minutes_array, n_chunk),
                    'mode': mode,
                    'layer': np.repeat(layers, T),
                    'current_speed_mps': current_speeds.ravel().astype(np.float32),
                    'travel_time_s': travel_times.ravel().astype(np.float32),
                    'congestion_factor': congestion.ravel().astype(np.float32)
                })
                
                # Convert to Arrow and write
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                
                writer.write_table(table)
                
                # Update stats
                total_records += n_records
                mode_stats[mode]['count'] += n_records
                mode_stats[mode]['congestion_sum'] += congestion.sum()
                
                # Clear memory
                del df_chunk, table, congestion, current_speeds, travel_times
                
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                    print(f"  Chunk {chunk_idx + 1}/{n_chunks} done")
    
    finally:
        if writer is not None:
            writer.close()
    
    file_size = os.path.getsize(output_file) / 1e6
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total records: {total_records:,}")
    print(f"File size: {file_size:.1f} MB")
    
    print("\nCongestion statistics by mode:")
    for mode, stats in mode_stats.items():
        avg = stats['congestion_sum'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {mode.upper()}: avg={avg:.3f}, records={stats['count']:,}")
    
    return None


if __name__ == "__main__":
    generate_combined_timeseries(
        edges_final_file='data/final/edges_final.csv',
        output_dir='data/final',
        start_time='00:00',
        duration_hours=24,
        interval_minutes=5
    )
    
    print("\n✓ Time-series generation complete!")
