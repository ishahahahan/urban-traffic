"""
Generate synthetic time-series traffic data for multi-layer network.
Extends existing single-layer traffic generation to all transport modes.

MEMORY-EFFICIENT VERSION: Processes in chunks and writes incrementally.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Pre-computed time-of-day multipliers for each hour (0-23)
TOD_MULTIPLIERS = {
    'car': np.array([0.9, 0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.6, 1.6, 
                     1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.7, 1.7, 1.7, 1.2, 
                     1.2, 1.2, 0.95, 0.95]),
    'metro': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15, 1.15, 1.15,
                       1.0, 1.0, 1.0, 1.0]),
    'walk': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.05,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.0,
                      1.0, 1.0, 1.0, 1.0]),
    'auto': np.array([0.9, 0.9, 0.9, 0.9, 0.9, 1.15, 1.15, 1.15, 1.5, 1.5, 
                      1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.6, 1.6, 1.6, 1.15, 
                      1.15, 1.15, 0.95, 0.95]),
}


def generate_multimodal_traffic_fast(edges_file, output_file, 
                                     steps_per_hour=12, hours=24,
                                     edge_chunk_size=5000):
    """
    MEMORY-EFFICIENT: Generate time-series in chunks and write incrementally.
    
    Processes edges in batches to avoid memory errors on large datasets.
    
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
    edge_chunk_size : int
        Number of edges to process at once (default: 5000)
    """
    
    print(f"Generating multimodal traffic data (MEMORY-EFFICIENT)...")
    print(f"Time resolution: {60//steps_per_hour} minutes")
    print(f"Duration: {hours} hours")
    print(f"Edge chunk size: {edge_chunk_size:,}")
    
    # Load edges
    edges = pd.read_csv(edges_file)
    n_edges = len(edges)
    print(f"Loaded {n_edges:,} edges")
    
    # Time parameters
    T = hours * steps_per_hour
    interval_minutes = 60 // steps_per_hour
    
    print(f"Total time intervals: {T}")
    print(f"Expected records: {n_edges * T:,}")
    
    # Create time index
    time_index = pd.date_range("2025-11-25 00:00", periods=T, 
                               freq=f"{interval_minutes}min")
    hours_array = np.array([t.hour for t in time_index])
    timestamps_str = time_index.strftime('%Y-%m-%d %H:%M:%S').values
    
    # Initialize random seed
    np.random.seed(42)
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Track statistics
    total_records = 0
    mode_records = {}
    mode_congestion_sum = {}
    mode_congestion_count = {}
    
    writer = None
    schema = None
    
    try:
        # Process each mode
        for mode in edges['mode'].unique():
            mode_edges = edges[edges['mode'] == mode].copy().reset_index(drop=True)
            n_mode = len(mode_edges)
            
            if n_mode == 0:
                continue
                
            print(f"\nProcessing {mode}: {n_mode:,} edges...")
            mode_records[mode] = 0
            mode_congestion_sum[mode] = 0.0
            mode_congestion_count[mode] = 0
            
            # Get TOD multipliers for this mode
            tod_mult = TOD_MULTIPLIERS.get(mode, TOD_MULTIPLIERS['car'])
            hour_multipliers = tod_mult[hours_array]  # Shape: (T,)
            
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
                
                # Per-edge random biases
                edge_biases = np.random.normal(1.0, 0.05, n_chunk)
                
                # Random jitter: (n_chunk, T)
                jitter = np.random.normal(0, 0.05, (n_chunk, T))
                
                # Congestion factor: (n_chunk, T)
                congestion = (edge_biases[:, np.newaxis] * hour_multipliers[np.newaxis, :]) + jitter
                congestion = np.clip(congestion, 0.5, 3.0)
                
                # Current speeds: (n_chunk, T)
                current_speeds = base_speeds[:, np.newaxis] / congestion
                current_speeds = np.maximum(current_speeds, 0.5)
                
                # Travel times: (n_chunk, T)
                travel_times = lengths[:, np.newaxis] / current_speeds
                
                # Build DataFrame for this chunk
                n_records = n_chunk * T
                
                df_chunk = pd.DataFrame({
                    'edge_id': np.repeat(edge_ids, T),
                    'timestamp': np.tile(timestamps_str, n_chunk),
                    'hour': np.tile(hours_array, n_chunk),
                    'mode': mode,
                    'layer': np.repeat(layers, T),
                    'current_speed_mps': current_speeds.ravel().astype(np.float32),
                    'travel_time_s': travel_times.ravel().astype(np.float32),
                    'congestion_factor': congestion.ravel().astype(np.float32)
                })
                
                # Convert to Arrow table
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                
                # Initialize writer on first chunk
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_file, schema, compression='snappy')
                
                # Write chunk
                writer.write_table(table)
                
                # Update stats
                total_records += n_records
                mode_records[mode] += n_records
                mode_congestion_sum[mode] += congestion.sum()
                mode_congestion_count[mode] += congestion.size
                
                # Clear memory
                del df_chunk, table, congestion, current_speeds, travel_times, jitter
                
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                    print(f"  Chunk {chunk_idx + 1}/{n_chunks} done ({end_idx:,}/{n_mode:,} edges)")
    
    finally:
        if writer is not None:
            writer.close()
    
    # Summary
    file_size = os.path.getsize(output_file) / 1e6
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {total_records:,}")
    print(f"File size: {file_size:.1f} MB")
    
    print(f"\nRecords by mode:")
    for mode, count in mode_records.items():
        print(f"  {mode}: {count:,}")
    
    print(f"\nAverage congestion by mode:")
    for mode in mode_records.keys():
        avg = mode_congestion_sum[mode] / mode_congestion_count[mode]
        print(f"  {mode}: {avg:.3f}")
    
    return None  # Don't return DataFrame to save memory


# Keep old function name for compatibility
def generate_multimodal_traffic(edges_file, output_file, 
                                steps_per_hour=12, hours=24,
                                chunk_size=10000):
    """
    Wrapper that calls the memory-efficient version.
    """
    return generate_multimodal_traffic_fast(
        edges_file=edges_file,
        output_file=output_file,
        steps_per_hour=steps_per_hour,
        hours=hours,
        edge_chunk_size=min(chunk_size, 5000)
    )


if __name__ == "__main__":
    # Example usage
    generate_multimodal_traffic_fast(
        edges_file='data/final/edges_final.csv',
        output_file='data/final/multimodal_timeseries.parquet',
        steps_per_hour=12,
        hours=24
    )
    
    print("\nMultimodal traffic generation complete!")
