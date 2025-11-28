"""
Main pipeline script to build complete multi-layer transportation network.
Orchestrates all data processing steps in correct order.
"""

import os
import sys
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.convert_road_layer import convert_road_layer
from data_processing.extract_metro_layer import create_synthetic_metro
from data_processing.create_walking_layer import create_walking_layer
from data_processing.create_transfers import create_transfer_connections, add_walking_layer_transfers
from data_processing.merge_all_layers import merge_all_layers
from data_processing.generate_multimodal_traffic import generate_multimodal_traffic


def print_separator():
    """Print a visual separator"""
    print("\n" + "="*80 + "\n")


def build_multilayer_network():
    """
    Execute complete pipeline to build multi-layer transportation network.
    """
    
    print("="*80)
    print("MULTI-LAYER URBAN TRANSPORTATION NETWORK BUILDER")
    print("="*80)
    print("\nThis script will:")
    print("  1. Convert road network to multi-layer format")
    print("  2. Create synthetic metro network")
    print("  3. Create walking layer")
    print("  4. Generate transfer connections")
    print("  5. Merge all layers")
    print("  6. Generate time-series traffic data")
    print("\n" + "="*80)
    
    start_time = time.time()
    
    # Define paths
    raw_dir = 'data/raw'
    multilayer_dir = 'data/multilayer'
    final_dir = 'data/final'
    
    # Create directories
    os.makedirs(multilayer_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    try:
        # Step 1: Convert road layer
        print_separator()
        print("STEP 1/6: Converting road network to multi-layer format...")
        print_separator()
        
        nodes_ml, edges_ml = convert_road_layer(
            nodes_file=f'{raw_dir}/nodes.csv',
            edges_file=f'{raw_dir}/edges.csv',
            output_dir=multilayer_dir
        )
        
        print(f"✓ Step 1 complete: {len(nodes_ml)} nodes, {len(edges_ml)} edges")
        
        # Step 2: Create metro layer
        print_separator()
        print("STEP 2/6: Creating synthetic metro network...")
        print_separator()
        
        metro_nodes, metro_edges = create_synthetic_metro(
            output_dir=multilayer_dir
        )
        
        print(f"✓ Step 2 complete: {len(metro_nodes)} stations, {len(metro_edges)} connections")
        
        # Step 3: Create walking layer
        print_separator()
        print("STEP 3/6: Creating walking layer...")
        print_separator()
        
        walk_nodes, walk_edges = create_walking_layer(
            road_edges_file=f'{multilayer_dir}/edges_multilayer.csv',
            road_nodes_file=f'{multilayer_dir}/nodes_multilayer.csv',
            output_dir=multilayer_dir,
            max_walk_distance=1000
        )
        
        print(f"✓ Step 3 complete: {len(walk_nodes)} nodes, {len(walk_edges)} edges")
        
        # Step 4: Create transfers
        print_separator()
        print("STEP 4/6: Generating transfer connections...")
        print_separator()
        
        # Metro transfers
        transfers = create_transfer_connections(
            road_nodes_file=f'{multilayer_dir}/nodes_multilayer.csv',
            metro_nodes_file=f'{multilayer_dir}/metro_nodes.csv',
            output_dir=multilayer_dir,
            max_transfer_distance=300
        )
        
        # Walking transfers
        walk_transfers = add_walking_layer_transfers(
            walk_nodes_file=f'{multilayer_dir}/walk_nodes.csv',
            road_nodes_file=f'{multilayer_dir}/nodes_multilayer.csv',
            output_dir=multilayer_dir
        )
        
        total_transfers = len(transfers) + len(walk_transfers)
        print(f"✓ Step 4 complete: {total_transfers} total transfers")
        
        # Step 5: Merge all layers
        print_separator()
        print("STEP 5/6: Merging all layers...")
        print_separator()
        
        nodes_final, edges_final, transfers_final = merge_all_layers(
            multilayer_dir=multilayer_dir,
            output_dir=final_dir
        )
        
        print(f"✓ Step 5 complete: {len(nodes_final)} nodes, {len(edges_final)} edges, {len(transfers_final)} transfers")
        
        # Step 6: Generate traffic data
        print_separator()
        print("STEP 6/6: Generating multimodal traffic time-series...")
        print_separator()
        
        generate_multimodal_traffic(
            edges_file=f'{final_dir}/edges_final.csv',
            output_file=f'{final_dir}/multimodal_timeseries.parquet',
            steps_per_hour=12,
            hours=24,
            chunk_size=10000
        )
        
        print(f"✓ Step 6 complete: Traffic data generated")
        
        # Final summary
        print_separator()
        print("PIPELINE COMPLETE!")
        print_separator()
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ All steps completed successfully in {elapsed:.1f} seconds")
        print(f"\nFinal network structure:")
        print(f"  - Nodes: {len(nodes_final):,}")
        print(f"  - Edges: {len(edges_final):,}")
        print(f"  - Transfers: {len(transfers_final):,}")
        print(f"  - Layers: 3 (Road, Metro, Walk)")
        print(f"\nOutput files saved to: {final_dir}/")
        print(f"  - nodes_final.csv")
        print(f"  - edges_final.csv")
        print(f"  - transfers_final.csv")
        print(f"  - multimodal_timeseries.parquet")
        print(f"  - network_summary.json")
        
        print("\n" + "="*80)
        print("Ready for multimodal routing!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = build_multilayer_network()
    sys.exit(0 if success else 1)
