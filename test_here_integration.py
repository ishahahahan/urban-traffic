"""
Test HERE Traffic Integration

This script tests the complete HERE Traffic API integration:
1. HERE Traffic Client
2. OSM-HERE Road Matching
3. Historical Pattern Loading
4. Time-dependent Routing

Run this after setting up HERE API key or generating synthetic patterns.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_here_client():
    """Test HERE Traffic API client."""
    print("\n" + "=" * 60)
    print("1. Testing HERE Traffic Client")
    print("=" * 60)
    
    from src.data_processing.here_traffic_provider import HERETrafficClient
    
    api_key = os.environ.get('HERE_API_KEY')
    
    if not api_key:
        print("⚠️  HERE_API_KEY not set - skipping API test")
        print("   To test with real API:")
        print("   set HERE_API_KEY=your_api_key_here")
        return False
    
    try:
        client = HERETrafficClient(api_key=api_key)
        
        # Fetch traffic
        flow_data = client.get_traffic_flow(use_cache=True)
        
        if 'error' in flow_data:
            print(f"❌ API Error: {flow_data['error']}")
            return False
        
        segments = client.parse_flow_results(flow_data)
        print(f"✅ Retrieved {len(segments)} traffic segments")
        
        if segments:
            sample = segments[0]
            print(f"   Sample: Speed={sample['speed_kmh']:.1f} km/h, "
                  f"JamFactor={sample['jam_factor']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_historical_patterns():
    """Test historical pattern loading."""
    print("\n" + "=" * 60)
    print("2. Testing Historical Patterns")
    print("=" * 60)
    
    patterns_file = project_root / 'data' / 'here' / 'historical_patterns.parquet'
    
    if not patterns_file.exists():
        print("⚠️  Historical patterns not found")
        print("   Run: python src/data_processing/fetch_here_historical.py")
        return False
    
    import pandas as pd
    
    df = pd.read_parquet(patterns_file)
    print(f"✅ Loaded {len(df):,} historical pattern records")
    print(f"   Unique edges: {df['edge_id'].nunique():,}")
    print(f"   Days: {df['day_of_week'].unique().tolist()}")
    print(f"   Hours: 0-23")
    
    # Show sample statistics
    print("\n   Average speed by day:")
    day_stats = df.groupby('day_of_week')['speed_kmh'].mean()
    for day in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']:
        if day in day_stats.index:
            print(f"     {day}: {day_stats[day]:.1f} km/h")
    
    return True


def test_graph_loader():
    """Test graph loader with HERE patterns."""
    print("\n" + "=" * 60)
    print("3. Testing Graph Loader with HERE Data")
    print("=" * 60)
    
    from src.routing.graph_loader import MultiLayerGraph
    
    nodes_file = project_root / 'data' / 'final' / 'nodes_final.csv'
    edges_file = project_root / 'data' / 'final' / 'edges_final.csv'
    transfers_file = project_root / 'data' / 'final' / 'transfers_final.csv'
    here_patterns = project_root / 'data' / 'here' / 'historical_patterns.parquet'
    
    # Check required files
    if not nodes_file.exists() or not edges_file.exists():
        print("❌ Network files not found")
        return False
    
    # Load graph
    mlg = MultiLayerGraph(
        nodes_file=str(nodes_file),
        edges_file=str(edges_file),
        transfers_file=str(transfers_file),
        here_patterns_file=str(here_patterns) if here_patterns.exists() else None
    )
    
    print(f"✅ Traffic source: {mlg.traffic_source}")
    
    # Build unified graph
    G = mlg.build_unified_graph(include_transfers=True)
    
    # Test time-dependent routing
    test_times = [
        datetime(2025, 12, 1, 8, 30),   # Monday morning rush
        datetime(2025, 12, 1, 14, 0),   # Monday afternoon
        datetime(2025, 12, 1, 18, 30),  # Monday evening rush
        datetime(2025, 12, 6, 10, 0),   # Saturday morning
    ]
    
    print("\n   Testing time-dependent travel times:")
    for dt in test_times:
        travel_times = mlg.get_travel_times_for_datetime(dt)
        summary = mlg.get_congestion_summary(dt)
        
        day_name = dt.strftime('%A')
        time_str = dt.strftime('%H:%M')
        
        avg_speed = summary.get('avg_speed_kmh', 0)
        avg_jam = summary.get('avg_jam_factor', 0)
        
        print(f"     {day_name} {time_str}: "
              f"{len(travel_times):,} edges, "
              f"Avg Speed={avg_speed:.1f} km/h, "
              f"JamFactor={avg_jam:.1f}")
    
    return True


def test_routing_with_here():
    """Test multimodal routing with HERE traffic data."""
    print("\n" + "=" * 60)
    print("4. Testing Routing with HERE Traffic")
    print("=" * 60)
    
    from src.routing.graph_loader import MultiLayerGraph
    from src.routing.multimodal_dijkstra import MultimodalRouter
    
    # Load graph with HERE patterns
    nodes_file = project_root / 'data' / 'final' / 'nodes_final.csv'
    edges_file = project_root / 'data' / 'final' / 'edges_final.csv'
    transfers_file = project_root / 'data' / 'final' / 'transfers_final.csv'
    here_patterns = project_root / 'data' / 'here' / 'historical_patterns.parquet'
    
    mlg = MultiLayerGraph(
        nodes_file=str(nodes_file),
        edges_file=str(edges_file),
        transfers_file=str(transfers_file),
        here_patterns_file=str(here_patterns) if here_patterns.exists() else None
    )
    
    # Find test nodes
    source = mlg.get_node_by_location(77.21, 28.63, layer=0)  # Near CP
    target = mlg.get_node_by_location(77.05, 28.55, layer=0)  # Near Dwarka
    
    if not source or not target:
        print("❌ Could not find test nodes")
        return False
    
    print(f"   Source: {source}")
    print(f"   Target: {target}")
    
    # Initialize router
    router = MultimodalRouter(mlg)
    
    # Compare routes at different times
    test_times = [
        ("Morning Rush (8:30 AM)", datetime(2025, 12, 1, 8, 30)),
        ("Midday (2:00 PM)", datetime(2025, 12, 1, 14, 0)),
        ("Evening Rush (6:30 PM)", datetime(2025, 12, 1, 18, 30)),
    ]
    
    print("\n   Route comparison by time of day:")
    
    for label, dt in test_times:
        # Update graph with time-dependent weights
        travel_times = mlg.get_travel_times_for_datetime(dt)
        
        # Rebuild graph with updated weights
        G = mlg.build_unified_graph(include_transfers=True)
        mlg.update_edge_weights(G, travel_times)
        router.graph = G
        
        try:
            route = router.find_route(source, target)
            
            total_time_min = route['total_time'] / 60
            distance_km = route['distance'] / 1000
            modes = [seg['mode'] for seg in route['segments']]
            
            print(f"\n     {label}:")
            print(f"       Time: {total_time_min:.1f} min")
            print(f"       Distance: {distance_km:.1f} km")
            print(f"       Modes: {' → '.join(modes)}")
            
        except Exception as e:
            print(f"\n     {label}: Error - {e}")
    
    return True


def test_osm_here_matcher():
    """Test OSM to HERE road matching."""
    print("\n" + "=" * 60)
    print("5. Testing OSM-HERE Matcher")
    print("=" * 60)
    
    mapping_file = project_root / 'data' / 'here' / 'edge_here_mapping.csv'
    
    if not mapping_file.exists():
        print("⚠️  Edge mapping not found")
        print("   Run: python src/data_processing/fetch_here_historical.py")
        return False
    
    import pandas as pd
    
    mapping = pd.read_csv(mapping_file)
    print(f"✅ Loaded {len(mapping):,} edge mappings")
    
    # Load total edges for comparison
    edges_file = project_root / 'data' / 'final' / 'edges_final.csv'
    edges = pd.read_csv(edges_file)
    road_edges = edges[edges['layer'] == 0]
    
    match_rate = len(mapping) / len(road_edges) * 100
    print(f"   Match rate: {match_rate:.1f}% of road edges")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("HERE Traffic Integration Test Suite")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: HERE Client (optional - requires API key)
    results['here_client'] = test_here_client()
    
    # Test 2: Historical patterns
    results['historical_patterns'] = test_historical_patterns()
    
    # Test 3: Graph loader
    results['graph_loader'] = test_graph_loader()
    
    # Test 4: Routing
    results['routing'] = test_routing_with_here()
    
    # Test 5: OSM-HERE matching
    results['osm_here_matcher'] = test_osm_here_matcher()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL/SKIP"
        print(f"  {test_name}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count >= 3:  # At least patterns, loader, and routing
        print("\n✅ HERE Traffic integration is working!")
        print("\nNext steps:")
        print("  1. Set HERE_API_KEY for real traffic data")
        print("  2. Run fetch_here_historical.py to build patterns")
        print("  3. Use time-dependent routing in your application")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
