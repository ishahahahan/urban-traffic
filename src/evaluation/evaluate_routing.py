"""
evaluate_routing.py
Comprehensive evaluation of routing algorithms.
Compare single-mode vs multimodal routing strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")


def compare_routing_strategies(router, test_pairs, departure_times):
    """
    Compare different routing strategies.
    
    Parameters:
    -----------
    router : MultimodalRouter
        Initialized router instance
    test_pairs : list of tuples
        [(source1, target1), (source2, target2), ...]
    departure_times : list of datetime
        Different times to test
        
    Returns:
    --------
    DataFrame : comparison results
    """
    
    print("\n" + "="*70)
    print("COMPARING ROUTING STRATEGIES")
    print("="*70)
    
    results = []
    total_tests = len(test_pairs) * len(departure_times)
    test_num = 0
    
    for source, target in test_pairs:
        for dep_time in departure_times:
            test_num += 1
            print(f"\rTest {test_num}/{total_tests}: {source} → {target} at {dep_time.strftime('%H:%M')}", end='')
            
            try:
                # Strategy 1: Car only (high penalty for metro/walk)
                car_route = router.find_route(
                    source, target, dep_time,
                    mode_preferences={'car': 1.0, 'metro': 999, 'walk': 999}
                )
                
                # Strategy 2: Metro-preferred multimodal
                metro_route = router.find_route(
                    source, target, dep_time,
                    mode_preferences={'car': 1.2, 'metro': 0.8, 'walk': 1.0}
                )
                
                # Strategy 3: Balanced multimodal
                balanced_route = router.find_route(
                    source, target, dep_time,
                    mode_preferences={'car': 1.0, 'metro': 1.0, 'walk': 1.0}
                )
                
                results.append({
                    'source': source,
                    'target': target,
                    'departure_time': dep_time,
                    'departure_hour': dep_time.hour,
                    'car_only_time': car_route['total_time'] if car_route else None,
                    'metro_preferred_time': metro_route['total_time'] if metro_route else None,
                    'balanced_time': balanced_route['total_time'] if balanced_route else None,
                    'car_only_transfers': car_route.get('transfers', 0) if car_route else None,
                    'metro_preferred_transfers': metro_route.get('transfers', 0) if metro_route else None,
                    'balanced_transfers': balanced_route.get('transfers', 0) if balanced_route else None
                })
                
            except Exception as e:
                print(f"\nError testing {source} → {target}: {e}")
                results.append({
                    'source': source,
                    'target': target,
                    'departure_time': dep_time,
                    'departure_hour': dep_time.hour,
                    'car_only_time': None,
                    'metro_preferred_time': None,
                    'balanced_time': None,
                    'car_only_transfers': None,
                    'metro_preferred_transfers': None,
                    'balanced_transfers': None
                })
    
    print("\n")
    
    df = pd.DataFrame(results)
    
    # Calculate improvement metrics
    df['improvement_pct'] = np.where(
        df['car_only_time'].notna() & df['balanced_time'].notna(),
        (df['car_only_time'] - df['balanced_time']) / df['car_only_time'] * 100,
        np.nan
    )
    
    return df


def plot_time_comparison(results, output_dir='output'):
    """
    Plot travel time comparison visualizations.
    
    Parameters:
    -----------
    results : DataFrame
        Results from compare_routing_strategies
    output_dir : str
        Output directory for plots
    """
    
    if not PLOTTING_AVAILABLE:
        print("Plotting not available - matplotlib/seaborn not installed")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter valid results
    valid_results = results.dropna(subset=['car_only_time', 'balanced_time'])
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Travel time by hour
    hourly = valid_results.groupby('departure_hour').agg({
        'car_only_time': 'mean',
        'metro_preferred_time': 'mean',
        'balanced_time': 'mean'
    }).reset_index()
    
    ax = axes[0]
    ax.plot(hourly['departure_hour'], hourly['car_only_time']/60, 
            'o-', label='Car Only', linewidth=2, markersize=6)
    ax.plot(hourly['departure_hour'], hourly['metro_preferred_time']/60, 
            's-', label='Metro Preferred', linewidth=2, markersize=6)
    ax.plot(hourly['departure_hour'], hourly['balanced_time']/60, 
            '^-', label='Balanced', linewidth=2, markersize=6)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Travel Time (minutes)', fontsize=12)
    ax.set_title('Travel Time by Departure Hour', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: Improvement percentage
    improvement_by_hour = valid_results.groupby('departure_hour')['improvement_pct'].mean()
    
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in improvement_by_hour.values]
    ax.bar(improvement_by_hour.index, improvement_by_hour.values, color=colors, alpha=0.7)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Time Savings (%)', fontsize=12)
    ax.set_title('Multimodal Improvement vs Car-Only', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    output_file = f'{output_dir}/routing_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_file}")


def plot_mode_usage(results, output_dir='output'):
    """
    Plot mode usage analysis.
    
    Parameters:
    -----------
    results : DataFrame
        Results with mode information
    output_dir : str
        Output directory
    """
    
    if not PLOTTING_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Additional visualization for mode usage could be added here
    pass


def analyze_results(results):
    """
    Analyze routing comparison results.
    
    Parameters:
    -----------
    results : DataFrame
        Results from compare_routing_strategies
        
    Returns:
    --------
    dict : Analysis summary
    """
    
    # Filter valid results
    valid = results.dropna(subset=['car_only_time', 'balanced_time'])
    
    if len(valid) == 0:
        return {'error': 'No valid results to analyze'}
    
    analysis = {
        'total_tests': len(results),
        'valid_tests': len(valid),
        'car_only': {
            'mean_time_s': float(valid['car_only_time'].mean()),
            'mean_time_min': float(valid['car_only_time'].mean() / 60),
            'std_time_s': float(valid['car_only_time'].std()),
            'min_time_s': float(valid['car_only_time'].min()),
            'max_time_s': float(valid['car_only_time'].max())
        },
        'multimodal_balanced': {
            'mean_time_s': float(valid['balanced_time'].mean()),
            'mean_time_min': float(valid['balanced_time'].mean() / 60),
            'std_time_s': float(valid['balanced_time'].std()),
            'min_time_s': float(valid['balanced_time'].min()),
            'max_time_s': float(valid['balanced_time'].max())
        },
        'improvement': {
            'mean_pct': float(valid['improvement_pct'].mean()),
            'median_pct': float(valid['improvement_pct'].median()),
            'max_pct': float(valid['improvement_pct'].max()),
            'min_pct': float(valid['improvement_pct'].min()),
            'positive_improvements': int((valid['improvement_pct'] > 0).sum()),
            'negative_improvements': int((valid['improvement_pct'] < 0).sum())
        }
    }
    
    # Peak vs off-peak analysis
    peak_mask = (valid['departure_hour'] >= 8) & (valid['departure_hour'] < 10) | \
                (valid['departure_hour'] >= 17) & (valid['departure_hour'] < 20)
    
    peak = valid[peak_mask]
    off_peak = valid[~peak_mask]
    
    if len(peak) > 0:
        analysis['peak_hours'] = {
            'count': len(peak),
            'mean_improvement_pct': float(peak['improvement_pct'].mean()),
            'car_mean_time_min': float(peak['car_only_time'].mean() / 60),
            'multimodal_mean_time_min': float(peak['balanced_time'].mean() / 60)
        }
    
    if len(off_peak) > 0:
        analysis['off_peak_hours'] = {
            'count': len(off_peak),
            'mean_improvement_pct': float(off_peak['improvement_pct'].mean()),
            'car_mean_time_min': float(off_peak['car_only_time'].mean() / 60),
            'multimodal_mean_time_min': float(off_peak['balanced_time'].mean() / 60)
        }
    
    return analysis


def print_analysis_summary(analysis):
    """
    Print formatted analysis summary.
    
    Parameters:
    -----------
    analysis : dict
        Analysis results from analyze_results
    """
    
    print("\n" + "="*70)
    print("ROUTING COMPARISON ANALYSIS")
    print("="*70)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"\nTest Statistics:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Valid tests: {analysis['valid_tests']}")
    
    print(f"\nCar-Only Routing:")
    print(f"  Average time: {analysis['car_only']['mean_time_min']:.1f} min")
    print(f"  Std deviation: {analysis['car_only']['std_time_s']/60:.1f} min")
    print(f"  Range: {analysis['car_only']['min_time_s']/60:.1f} - {analysis['car_only']['max_time_s']/60:.1f} min")
    
    print(f"\nMultimodal Routing (Balanced):")
    print(f"  Average time: {analysis['multimodal_balanced']['mean_time_min']:.1f} min")
    print(f"  Std deviation: {analysis['multimodal_balanced']['std_time_s']/60:.1f} min")
    print(f"  Range: {analysis['multimodal_balanced']['min_time_s']/60:.1f} - {analysis['multimodal_balanced']['max_time_s']/60:.1f} min")
    
    print(f"\nImprovement (Multimodal vs Car-Only):")
    print(f"  Mean improvement: {analysis['improvement']['mean_pct']:.1f}%")
    print(f"  Median improvement: {analysis['improvement']['median_pct']:.1f}%")
    print(f"  Best improvement: {analysis['improvement']['max_pct']:.1f}%")
    print(f"  Worst case: {analysis['improvement']['min_pct']:.1f}%")
    print(f"  Routes improved: {analysis['improvement']['positive_improvements']}")
    print(f"  Routes slower: {analysis['improvement']['negative_improvements']}")
    
    if 'peak_hours' in analysis:
        print(f"\nPeak Hours (8-10 AM, 5-8 PM):")
        print(f"  Number of tests: {analysis['peak_hours']['count']}")
        print(f"  Mean improvement: {analysis['peak_hours']['mean_improvement_pct']:.1f}%")
        print(f"  Car avg time: {analysis['peak_hours']['car_mean_time_min']:.1f} min")
        print(f"  Multimodal avg time: {analysis['peak_hours']['multimodal_mean_time_min']:.1f} min")
    
    if 'off_peak_hours' in analysis:
        print(f"\nOff-Peak Hours:")
        print(f"  Number of tests: {analysis['off_peak_hours']['count']}")
        print(f"  Mean improvement: {analysis['off_peak_hours']['mean_improvement_pct']:.1f}%")
        print(f"  Car avg time: {analysis['off_peak_hours']['car_mean_time_min']:.1f} min")
        print(f"  Multimodal avg time: {analysis['off_peak_hours']['multimodal_mean_time_min']:.1f} min")
    
    print("\n" + "="*70)


def run_evaluation(router, num_test_pairs=10, output_dir='output'):
    """
    Run full evaluation pipeline.
    
    Parameters:
    -----------
    router : MultimodalRouter
        Initialized router
    num_test_pairs : int
        Number of random test pairs to generate
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    tuple : (results_df, analysis_dict)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available nodes for testing
    nodes = router.nodes
    road_nodes = nodes[nodes['layer'] == 0]['node_id'].tolist()
    
    if len(road_nodes) < 2:
        print("Error: Not enough road nodes for testing")
        return None, None
    
    # Generate random test pairs
    np.random.seed(42)
    test_pairs = []
    for _ in range(num_test_pairs):
        source, target = np.random.choice(road_nodes, size=2, replace=False)
        test_pairs.append((source, target))
    
    print(f"Generated {len(test_pairs)} test pairs")
    
    # Generate departure times (every 2 hours from 6 AM to 10 PM)
    base_date = datetime(2025, 11, 25)
    departure_times = [base_date.replace(hour=h) for h in range(6, 23, 2)]
    
    print(f"Testing at {len(departure_times)} different times")
    
    # Run comparison
    results = compare_routing_strategies(router, test_pairs, departure_times)
    
    # Save results
    results.to_csv(f'{output_dir}/routing_comparison_results.csv', index=False)
    print(f"✓ Saved results to {output_dir}/routing_comparison_results.csv")
    
    # Analyze
    analysis = analyze_results(results)
    print_analysis_summary(analysis)
    
    # Save analysis
    with open(f'{output_dir}/routing_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to {output_dir}/routing_analysis.json")
    
    # Plot
    if PLOTTING_AVAILABLE:
        plot_time_comparison(results, output_dir)
    
    return results, analysis


if __name__ == "__main__":
    from routing.graph_loader import MultiLayerGraph
    from routing.multimodal_dijkstra import MultimodalRouter
    
    print("Loading network...")
    
    # Load router
    try:
        mlg = MultiLayerGraph(
            nodes_file='data/final/nodes_final.csv',
            edges_file='data/final/edges_final.csv',
            transfers_file='data/final/transfers_final.csv',
            timeseries_file='data/final/multimodal_timeseries.parquet'
        )
        
        router = MultimodalRouter(mlg)
        
        # Run evaluation
        results, analysis = run_evaluation(
            router,
            num_test_pairs=20,
            output_dir='output'
        )
        
        print("\n✓ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found: {e}")
        print("Please run build_multilayer_network.py first to generate the network data.")
        sys.exit(1)
