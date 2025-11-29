"""
Test Multimodal Dijkstra Routing
================================
This script tests the multimodal routing algorithm on the final network,
checking if routes properly traverse multiple network layers (road, metro, walk).
"""

import sys
sys.path.append('src')

from routing.graph_loader import MultiLayerGraph
from routing.multimodal_dijkstra import MultimodalRouter
from visualization.plot_network import plot_route, LAYER_COLORS, MODE_COLORS
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd


def print_route_details(route):
    """Print detailed route information"""
    if not route:
        print("  No route found!")
        return
    
    print(f"\n  Total Time: {route['total_time_s']/60:.1f} min")
    print(f"  Total Distance: {route['total_distance_m']/1000:.2f} km")
    print(f"  Number of Segments: {len(route['segments'])}")
    print(f"  Number of Transfers: {route['num_transfers']}")
    
    print("\n  Segments:")
    for i, seg in enumerate(route['segments']):
        mode = seg['mode'].upper()
        time_min = seg['time_s'] / 60
        dist_km = seg['distance_m'] / 1000
        print(f"    {i+1}. {mode}: {time_min:.1f} min, {dist_km:.2f} km")
    
    if route['transfers']:
        print("\n  Transfers:")
        for t in route['transfers']:
            layer_names = {0: 'Road', 1: 'Metro', 2: 'Walk'}
            from_layer = layer_names.get(t['from_layer'], t['from_layer'])
            to_layer = layer_names.get(t['to_layer'], t['to_layer'])
            print(f"    - {t['transfer_type']}: {from_layer} -> {to_layer}")


def visualize_route(route, nodes_df, title="Multimodal Route", figsize=(14, 10)):
    """
    Visualize a route on the map showing the path through different layers.
    """
    if not route:
        print("No route to visualize!")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create node lookup
    node_coords = nodes_df.set_index('node_id')[['x', 'y', 'layer']].to_dict('index')
    
    # Get bounding box of route with padding
    route_nodes = [n for n in route['path'] if n in node_coords]
    if not route_nodes:
        print("No valid nodes in route!")
        return None, None
    
    xs = [node_coords[n]['x'] for n in route_nodes]
    ys = [node_coords[n]['y'] for n in route_nodes]
    
    padding = 0.01  # degrees
    x_min, x_max = min(xs) - padding, max(xs) + padding
    y_min, y_max = min(ys) - padding, max(ys) + padding
    
    # Plot background nodes (within bounding box for speed)
    bg_nodes = nodes_df[
        (nodes_df['x'] >= x_min) & (nodes_df['x'] <= x_max) &
        (nodes_df['y'] >= y_min) & (nodes_df['y'] <= y_max)
    ]
    
    # Plot by layer
    for layer in [0, 1, 2]:
        layer_nodes = bg_nodes[bg_nodes['layer'] == layer]
        color = LAYER_COLORS[layer]
        size = 50 if layer == 1 else 3
        alpha = 0.4 if layer == 1 else 0.2
        ax.scatter(layer_nodes['x'], layer_nodes['y'], c=color, s=size, alpha=alpha)
    
    # Plot route segments with different colors
    MODE_PLOT_COLORS = {
        'car': '#E74C3C',      # Red
        'metro': '#27AE60',    # Green
        'walk': '#3498DB',     # Blue
    }
    
    for seg in route['segments']:
        mode = seg['mode']
        color = MODE_PLOT_COLORS.get(mode, 'gray')
        
        lines = []
        for i in range(len(seg['nodes']) - 1):
            u, v = seg['nodes'][i], seg['nodes'][i + 1]
            if u in node_coords and v in node_coords:
                lines.append([
                    (node_coords[u]['x'], node_coords[u]['y']),
                    (node_coords[v]['x'], node_coords[v]['y'])
                ])
        
        if lines:
            lw = 5 if mode == 'metro' else 3
            lc = LineCollection(lines, colors=color, linewidths=lw, alpha=0.9, 
                              label=f"{mode.upper()}: {seg['time_s']/60:.1f} min")
            ax.add_collection(lc)
        
        # Plot segment nodes
        seg_coords = [(node_coords[n]['x'], node_coords[n]['y']) 
                      for n in seg['nodes'] if n in node_coords]
        if seg_coords:
            seg_xs, seg_ys = zip(*seg_coords)
            ax.scatter(seg_xs, seg_ys, c=color, s=20, edgecolors='white', 
                      linewidths=0.5, zorder=3)
    
    # Plot transfer points
    for t in route['transfers']:
        if t['from_node'] in node_coords:
            x, y = node_coords[t['from_node']]['x'], node_coords[t['from_node']]['y']
            ax.scatter(x, y, c='#F39C12', s=200, marker='D', edgecolors='black', 
                      linewidths=2, zorder=5)
            ax.annotate(t['transfer_type'].replace('_', '\n'), (x, y),
                       fontsize=7, ha='center', va='bottom',
                       xytext=(0, 15), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Plot start and end
    start_node = route['path'][0]
    end_node = route['path'][-1]
    
    if start_node in node_coords:
        x, y = node_coords[start_node]['x'], node_coords[start_node]['y']
        ax.scatter(x, y, c='#2ECC71', s=300, marker='o', edgecolors='black', 
                  linewidths=3, zorder=6, label='START')
        ax.annotate('START', (x, y), fontsize=10, fontweight='bold',
                   xytext=(-10, 15), textcoords='offset points',
                   color='green')
    
    if end_node in node_coords:
        x, y = node_coords[end_node]['x'], node_coords[end_node]['y']
        ax.scatter(x, y, c='#E74C3C', s=400, marker='*', edgecolors='black', 
                  linewidths=3, zorder=6, label='END')
        ax.annotate('END', (x, y), fontsize=10, fontweight='bold',
                   xytext=(-10, 15), textcoords='offset points',
                   color='red')
    
    # Plot metro stations along route
    metro_in_route = [n for n in route['path'] if n in node_coords 
                      and node_coords[n]['layer'] == 1]
    for metro_id in metro_in_route:
        metro_node = nodes_df[nodes_df['node_id'] == metro_id]
        if len(metro_node) > 0:
            x, y = metro_node.iloc[0]['x'], metro_node.iloc[0]['y']
            name = metro_node.iloc[0]['name']
            ax.scatter(x, y, c='#27AE60', s=150, marker='s', edgecolors='black', 
                      linewidths=2, zorder=5)
            if pd.notna(name):
                ax.annotate(name, (x, y), fontsize=8, 
                           xytext=(5, -10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f"{title}\nTime: {route['total_time_s']/60:.1f} min | "
                f"Distance: {route['total_distance_m']/1000:.2f} km | "
                f"Transfers: {route['num_transfers']}", 
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax


def main():
    print("=" * 60)
    print("MULTIMODAL DIJKSTRA ROUTING TEST")
    print("=" * 60)
    
    # Load the network
    print("\nLoading network...")
    mlg = MultiLayerGraph(
        'data/final/nodes_final.csv',
        'data/final/edges_final.csv',
        'data/final/transfers_final.csv'
    )
    
    # Create router
    router = MultimodalRouter(mlg)
    
    # Get sample nodes
    road_nodes = mlg.nodes_df[mlg.nodes_df['layer'] == 0]['node_id'].tolist()
    metro_nodes = mlg.nodes_df[mlg.nodes_df['layer'] == 1]['node_id'].tolist()
    walk_nodes = mlg.nodes_df[mlg.nodes_df['layer'] == 2]['node_id'].tolist()
    
    # Find specific metro stations
    rajiv_chowk = mlg.nodes_df[mlg.nodes_df['name'].str.contains('Rajiv Chowk', case=False, na=False)]
    kashmere_gate = mlg.nodes_df[mlg.nodes_df['name'].str.contains('Kashmere Gate', case=False, na=False)]
    hauz_khas = mlg.nodes_df[mlg.nodes_df['name'].str.contains('Hauz Khas', case=False, na=False)]
    
    rajiv_chowk_id = rajiv_chowk.iloc[0]['node_id'] if len(rajiv_chowk) > 0 else None
    kashmere_gate_id = kashmere_gate.iloc[0]['node_id'] if len(kashmere_gate) > 0 else None
    hauz_khas_id = hauz_khas.iloc[0]['node_id'] if len(hauz_khas) > 0 else None
    
    print(f"\nMetro Stations Found:")
    print(f"  Rajiv Chowk: {rajiv_chowk_id}")
    print(f"  Kashmere Gate: {kashmere_gate_id}")
    print(f"  Hauz Khas: {hauz_khas_id}")
    
    # =====================================================
    # TEST 1: Road to Road (far apart - should use metro)
    # =====================================================
    print("\n" + "=" * 60)
    print("TEST 1: Road to Road (Far Apart - Should Use Metro)")
    print("=" * 60)
    
    source = road_nodes[100]
    target = road_nodes[55000]
    print(f"\nSource: {source}")
    print(f"Target: {target}")
    
    route1 = router.find_route(source, target)
    print_route_details(route1)
    
    # =====================================================
    # TEST 2: Road to Metro Station (Rajiv Chowk)
    # =====================================================
    print("\n" + "=" * 60)
    print("TEST 2: Road to Rajiv Chowk Metro Station")
    print("=" * 60)
    
    source = road_nodes[500]
    target = rajiv_chowk_id
    print(f"\nSource: {source}")
    print(f"Target: {target}")
    
    route2 = router.find_route(source, target)
    print_route_details(route2)
    
    # =====================================================
    # TEST 3: Metro to Metro (Different Lines)
    # =====================================================
    print("\n" + "=" * 60)
    print("TEST 3: Rajiv Chowk to Hauz Khas (Blue/Yellow to Magenta)")
    print("=" * 60)
    
    route3 = None
    if rajiv_chowk_id and hauz_khas_id:
        source = rajiv_chowk_id
        target = hauz_khas_id
        print(f"\nSource: Rajiv Chowk ({source})")
        print(f"Target: Hauz Khas ({target})")
        
        route3 = router.find_route(source, target)
        print_route_details(route3)
    
    # =====================================================
    # TEST 4: Walk to Metro to Walk
    # =====================================================
    print("\n" + "=" * 60)
    print("TEST 4: Walk Layer -> Metro -> Walk Layer")
    print("=" * 60)
    
    source = walk_nodes[100]
    target = walk_nodes[50000]
    print(f"\nSource: {source}")
    print(f"Target: {target}")
    
    route4 = router.find_route(source, target)
    print_route_details(route4)
    
    # =====================================================
    # TEST 5: Full Multimodal (Road -> Walk -> Metro -> Walk -> Road)
    # =====================================================
    print("\n" + "=" * 60)
    print("TEST 5: Full Multimodal Journey")
    print("=" * 60)
    
    # Pick nodes that are far apart to encourage multimodal routing
    source = road_nodes[200]
    target = road_nodes[60000]
    print(f"\nSource: {source}")
    print(f"Target: {target}")
    
    route5 = router.find_route(source, target, transfer_penalty=30)  # Add 30s penalty per transfer
    print_route_details(route5)
    
    # Check which layers were used
    if route5:
        layers_used = set()
        for seg in route5['segments']:
            if seg['mode'] == 'car':
                layers_used.add('Road')
            elif seg['mode'] == 'metro':
                layers_used.add('Metro')
            elif seg['mode'] == 'walk':
                layers_used.add('Walk')
        
        print(f"\n  Layers Used: {', '.join(sorted(layers_used))}")
        
        if len(layers_used) > 1:
            print("  ✓ MULTIMODAL ROUTING SUCCESSFUL!")
        else:
            print("  Single mode route (multimodal not needed for this distance)")
    
    print("\n" + "=" * 60)
    print("VISUALIZING ROUTES")
    print("=" * 60)
    
    # Visualize all routes
    routes = [
        (route1, "Test 1: Road to Road (via Metro)"),
        (route2, "Test 2: Road to Rajiv Chowk Metro"),
        (route3, "Test 3: Rajiv Chowk to Hauz Khas"),
        (route5, "Test 5: Full Multimodal Journey"),
    ]
    
    for route, title in routes:
        if route:
            print(f"\nShowing: {title}")
            fig, ax = visualize_route(route, mlg.nodes_df, title=title)
            if fig:
                plt.show()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
