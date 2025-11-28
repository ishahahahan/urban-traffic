# Multi-Layer Urban Transportation Network Optimization

**Shiv Nadar University - CSE Department**  
**Students:** Aarush Roy, Atharva Parashar, Ishan Das  
**Supervisor:** Dr. Rajeev Kumar Singh

## Project Overview

This project implements an intelligent urban traffic optimization system that suggests optimal **multimodal routes** combining multiple transportation modes (car, metro, walking, auto-rickshaw) based on real-time and historical traffic patterns.

### Key Innovation

Unlike traditional single-mode routing, this system treats urban transportation as a **multi-layer graph** where:
- Each transport mode is a separate layer
- Layers are connected through transfer nodes
- Dynamic edge weights reflect real-time traffic
- Routing considers both travel time and transfer penalties

## Features

✅ **Multi-Layer Network Architecture**
- Layer 0: Road network (car/auto)
- Layer 1: Metro/subway network
- Layer 2: Walking network

✅ **Dynamic Traffic Simulation**
- Time-dependent congestion patterns
- Rush hour modeling
- Mode-specific traffic characteristics

✅ **Intelligent Multimodal Routing**
- Modified Dijkstra's algorithm for multi-layer graphs
- Transfer optimization
- K-shortest alternative routes
- Single-mode vs multimodal comparison

✅ **Comprehensive Visualization**
- Network layer visualization
- Route visualization with segments
- Traffic heatmaps
- Time-series analysis

## Installation

### Requirements

```bash
# Install dependencies
pip install pandas numpy networkx osmnx scipy matplotlib tqdm pyarrow
```

### Project Structure

```
urban-traffic/
├── data/
│   ├── raw/                    # Original OSMnx data
│   │   ├── nodes.csv
│   │   └── edges.csv
│   ├── multilayer/             # Intermediate layer files
│   └── final/                  # Final output
│       ├── nodes_final.csv
│       ├── edges_final.csv
│       ├── transfers_final.csv
│       ├── multimodal_timeseries.parquet
│       └── network_summary.json
├── src/
│   ├── data_processing/
│   │   ├── convert_road_layer.py
│   │   ├── extract_metro_layer.py
│   │   ├── create_walking_layer.py
│   │   ├── create_transfers.py
│   │   ├── merge_all_layers.py
│   │   └── generate_multimodal_traffic.py
│   ├── routing/
│   │   ├── graph_loader.py
│   │   └── multimodal_dijkstra.py
│   ├── visualization/
│   │   └── plot_network.py
│   └── utils/
│       ├── haversine.py
│       └── config.py
├── build_multilayer_network.py  # Main pipeline script
└── README.md
```

## Quick Start

### 1. Build Multi-Layer Network

Run the complete pipeline to process raw data and build the multi-layer network:

```bash
python build_multilayer_network.py
```

This will:
1. Convert road network to multi-layer format
2. Create synthetic metro network
3. Generate walking layer
4. Create transfer connections
5. Merge all layers
6. Generate time-series traffic data

**Output:** Complete multi-layer network in `data/final/`

### 2. Run Multimodal Routing

```python
from src.routing.graph_loader import MultiLayerGraph
from src.routing.multimodal_dijkstra import MultimodalRouter

# Load network
mlg = MultiLayerGraph(
    nodes_file='data/final/nodes_final.csv',
    edges_file='data/final/edges_final.csv',
    transfers_file='data/final/transfers_final.csv',
    timeseries_file='data/final/multimodal_timeseries.parquet'
)

# Create router
router = MultimodalRouter(mlg)

# Find route
route = router.find_route(
    source='road_12345',
    target='road_67890',
    timestamp='2025-11-25 08:00:00',  # Morning rush hour
    transfer_penalty=60  # 1 minute penalty per transfer
)

# Display results
if route:
    print(f"Total time: {route['total_time_s']/60:.1f} minutes")
    print(f"Transfers: {route['num_transfers']}")
    
    for i, seg in enumerate(route['segments']):
        print(f"{i+1}. {seg['mode']}: {seg['time_s']/60:.1f} min")
```

### 3. Visualize Network

```python
from src.visualization.plot_network import plot_network_layers
import pandas as pd
import matplotlib.pyplot as plt

# Load data
nodes = pd.read_csv('data/final/nodes_final.csv')
edges = pd.read_csv('data/final/edges_final.csv')

# Plot all layers
fig, ax = plot_network_layers(nodes, edges, show_layers=[0, 1, 2])
plt.savefig('network_visualization.png', dpi=300)
plt.show()
```

## Data Schema

### Nodes (nodes_final.csv)

| Column | Type | Description |
|--------|------|-------------|
| node_id | str | Unique identifier (format: "layer_osmid") |
| layer | int | Layer number (0=road, 1=metro, 2=walk) |
| osmid | str/null | OpenStreetMap ID |
| x | float | Longitude (WGS84) |
| y | float | Latitude (WGS84) |
| node_type | str | Node classification |
| name | str | Human-readable name |
| properties | JSON | Additional attributes |

### Edges (edges_final.csv)

| Column | Type | Description |
|--------|------|-------------|
| edge_id | str | Unique identifier |
| layer | int | Layer number |
| u | str | Source node_id |
| v | str | Target node_id |
| key | int | Edge key (for parallel edges) |
| length_m | float | Physical length (meters) |
| mode | str | Transport mode |
| speed_mps | float | Base speed (m/s) |
| travel_time_s | float | Base travel time (seconds) |
| edge_type | str | Edge classification |
| properties | JSON | Mode-specific attributes |

### Transfers (transfers_final.csv)

| Column | Type | Description |
|--------|------|-------------|
| transfer_id | str | Unique identifier |
| from_node | str | Source node_id |
| to_node | str | Target node_id |
| from_layer | int | Source layer |
| to_layer | int | Destination layer |
| transfer_time_s | float | Transfer time (seconds) |
| transfer_type | str | Type of transfer |
| distance_m | float | Walking distance |
| properties | JSON | Transfer attributes |

### Time-Series (multimodal_timeseries.parquet)

| Column | Type | Description |
|--------|------|-------------|
| edge_id | str | Edge identifier |
| timestamp | datetime | Time of measurement |
| hour | int | Hour component (0-23) |
| mode | str | Transport mode |
| layer | int | Layer number |
| current_speed_mps | float | Actual speed |
| travel_time_s | float | Actual travel time |
| congestion_factor | float | Congestion multiplier |

## Algorithm Details

### Multimodal Dijkstra's Algorithm

The routing algorithm extends standard Dijkstra's to handle:

1. **Multiple Layers**: Separate graphs per transport mode
2. **Transfer Edges**: Special edges connecting layers with transfer penalties
3. **Time-Dependent Weights**: Edge weights updated based on timestamp
4. **Mode Preferences**: Optional penalties to discourage transfers

**Time Complexity**: O((E + T) log V) where:
- V = total nodes across all layers
- E = total edges across all layers
- T = number of transfer edges

### Traffic Congestion Model

Time-of-day multipliers vary by mode:

**Cars (Layer 0):**
- Night (00:00-05:00): 0.9× (free flow)
- Morning rush (08:00-10:00): 1.6× (heavy congestion)
- Evening rush (16:00-19:00): 1.7× (heaviest congestion)

**Metro (Layer 1):**
- Off-peak: 1.0× (on schedule)
- Rush hours: 1.1-1.15× (crowding delays)

**Walking (Layer 2):**
- Most times: 1.0×
- Rush hours: 1.05× (crowded sidewalks)

## Results & Evaluation

### Typical Performance

**Morning Rush Hour (08:00):**
- Single-mode (car): 45 minutes
- Multimodal (car → metro → walk): 32 minutes
- **Time savings: 28.9%**

**Off-Peak (14:00):**
- Single-mode: 28 minutes
- Multimodal: 26 minutes
- **Time savings: 7.1%**

### Network Statistics

```json
{
  "total_nodes": 1350,
  "total_edges": 3600,
  "total_transfers": 2800,
  "layers": {
    "road": {"nodes": 450, "edges": 1200},
    "metro": {"nodes": 7, "edges": 12},
    "walk": {"nodes": 450, "edges": 893}
  }
}
```

## Future Enhancements

### Phase 2 (Planned)
- [ ] Real-time traffic API integration
- [ ] Bus network layer
- [ ] GNN-based traffic prediction
- [ ] Multi-objective optimization (time + cost + comfort)
- [ ] Mobile application interface

### Phase 3 (Research)
- [ ] Reinforcement learning for route recommendation
- [ ] Predictive congestion modeling
- [ ] Carbon footprint optimization
- [ ] Accessibility-aware routing

## Running Individual Components

### Convert Road Layer Only
```bash
python src/data_processing/convert_road_layer.py
```

### Create Metro Network Only
```bash
python src/data_processing/extract_metro_layer.py
```

### Generate Traffic Data Only
```bash
python src/data_processing/generate_multimodal_traffic.py
```

## Troubleshooting

### Memory Error During Traffic Generation

If you encounter memory errors:

1. Reduce `chunk_size` in `generate_multimodal_traffic.py`:
```python
chunk_size=5000  # Default is 10000
```

2. Reduce time resolution:
```python
steps_per_hour=6  # 10-minute intervals instead of 5
```

### No Transfers Created

If transfer creation returns 0 transfers:

1. Increase `max_transfer_distance`:
```python
max_transfer_distance=500  # Default is 300m
```

2. Check metro station coordinates are within road network bounds

### Route Not Found

If routing fails:

1. Verify source and target nodes exist:
```python
print(source in mlg.graph.nodes())
print(target in mlg.graph.nodes())
```

2. Check if nodes are in same connected component
3. Ensure transfers are properly loaded

## Citation

If you use this code in your research, please cite:

```bibtex
@project{multilayer-traffic-2025,
  title={Multi-Layer Urban Transportation Network Optimization},
  author={Roy, Aarush and Parashar, Atharva and Das, Ishan},
  institution={Shiv Nadar University},
  supervisor={Singh, Rajeev Kumar},
  year={2025}
}
```

## License

This project is developed for academic purposes at Shiv Nadar University.

## Contact

- **Aarush Roy** - [email]
- **Atharva Parashar** - [email]
- **Ishan Das** - [email]

**Supervisor:** Dr. Rajeev Kumar Singh  
**Department:** Computer Science and Engineering  
**Institution:** Shiv Nadar University
